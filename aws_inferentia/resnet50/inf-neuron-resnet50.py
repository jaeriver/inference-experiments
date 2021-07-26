#!/usr/bin/env python
# coding: utf-8

# # AWS Inferentia inference on Amazon EC2 Inf1 instance
# This example demonstrates AWS Inferentia inference with TensorFlow and AWS Neuron SDK compiler and runtime
# 
# This example was tested on Amazon EC2 `inf1.xlarge` the following AWS Deep Learning AMI: 
# `Deep Learning AMI (Ubuntu 18.04) Version 35.0`
# 
# Run this notebook using the following conda environment:
# `aws_neuron_tensorflow_p36`
# 
# Prepare your imagenet validation TFRecord files using the following helper script:
# https://github.com/tensorflow/models/blob/archive/research/inception/inception/data/download_and_preprocess_imagenet.sh
# 
# Save it to `/home/ubuntu/datasets/` or update the dataset location in the `get_dataset()` function

# In[2]:


# !pip install matplotlib pandas


# In[3]:


get_ipython().system('/opt/aws/neuron/bin/neuron-cli reset')
import os
import time
import shutil
import json
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.neuron as tfn
import tensorflow.compat.v1.keras as keras
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from concurrent import futures
from itertools import compress

print('test')


# In[4]:


# https://github.com/tensorflow/tensorflow/issues/29931
temp = tf.zeros([8, 224, 224, 3])
_ = tf.keras.applications.resnet50.preprocess_input(temp)


# ### Resnet50 FP32 saved model

# In[5]:


# Export SavedModel
saved_model_dir = 'resnet50_saved_model'
shutil.rmtree(saved_model_dir, ignore_errors=True)

keras.backend.set_learning_phase(0)
model = ResNet50(weights='imagenet')
tf.saved_model.simple_save(session = keras.backend.get_session(),
                           export_dir = saved_model_dir,
                           inputs = {'input_1:0': model.inputs[0]},
                           outputs = {'probs/Softmax:0': model.outputs[0]})


# ### Compile models with different batch sizes and cores

# In[6]:


def compile_inf1_model(saved_model_dir, inf1_model_dir, batch_size=1, num_cores=1, use_static_weights=False):
    print(f'-----------batch size: {batch_size}, num cores: {num_cores}----------')
    print('Compiling...')
    
    compiled_model_dir = f'resnet50_batch_{batch_size}_inf1_cores_{num_cores}'
    inf1_compiled_model_dir = os.path.join(inf1_model_dir, compiled_model_dir)
    shutil.rmtree(inf1_compiled_model_dir, ignore_errors=True)

    example_input = np.zeros([batch_size,224,224,3], dtype='float32')

    compiler_args = ['--verbose','1', '--neuroncore-pipeline-cores', str(num_cores)]
    if use_static_weights:
        compiler_args.append('--static-weights')
    
    start_time = time.time()
    compiled_res = tfn.saved_model.compile(model_dir = saved_model_dir,
                            model_feed_dict={'input_1:0': example_input},
                            new_model_dir = inf1_compiled_model_dir,
                            dynamic_batch_size=True,
                            compiler_args = compiler_args)
    print(f'Compile time: {time.time() - start_time}')
    
    compile_success = False
    perc_on_inf = compiled_res['OnNeuronRatio'] * 100
    if perc_on_inf > 50:
        compile_success = True
            
    print(inf1_compiled_model_dir)
    print(compiled_res)
    print('----------- Done! ----------- \n')
    
    return compile_success


# ### Use `tf.data` to read ImageNet validation dataset

# In[7]:


def deserialize_image_record(record):
    feature_map = {'image/encoded': tf.io.FixedLenFeature([], tf.string, ''),
                  'image/class/label': tf.io.FixedLenFeature([1], tf.int64, -1),
                  'image/class/text': tf.io.FixedLenFeature([], tf.string, '')}
    obj = tf.io.parse_single_example(serialized=record, features=feature_map)
    imgdata = obj['image/encoded']
    label = tf.cast(obj['image/class/label'], tf.int32)   
    label_text = tf.cast(obj['image/class/text'], tf.string)   
    return imgdata, label, label_text

def val_preprocessing(record):
    imgdata, label, label_text = deserialize_image_record(record)
    label -= 1
    image = tf.io.decode_jpeg(imgdata, channels=3, 
                              fancy_upscaling=False, 
                              dct_method='INTEGER_FAST')

    shape = tf.shape(image)
    height = tf.cast(shape[0], tf.float32)
    width = tf.cast(shape[1], tf.float32)
    side = tf.cast(tf.convert_to_tensor(256, dtype=tf.int32), tf.float32)

    scale = tf.cond(tf.greater(height, width),
                  lambda: side / width,
                  lambda: side / height)
    
    new_height = tf.cast(tf.math.rint(height * scale), tf.int32)
    new_width = tf.cast(tf.math.rint(width * scale), tf.int32)
    
    image = tf.image.resize(image, [new_height, new_width], method='bicubic')
    image = tf.image.resize_with_crop_or_pad(image, 224, 224)
    
    image = tf.keras.applications.resnet50.preprocess_input(image)
    
    return image, label, label_text

def get_dataset(batch_size, use_cache=False):
    data_dir = '/home/ubuntu/datasets/*'
    files = tf.io.gfile.glob(os.path.join(data_dir))
    dataset = tf.data.TFRecordDataset(files)
    
    dataset = dataset.map(map_func=val_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat(count=1)
    
    if use_cache:
        shutil.rmtree('tfdatacache', ignore_errors=True)
        os.mkdir('tfdatacache')
        dataset = dataset.cache(f'./tfdatacache/imagenet_val')
    
    return dataset


# ## Single AWS Inferentia chip execution
# * Single core compiled models with automatic data parallel model upto 4 cores
# * Multi-core compiled models for pipeline execution

# In[8]:


def inf1_predict_benchmark_single_threaded(neuron_saved_model_name, batch_size, user_batch_size, num_cores, use_cache=False, warm_up=10):
    print(f'Running model {neuron_saved_model_name}, user_batch_size: {user_batch_size}\n')

    model_inf1 = tf.contrib.predictor.from_saved_model(neuron_saved_model_name)

    iter_times = []
    pred_labels = []
    actual_labels = []
    display_threshold = 0
    warm_up = 10

    ds = get_dataset(user_batch_size, use_cache)

    ds_iter = ds.make_initializable_iterator()
    ds_next = ds_iter.get_next()
    ds_init_op = ds_iter.initializer

    with tf.Session() as sess:
        if use_cache:
            sess.run(ds_init_op)
            print('\nCaching dataset ...')
            start_time = time.time()
            try:
                while True:
                    (validation_ds,label,_) = sess.run(ds_next)
            except tf.errors.OutOfRangeError:
                pass
            print(f'Caching finished: {time.time()-start_time} sec')  

        try:
            sess.run(ds_init_op)
            counter = 0
            
            total_datas = 1000
            display_every = 100
            display_threshold = display_every
            
            ipname = list(model_inf1.feed_tensors.keys())[0]
            resname = list(model_inf1.fetch_tensors.keys())[0]
            
            walltime_start = time.time()
            sess_time = []
            extend_time = []
            while True:
                sess_start = time.time()
                (validation_ds,batch_labels,_) = sess.run(ds_next)
                sess_time.append(time.time() - sess_start)
                
                model_feed_dict={ipname: validation_ds}

                if counter == 0:
                    for i in range(warm_up):
                        _ = model_inf1(model_feed_dict);                    

                start_time = time.time()
                inf1_results = model_inf1(model_feed_dict);
                iter_times.append(time.time() - start_time)
                
                extend_start = time.time()
                actual_labels.extend(label for label_list in batch_labels for label in label_list)
                pred_labels.extend(list(np.argmax(inf1_results[resname], axis=1)))
                extend_time.append(time.time() - extend_start)
                
                if counter*user_batch_size >= display_threshold:
                    print(f'Images {counter*user_batch_size}/{total_datas}. Average i/s {np.mean(user_batch_size/np.array(iter_times[-display_every:]))}')
                    display_threshold+=display_every

                counter+=1

        except tf.errors.OutOfRangeError:
            pass
    
    labeling_start = time.time()
    
    acc_inf1 = np.sum(np.array(actual_labels) == np.array(pred_labels))/len(actual_labels)
    iter_times = np.array(iter_times)
    labeling_time = time.time() - labeling_start
    
    results = pd.DataFrame(columns = [f'inf1_compiled_batch_size_{batch_size}_compiled_cores_{num_cores}'])
    results.loc['compiled_batch_size']     = [batch_size]
    results.loc['user_batch_size']         = [user_batch_size]
    results.loc['accuracy']                = [acc_inf1]
    results.loc['prediction_time']         = [np.sum(iter_times)]
    results.loc['sess_time']               = [np.sum(np.array(sess_time))]
    results.loc['extend_time']             = [np.sum(np.array(extend_time))]
    results.loc['labeling_time']           = [np.sum(np.array(labeling_time))]
    results.loc['wall_time']               = [time.time() - walltime_start]
    results.loc['images_per_sec_mean']     = [np.mean(user_batch_size / iter_times)]
    results.loc['images_per_sec_std']      = [np.std(user_batch_size / iter_times, ddof=1)]
    results.loc['latency_mean']            = [np.mean(iter_times) * 1000]
    results.loc['latency_99th_percentile'] = [np.percentile(iter_times, q=99, interpolation="lower") * 1000]
    results.loc['latency_median']          = [np.median(iter_times) * 1000]
    results.loc['latency_min']             = [np.min(iter_times) * 1000]
    display(results.T)
#     shutil.rmtree(neuron_saved_model_name, ignore_errors=True)

    return results, iter_times


# In[16]:


inf1_model_dir = 'resnet50_inf1_saved_models'
saved_model_dir = 'resnet50_saved_model'


# testing batch size
batch_list = [4]
num_of_cores = [1]

inf1_model_dir = 'resnet50_inf1_saved_models'

for batch_size in batch_list:
    iter_ds = pd.DataFrame()
    results = pd.DataFrame()
    for num_cores in num_of_cores:
        opt ={'batch_size': batch_size, 'num_cores': num_of_cores}
        compiled_model_dir = f'resnet50_batch_{batch_size}_inf1_cores_{num_cores}'
        inf1_compiled_model_dir = os.path.join(inf1_model_dir, compiled_model_dir)

        print(f'inf1_compiled_model_dir: {inf1_compiled_model_dir}')
        col_name = lambda opt: f'inf1_{batch_size}_multicores_{num_cores}'

        res, iter_times = inf1_predict_benchmark_single_threaded(inf1_compiled_model_dir,
                                                                         batch_size = batch_size,
                                                                         user_batch_size = batch_size*10,
                                                                         num_cores = num_cores,
                                                                         use_cache=False, 
                                                                         warm_up=10)

        iter_ds = pd.concat([iter_ds, pd.DataFrame(iter_times, columns=[col_name(opt)])], axis=1)
        results = pd.concat([results, res], axis=1)

    display(results)


# In[ ]:





# In[ ]:




