import os, re, time, json
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import resnet50
AUTOTUNE = tf.data.AUTOTUNE
print("Tensorflow version " + tf.__version__)

PROJECT = "jg-project-328708" #@param {type:"string"}
BUCKET = "gs://jg-tpubucket"  #@param {type:"string", default:"jddj"}
NEW_MODEL = True #@param {type:"boolean"}
MODEL_NAME = "resnet50" #@param {type:"string"}
MODEL_VERSION = "v1" #@param {type:"string"}

assert PROJECT, 'For this part, you need a GCP project. Head to http://console.cloud.google.com/ and create one.'
assert re.search(r'gs://.+', BUCKET), 'For this part, you need a GCS bucket. Head to http://console.cloud.google.com/storage and create one.'

# detect TPUs
#tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect('jg-tpu') # TPU detection
#strategy = tf.distribute.TPUStrategy(tpu)
#print("Number of accelerators: ", strategy.num_replicas_in_sync)



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

    label_text = tf.cast(label_text, tf.float32)
    label = tf.cast(label, tf.int32)
    image = resnet50.preprocess_input(image)
    image = tf.cast(image, tf.float32)
    return image, label, label_text

def get_dataset(batch_size, use_cache=False):
    data_dir = 'gs://jg-tpubucket/tf-record/validation-00000-of-00001'
    files = tf.io.gfile.glob(os.path.join(data_dir))
    dataset = tf.data.TFRecordDataset(files)
    
    dataset = dataset.map(map_func=val_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat(count=1)
    
    return dataset

def connect_to_tpu(tpu_address: str = None):
    if tpu_address is not None:  # When using GCP
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=tpu_address)
        if tpu_address not in ("", "local"):
            tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
        print("Running on TPU ", cluster_resolver.master())
        print("REPLICAS: ", strategy.num_replicas_in_sync)
        return cluster_resolver, strategy
    else:                           # When using Colab or Kaggle
        try:
            cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
            strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
            print("Running on TPU ", cluster_resolver.master())
            print("REPLICAS: ", strategy.num_replicas_in_sync)
            return cluster_resolver, strategy
        except:
            print("WARNING: No TPU detected.")
            mirrored_strategy = tf.distribute.MirroredStrategy()
            return None, mirrored_strategy
        
def tpu_inference(tpu_saved_model_name, batch_size):
    # Google TPU VM
    cluster_resolver, tpu_strategy = connect_to_tpu('jg-tpu')

    walltime_start = time.time()
    first_iter_time = 0
    iter_times = []
    pred_labels = []
    actual_labels = []
    display_threshold = 0
    ds = get_dataset(batch_size)
    
    tpu_saved_model_name = f'gs://jg-tpubucket/resnet50'
#   load_locally = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
    with tpu_strategy.scope():
        model_tpu = load_model(tpu_saved_model_name)
    yhat_np = model_tpu.predict(ds)
    print(yhat_np)

    iter_times = np.array(iter_times)
    acc_inf1 =''
    results = pd.DataFrame(columns = [f'tpu_{batch_size}'])
    results.loc['batch_size']              = [batch_size]
    results.loc['accuracy']                = [acc_inf1]
    results.loc['first_prediction_time']   = [first_iter_time]
    results.loc['average_prediction_time'] = [np.mean(iter_times)]
    results.loc['wall_time']               = [time.time() - walltime_start]

    return results, iter_times

batch_list = [8]
model_type = 'resnet50'

tpu_model = ''
for batch_size in batch_list:
  opt = {'batch_size': batch_size}
  iter_ds = pd.DataFrame()
  results = pd.DataFrame()
  res, iter_times = tpu_inference(tpu_model, batch_size)
  col_name = lambda opt: f'inf1_{batch_size}'
  iter_ds = pd.concat([iter_ds, pd.DataFrame(iter_times, columns=[col_name(opt)])], axis=1)
  results = pd.concat([results, res], axis=1)
  print(results)
   

