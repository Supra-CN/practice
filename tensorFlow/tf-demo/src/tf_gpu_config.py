import tensorflow as tf
import humanize

import os
tag = "demo_gpu_cfg"

def set_no_gpu():
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def set_memory_growth():
  os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
  print(f"{tag} :: env[TF_GPU_ALLOCATOR] => {os.getenv('TF_GPU_ALLOCATOR')}")

  gpus = tf.config.experimental.list_physical_devices('GPU')

  if gpus:
    try:
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        # tf.config.set_logical_device_configuration(
        #     gpu, [tf.config.experimental.VirtualDeviceConfiguration(
        #         memory_limit=100)])
        print(f"{tag} :: for gpu => {gpu}")
    except RuntimeError as e:
      print(f"{tag} :: except => {e}")