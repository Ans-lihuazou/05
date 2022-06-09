模拟训练代码

```
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader



image_path = tf.keras.utils.get_file(
    'flower_photos.tgz',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    extract=True)
image_path = os.path.join(os.path.dirname(image_path), 'flower_photos')

data = DataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)

inception_v3_spec = image_classifier.ModelSpec(
    uri='https://storage.googleapis.com/tfhub-modules/tensorflow/efficientnet/lite0/feature-vector/2.tar.gz')
inception_v3_spec.input_image_shape = [240, 240]
model = image_classifier.create(train_data, model_spec=inception_v3_spec)

loss, accuracy = model.evaluate(test_data)

model.export(export_dir='.')
```

将生成的文件在实验3程序中运行
郁金香:
![效果](1.jpg)
