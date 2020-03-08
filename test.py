import tensorflow as tf
import model
from config import *
import os

test_file = 'test/fe7e27d0ef8c8a33312bf524bd7f5a7d.jpg'
save_file = 'testp/%d.png' % (len(os.listdir('testp')) + 1)

print('正在加载图片')

img = tf.io.read_file(test_file)
img = tf.image.decode_image(img, channels=3)
img = tf.image.convert_image_dtype(img, tf.float32)
img = tf.image.rgb_to_yuv(img)

gray, uv = tf.split(img, [1, 2], axis=2)

print('正在加载模型')

gen = model.Generator(input_shape=gray.shape)
gen.load_weights(GenWeightPath)

print('正在预测...')

pred = gen(tf.expand_dims(gray, axis=0))[0]
pred_img = tf.concat([gray, pred], axis=2)
pred_img = tf.image.yuv_to_rgb(pred_img)
pred_img = tf.clip_by_value(pred_img, 0., 1.)
pred_img = tf.image.convert_image_dtype(pred_img, tf.uint8)
pred_data = tf.image.encode_png(pred_img)

tf.io.write_file(save_file, pred_data)
