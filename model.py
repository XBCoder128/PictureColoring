import tensorflow as tf
import tensorflow.keras as keras
from config import *


def conv2d(input, filters, kernel_size, strides):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides, 'same')(input)
    x = tf.keras.layers.LeakyReLU()(x)
    return x


def res_net(input, filters):
    x = input
    net = conv2d(x, filters // 2, (1, 1), 1)
    net = conv2d(net, filters, (3, 3), 1)  # 由于是以same的方式填充，且步长为1，因此结果的长宽尺寸不变
    net = net + x
    return net


def upsample_layer(inputs, out_shape):
    new_height, new_width = out_shape[0], out_shape[1]
    inputs = tf.image.resize(inputs, (new_height, new_width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return inputs  # 使用最邻近采样，这样不会改变原有的特征信息


def Generator(input_shape=[net_input_size, net_input_size, 1]):
    x = tf.keras.Input(shape=input_shape)  # 假设输入尺寸是[n, 256, 256, 1]
    net_1 = conv2d(x, 64, (3, 3), strides=1)  # [n, 256, 256, 64]
    net_2 = conv2d(net_1, 64, (3, 3), strides=1)  # [n, 256, 256, 64]
    res_1 = res_net(net_2, 64)  # [n, 256, 256, 64]

    down_1 = conv2d(res_1, 128, (3, 3), strides=1)  # [n, 256, 256, 128]
    res_2 = res_net(down_1, 128)  # [n, 256, 256, 128]
    res_2 = keras.layers.MaxPooling2D(strides=2)(res_2)  # [n, 128, 128, 128]

    down_2 = conv2d(res_2, 256, (3, 3), strides=1)  # [n, 128, 128, 256]
    res_3 = res_net(down_2, 256)  # [n, 128, 128, 256]
    res_3 = keras.layers.MaxPooling2D(strides=2)(res_3)  # [n, 64, 64, 256]

    down_3 = conv2d(res_3, 512, (3, 3), strides=1)  # [ n, 64, 64, 512]
    res_4 = res_net(down_3, 512)  # [ n, 64, 64, 512]
    res_4 = keras.layers.MaxPooling2D(strides=2)(res_4)  # [ n, 32, 32, 512]

    down_4 = conv2d(res_4, 512, (3, 3), strides=1)  # [ n, 32, 32, 512]
    res_5 = res_net(down_4, 512)  # [ n, 32, 32, 512]
    res_5 = keras.layers.MaxPooling2D(strides=2)(res_5)  # [ n, 16, 16, 512]

    UPsamp_1 = tf.keras.layers.Conv2DTranspose(512, (3, 3), (2, 2), padding='same', activation='relu')(res_5)  # 8
    UPsamp_1 = upsample_layer(UPsamp_1, (res_4.shape[1], res_4.shape[2]))
    up_1 = tf.keras.layers.Concatenate(-1)([UPsamp_1, res_4])
    up_1 = tf.keras.layers.Conv2DTranspose(512, (3, 3), (1, 1), padding='same', activation='relu')(up_1)

    UPsamp_2 = tf.keras.layers.Conv2DTranspose(256, (3, 3), (2, 2), padding='same', activation='relu')(up_1)
    UPsamp_2 = upsample_layer(UPsamp_2, (res_3.shape[1], res_3.shape[2]))
    up_2 = tf.keras.layers.Concatenate(-1)([UPsamp_2, res_3])  # 16
    up_2 = tf.keras.layers.Conv2DTranspose(256, (3, 3), (1, 1), padding='same', activation='relu')(up_2)

    UPsamp_3 = tf.keras.layers.Conv2DTranspose(128, (3, 3), (2, 2), padding='same', activation='relu')(up_2)
    UPsamp_3 = upsample_layer(UPsamp_3, (res_2.shape[1], res_2.shape[2]))
    up_3 = tf.keras.layers.Concatenate(-1)([UPsamp_3, res_2])  # 32
    up_3 = tf.keras.layers.Conv2DTranspose(128, (3, 3), (1, 1), padding='same', activation='relu')(up_3)
    up_3 = res_net(up_3, 128)

    UPsamp_4 = tf.keras.layers.Conv2DTranspose(64, (3, 3), (2, 2), padding='same', activation='relu')(up_3)
    UPsamp_4 = upsample_layer(UPsamp_4, (res_1.shape[1], res_1.shape[2]))
    up_4 = tf.keras.layers.Concatenate(-1)([UPsamp_4, res_1])  # 64
    up_4 = tf.keras.layers.Conv2DTranspose(64, (3, 3), (1, 1), padding='same', activation='relu')(up_4)
    up_4 = res_net(up_4, 64)

    out = tf.keras.layers.Conv2D(OutputChannels, (3, 3), (1, 1), padding='same', activation='tanh')(up_4)
    out *= 0.5
    return keras.Model(inputs=x, outputs=out)


def Discriminator():
    ipt = keras.Input(shape=[net_input_size, net_input_size, 1], name='input_image')
    tar = keras.Input(shape=[net_input_size, net_input_size, 2], name='target_image')

    x = keras.layers.concatenate([ipt, tar])
    down0 = conv2d(x, 64, (3, 3), strides=1)

    down1 = res_net(down0, 64)
    down1 = conv2d(down1, 128, (3, 3), strides=2)  # 128*128
    down2 = res_net(down1, 128)
    down2 = conv2d(down2, 256, (3, 3), strides=2)  # 64 * 64
    down3 = res_net(down2, 256)
    down3 = keras.layers.MaxPooling2D(strides=2)(down3)  # 32 * 32

    zero_pad = keras.layers.ZeroPadding2D()(down3)  # [bs, 34, 34, 256]

    conv = keras.layers.Conv2D(512, 3, strides=1,
                               use_bias=False)(zero_pad)  # valid填充
    # [bs, 32, 32, 256]

    relu = keras.layers.LeakyReLU()(conv)

    last = keras.layers.Conv2D(1, 3, strides=1)(relu)
    # [bs, 30, 30, 1]
    return keras.Model(inputs=[ipt, tar], outputs=last)


loss_object = keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(disc_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_output), disc_output)
    l1_loss = 1000 * tf.reduce_mean(tf.abs(target - gen_output))
    tot_gen_loss = gan_loss + l1_loss
    return tot_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_gene_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    gene_loss = loss_object(tf.zeros_like(disc_gene_output), disc_gene_output)

    tot_disc_loss = real_loss + gene_loss
    return tot_disc_loss


model = Generator()
