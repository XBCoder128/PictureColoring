import tensorflow as tf
from config import *
import utils
import model
import os
import time

dataset = utils.GetDataset(TrainDataPath, Epochs, BatchSize)

gen = model.Generator()
dis = model.Discriminator()

if os.path.exists(GenWeightPath):
    print('加载GEN权重中……')
    gen.load_weights(GenWeightPath)

if os.path.exists(DisWeightPath):
    print('加载DIS权重中……')
    dis.load_weights(DisWeightPath)

if not os.path.exists(PredPath):
    os.mkdir(PredPath)

generator_optimizer = tf.keras.optimizers.RMSprop()
discriminator_optimizer = tf.keras.optimizers.RMSprop()


@tf.function()
def train_step(input_data, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        gen_output = gen(input_data)

        dis_real_output = dis([input_data, target])
        dis_gene_output = dis([input_data, gen_output])

        tot_gen_loss, gen_loss, gen_l1_loss = model.generator_loss(dis_gene_output, gen_output, target)
        tot_dis_loss = model.discriminator_loss(dis_real_output, dis_gene_output)

    gen_gradients = gen_tape.gradient(tot_gen_loss, gen.trainable_variables)
    dis_gradients = dis_tape.gradient(tot_dis_loss, dis.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(gen_gradients, gen.trainable_variables)
    )

    discriminator_optimizer.apply_gradients(
        zip(dis_gradients, dis.trainable_variables)
    )

    return tot_gen_loss, gen_loss, gen_l1_loss, tot_dis_loss


start_time = time.time()
print('Strat...')
try:
    for step, data in enumerate(dataset):
        # yuv y   uv
        # batch, h, w, c
        input_data, targe = tf.split(data, [1, 2], 3)
        gan_tot_loss, gan_loss, gan_l1loss, dis_loss = train_step(input_data, targe)

        if step % 10 == 0:
            print('step: %d, Gen totle loss: %.3f, L1 loss: %.3f, Gen loss: %.3f, Dis totle loss: %.3f' % (
                step, gan_tot_loss.numpy(), gan_l1loss.numpy(), gan_loss.numpy(), dis_loss.numpy()))

        if step % 50 == 0:
            ytrue = data[0]
            ytrue = tf.image.yuv_to_rgb(ytrue)
            ytrue = tf.clip_by_value(ytrue, 0.0, 1.0)
            ytrue = tf.image.convert_image_dtype(ytrue, tf.uint8)
            pred = gen(input_data)[0]
            pred = tf.concat([input_data[0], pred], axis=2)
            pred = tf.image.yuv_to_rgb(pred)
            pred = tf.clip_by_value(pred, 0.0, 1.0)
            pred = tf.image.convert_image_dtype(pred, tf.uint8)
            tf.io.write_file(PredPath + '\output-%05d-t.png' % step, tf.image.encode_png(ytrue))
            tf.io.write_file(PredPath + '\output-%05d-p.png' % step, tf.image.encode_png(pred))

        if step % 1000 == 0:
            # step == 1000:………………
            gen.save_weights(GenWeightPath)
            dis.save_weights(DisWeightPath)

except KeyboardInterrupt:
    gen.save_weights(GenWeightPath)
    dis.save_weights(DisWeightPath)


used_time = (time.time() - start_time) // 60
print("用时：", used_time, '分钟')
gen.save_weights(GenWeightPath)
dis.save_weights(DisWeightPath)
print("训练完成，最终保存模型")
