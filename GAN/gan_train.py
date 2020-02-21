import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import glob
from GAN import Generator, Discriminator
from dataset import make_anime_dataset
import sys
import util
from BackGrad import BackGrad

def celoss_ones(logits):
    # 计算属于与标签为1的交叉熵
    y = tf.ones_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    # loss = keras.losses.mse(y, logits)
    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    # 计算属于与标签为0的交叉熵
    y = tf.zeros_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    # loss = keras.losses.mse(y, logits)
    return tf.reduce_mean(loss)


def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    # 计算判别器的误差函数
    # 采样生成图片
    fake_image = generator(batch_z, is_training)
    # 判定生成图片
    d_fake_logits = discriminator(fake_image, is_training)
    # 判定真实图片
    d_real_logits = discriminator(batch_x, is_training)
    # 真实图片与1之间的误差
    d_loss_real = celoss_ones(d_real_logits)
    # 生成图片与0之间的误差
    d_loss_fake = celoss_zeros(d_fake_logits)
    # 合并误差
    loss = d_loss_real + d_loss_fake
    return loss


def g_loss_fn(generator, discriminator, batch_z, is_training):
    # 采样生成图片
    fake_image = generator(batch_z, is_training)
    # 在训练生成网络时，需要迫使生成图片判定为真
    d_fake_logits = discriminator(fake_image, is_training)
    # 计算生成图片与1之间的误差
    loss = celoss_ones(d_fake_logits)
    return loss


def neg_grads(grads):
    grads_1 = []
    for grad in grads:
        grads_1.append(tf.negative(grad))
    return grads_1


def main():
    tf.random.set_seed(3333)
    np.random.seed(3333)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    assert tf.__version__.startswith('2.')

    z_dim = 100  # 隐藏向量z的长度
    batch_size = 100  # batch size
    learning_rate = 2e-4
    is_training = True
    last_batch = -1

    for f in os.listdir("images"):
        os.remove("images/" + f)

    # 获取数据集路径
    img_path = glob.glob(r'faces\*.jpg')
    # 构建数据集对象
    batches = 100000 * len(img_path) // batch_size
    dataset, img_shape, _ = make_anime_dataset(img_path, batch_size, resize=64, shuffle=True, repeat=100000)
    sample = next(iter(dataset))  # 采样
    print("data shape: ", sample.shape, ", sample range: [",
          tf.reduce_max(sample).numpy(), ", ", tf.reduce_min(sample).numpy(), "]", sep='')
    db_iter = iter(dataset)

    generator = Generator()  # 创建生成器
    generator.build(input_shape=(4, z_dim))
    discriminator = Discriminator()  # 创建判别器
    discriminator.build(input_shape=(4, 64, 64, 3))

    dirs = os.listdir(sys.path[0] + "\\model")
    if len(dirs) > 0:
        last_batch = int(dirs[-1])
        generator.load_weights(r'%s\model\%d\generator.ckpt' % (sys.path[0], last_batch))
        discriminator.load_weights(r'%s\model\%d\discriminator.ckpt' % (sys.path[0], last_batch))
        print('Loaded models of batch', last_batch)

    log = tf.summary.create_file_writer("logs/" + util.time_str().replace(":", '-'))

    # 分别为生成器和判别器创建优化器
    d_optimizer = tf.optimizers.Adam(lr=learning_rate, beta_1=0.999)
    g_optimizer = tf.optimizers.Adam(lr=learning_rate, beta_1=0.999)

    d_losses, g_losses = [], []
    for batch in range(last_batch + 1, batches):

        # 1. 训练判别器
        for _ in range(1):
            # 采样隐藏向量
            z = tf.random.normal([batch_size, z_dim])
            x = next(db_iter)  # 采样真实图片
            # 判别器前向计算
            with tf.GradientTape() as tape:
                d_loss = d_loss_fn(generator, discriminator, z, x, is_training)
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        # 2. 训练生成器
        # 采样隐藏向量
        for _ in range(5):
            z = tf.random.normal([batch_size, z_dim])
            # batch_x = next(db_iter)  # 采样真实图片
            # 生成器前向计算
            # for _ in range(iterations_per_batch):
            with tf.GradientTape() as tape:
                g_loss = g_loss_fn(generator, discriminator, z, is_training)
            grads = tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if batch % 100 == 0:
            print(util.time_str(), batch, 'd-loss=%.4f, g-loss=%.4f' % (float(d_loss), float(g_loss)))

            # 可视化
            z = tf.random.normal([batch_size, z_dim])
            fake_image = generator(z, training=False)
            util.save_image(fake_image.numpy(), "images/gan-%05d.png" % batch)

            d_losses.append(float(d_loss))
            g_losses.append(float(g_loss))
            # with log.as_default():
            #     tf.summary.scalar('g-loss', float(g_loss), step=batch)
            #     tf.summary.scalar('d-loss', float(d_loss), step=batch)

            # generator.save_weights("model/%d/generator.ckpt" % batch)
            # discriminator.save_weights("model/%d/discriminator.ckpt" % batch)
            # print("saved model weights of batch", batch)


if __name__ == '__main__':
    main()
