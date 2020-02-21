import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import glob
from GAN import Generator, Discriminator
from dataset import make_anime_dataset
from datetime import datetime


def save_image(val_out, path):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        return img

    val_block_size = int(val_out.shape[0] ** 0.5)
    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b + 1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    Image.fromarray(final_image).save(path)


def celoss_ones(logits):
    # 计算属于与标签为1的交叉熵
    y = tf.ones_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    # 计算属于与便签为0的交叉熵
    y = tf.zeros_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)


# WGAN 判别器的损失函数计算与GAN 不一样，WGAN 是直接最大化真实样本的输出
# 值，最小化生成样本的输出值，并没有交叉熵计算的过程。代码实现如下：
def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    # 计算D 的损失函数
    fake_image = generator(batch_z, is_training)  # 假样本
    d_fake_logits = discriminator(fake_image, is_training)  # 假样本的输出
    d_real_logits = discriminator(batch_x, is_training)  # 真样本的输出
    # 计算梯度惩罚项
    gp = gradient_penalty(discriminator, batch_x, fake_image)
    # WGAN-GP D 损失函数的定义，这里并不是计算交叉熵，而是直接最大化正样本的输出
    # 最小化假样本的输出和梯度惩罚项
    loss = tf.reduce_mean(d_fake_logits) - tf.reduce_mean(d_real_logits) + 10. * gp
    return loss, gp


# WGAN 生成器G 的损失函数是只需要最大化生成样本在判别器D 的输出值即可，同
# 样没有交叉熵的计算步骤。代码实现如下：
def g_loss_fn(generator, discriminator, batch_z, is_training):
    # 生成器的损失函数
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    # WGAN-GP G 损失函数，最大化假样本的输出值
    loss = - tf.reduce_mean(d_fake_logits)
    return loss


def time_str():
    return datetime.now().isoformat()[:19]


# WGAN-GP 模型可以在原来GAN 代码实现的基础上仅做少量修改。WGAN-GP 模型
# 的判别器D 的输出不再是样本类别的概率，输出不需要加Sigmoid 激活函数。同时添加梯
# 度惩罚项，实现如下：
def gradient_penalty(discriminator, batch_x, fake_image):
    # 梯度惩罚项计算函数
    batchsz = batch_x.shape[0]
    # 每个样本均随机采样t,用于插值
    t = tf.random.uniform([batchsz, 1, 1, 1])
    # 自动扩展为x 的形状，[b, 1, 1, 1] => [b, h, w, c]
    t = tf.broadcast_to(t, batch_x.shape)
    # 在真假图片之间做线性插值
    interplate = t * batch_x + (1 - t) * fake_image
    # 在梯度环境中计算D 对插值样本的梯度
    with tf.GradientTape() as tape:
        tape.watch([interplate])  # 加入梯度观察列表
        d_interplote_logits = discriminator(interplate)
        grads = tape.gradient(d_interplote_logits, interplate)
    # 计算每个样本的梯度的范数:[b, h, w, c] => [b, -1]
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1)  # [b]
    # 计算梯度惩罚项
    gp = tf.reduce_mean((gp - 1.) ** 2)
    return gp


def main():
    tf.random.set_seed(3333)
    np.random.seed(3333)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    assert tf.__version__.startswith('2.')

    z_dim = 100  # 隐藏向量z的长度
    batch_size = 400  # batch size
    learning_rate = 0.0002
    is_training = True

    for f in os.listdir("images"):
        os.remove("images/" + f)

    # 获取数据集路径
    img_path = glob.glob(r'faces\*.jpg')
    # 构建数据集对象
    epochs = 100000 * len(img_path) // batch_size
    dataset, img_shape, _ = make_anime_dataset(img_path, batch_size, resize=64, shuffle=True, repeat=100000)
    sample = next(iter(dataset))  # 采样
    print("data shape: ", sample.shape, ", sample range: [",
          tf.reduce_max(sample).numpy(), ", ", tf.reduce_min(sample).numpy(), "]", sep='')
    db_iter = iter(dataset)

    generator = Generator()  # 创建生成器
    generator.build(input_shape=(4, z_dim))
    discriminator = Discriminator()  # 创建判别器
    discriminator.build(input_shape=(4, 64, 64, 3))

    # 分别为生成器和判别器创建优化器
    g_optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=0.5)
    d_optimizer = keras.optimizers.Adam(lr=learning_rate, beta_1=0.5)

    # generator.load_weights('model/generator.ckpt')
    # discriminator.load_weights('model/discriminator.ckpt')
    # print('Loaded models.')

    log = tf.summary.create_file_writer("logs/" + time_str().replace(":", '-'))

    d_losses, g_losses = [], []
    iterations_per_batch = 10
    for epoch in range(epochs):  # 训练epochs次
        # 1. 训练判别器
        for _ in range(iterations_per_batch):
            # 采样隐藏向量
            batch_z = tf.random.normal([batch_size, z_dim])
            batch_x = next(db_iter)  # 采样真实图片
            # 判别器前向计算
            with tf.GradientTape() as tape:
                d_loss, gp = d_loss_fn(generator, discriminator, batch_z, batch_x, is_training)
                grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        # 2. 训练生成器
        for _ in range(iterations_per_batch // 2):
            # 采样隐藏向量
            batch_z = tf.random.normal([batch_size, z_dim])
            # batch_x = next(db_iter)  # 采样真实图片
            # 生成器前向计算
            with tf.GradientTape() as tape:
                g_loss = g_loss_fn(generator, discriminator, batch_z, is_training)
            grads = tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch % 10 == 0:
            print(time_str(), epoch, 'd-loss: %.4f, g-loss: %.4f' % (float(d_loss), float(g_loss)))
            # 可视化
            z = tf.random.normal([batch_size, z_dim])
            fake_image = generator(z, training=False)
            save_image(fake_image.numpy(), "images/gan-%5d.png" % epoch)

            d_losses.append(float(d_loss))
            g_losses.append(float(g_loss))
            with log.as_default():
                tf.summary.scalar('g-loss', float(g_loss), step=epoch)
                tf.summary.scalar('d-loss', float(d_loss), step=epoch)

        if epoch % 99 == 0:
            generator.save_weights("model/generator.ckpt")
            discriminator.save_weights("model/discriminator.ckpt")


if __name__ == '__main__':
    main()
