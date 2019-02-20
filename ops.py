#coding:utf-8
import tensorflow as tf
import scipy.misc
import numpy as np

BATCH_SIZE = 1


def get_loss(image, noise, label):

    g_outputs = generator(noise, label, training=True)
    d_logits_real = discriminator(image, label)
    d_logits_fake = discriminator(g_outputs, label, reuse=True)
    #  Wasserstein distance
    d_loss_real = -tf.reduce_mean(d_logits_real)
    d_loss_fake = tf.reduce_mean(d_logits_fake)
    d_loss = tf.add(d_loss_real, d_loss_fake)
    g_loss = -d_loss_fake

    return g_loss, d_loss


def get_optimizer(g_loss, d_loss, beta1=0.5, learning_rate=0.001):

    train_vars = tf.trainable_variables()
    g_vars = [var for var in train_vars if var.name.startswith("generator")]
    d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        g_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)
        d_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
    # interpret the range  tof weights value
    clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]

    return g_vars, g_opt, d_opt, clip_D


def weight_variable(shape, name, stddev=0.02, trainable=True):
    dtype = tf.float32
    var = tf.get_variable(name, shape, tf.float32, trainable=trainable,
                          initializer=tf.random_normal_initializer(stddev=stddev, dtype=dtype))
    return var


def bias_variable(shape, name, bias_start=0.0, trainable = True):
    dtype = tf.float32
    var = tf.get_variable(name, shape, tf.float32, trainable=trainable,
                          initializer=tf.constant_initializer(bias_start, dtype=dtype))
    return var

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def conv_cond_concat(value, cond):
    # value BATCH_SIZE*28*28*1 , cond: BATCH_SIZE*1*1*10
    value_shapes = value.get_shape().as_list()
    cond_shapes = cond.get_shape().as_list()
    return tf.concat([value, cond * tf.ones(value_shapes[0:3] + cond_shapes[3:])], 3)

#  定义生成器，z:?*100, y:?*10
def generator(z, y, training=True):
    with tf.variable_scope("generator", reuse=not training):
        yb = tf.reshape(y, [BATCH_SIZE, 1, 1, 5], name="yb")  # y:?*1*1*10
        z = tf.concat([z, y], 1)  # z:?*110

        channels_in= z.get_shape().as_list()[1]
        w_fc1 = weight_variable([channels_in, 2 * 58 * 256], name='weight_1')
        b_fc1 = bias_variable([2 * 58 * 256,], name='biase_1')
        # w_fc1 = tf.get_variable('w1', [z[1], 2 * 58 * 256], initializer=tf.random_normal_initializer(stddev=0.02))
        # b_fc1 = tf.get_variable('b1', [2 * 58 * 256 ], initializer=tf.constant_initializer(0.1))

        h1 =  tf.matmul(z, w_fc1) + b_fc1
        h1 = tf.nn.relu(tf.layers.batch_normalization(h1, training=training, name='g_h1_batch_norm'))
        h1 = tf.reshape(h1, [-1, 2, 58, 256])
        h1 = conv_cond_concat(h1, yb)

        h2 = tf.layers.conv2d_transpose(h1, 128, 3, strides=2, padding='same')
        h2 = tf.nn.relu(tf.layers.batch_normalization(h2, training=training, name='g_h2_batch_norm'))  # h3: ?*14*14*128
        h2 = conv_cond_concat(h2, yb)

        h3 = tf.layers.conv2d_transpose(h2, 64, 3, strides=2, padding='same')
        h3 = tf.nn.relu(tf.layers.batch_normalization(h3, training=training, name='g_h3_batch_norm'))  # h3: ?*14*14*128
        h3 = conv_cond_concat(h3, yb)

        h4 = tf.layers.conv2d_transpose(h3, 1, 3, strides=2, padding='same', )
        h4 = tf.tanh(h4)
        return h4


def discriminator(image, y, reuse=False, training=True):
    with tf.variable_scope("discriminator", reuse=reuse):

        yb = tf.reshape(y, [BATCH_SIZE, 1, 1, 5], name='yb')  # BATCH_SIZE*1*1*10
        x = conv_cond_concat(image, yb)  # image: BATCH_SIZE*28*28*1 ,x: BATCH_SIZE*28*28*11

        h1 = tf.layers.conv2d(x, 64, 3, strides=2, padding='same')
        h1 = lrelu(tf.layers.batch_normalization(h1, name='d_h1_batch_norm', training=training, reuse=reuse))  # h1: BATCH_SIZE*14*14*11
        h1 = conv_cond_concat(h1, yb)  # h1: BATCH_SIZE*14*14*21

        h2 = tf.layers.conv2d(h1, 128, 3, strides=2, padding='same')
        h2 = lrelu(tf.layers.batch_normalization(h2, name='d_h2_batch_norm', training=training, reuse=reuse))  # h1: BATCH_SIZE*14*14*11
        h2 = conv_cond_concat(h2, yb)  # h1: BATCH_SIZE*14*14*21

        h3 = tf.layers.conv2d(h2, 256, 3, strides=2, padding='same')
        h3 = lrelu(tf.layers.batch_normalization(h3, name='d_h3_batch_norm', training=training, reuse=reuse))  # BATCH_SIZE*7*7*74
        h3 = tf.reshape(h3, [BATCH_SIZE, -1])  # BATCH_SIZE*3626
        h3 = tf.concat([h3, y], 1)  # BATCH_SIZE*3636

        channels_in = h3.get_shape().as_list()[1]
        w_fc1 = weight_variable([channels_in,  2 * 58 * 256], name='weight_1')
        b_fc1 = bias_variable([2 * 58 * 256], name='biase_1')
        h3 = tf.matmul(h3, w_fc1) + b_fc1
        # h3 = lrelu(tf.layers.batch_normalization(h3, name='d_h4_batch_norm', training=training, reuse=reuse))  # BATCH_SIZE*1024
        # h3 = tf.concat([h3, y], 1)  # BATCH_SIZE*1034
        #
        # w_fc2 = weight_variable([h3[1], 1], name='weight_2')
        # b_fc2 = bias_variable([1], name='biase_2')
        # h4 = tf.matmul(h3, w_fc2) + b_fc2
        h4 = tf.nn.sigmoid(h3)
        return h4

def save_images(images, size, path):
    # 图片归一化，主要用于生成器输出是 tanh 形式的归一化
    img = (images + 1.0) / 2.0
    h, w = img.shape[1], img.shape[2]
    # 产生一个大画布，用来保存生成的 batch_size 个图像
    merge_img = np.zeros((h * size[0], w * size[1], 3))
    # 循环使得画布特定地方值为某一幅图像的值
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        if j >= size[0]:
            break
        merge_img[j * h:j * h + h, i * w:i * w + w, :] = image
    # 保存画布
    return scipy.misc.imsave(path, merge_img)
