#coding:utf-8
from tensorflow.examples.tutorials.mnist import input_data
from ops import *
import numpy as np
import random
import pickle

address = "/home/hanzijun/GAN/data_40_16/ls/11.pkl"
csi = pickle.load(open(address))
csi = [csi,[1,0,0,0,0]]

GESTURENUMBER = 5
img_height, img_width,img_depth = 16, 464, 1
beta, learning_rate = 0.5, 0.001
noise_size = 100
param_file = False

label = tf.placeholder(tf.float32, [BATCH_SIZE, GESTURENUMBER], name='label')
images = tf.placeholder(tf.float32, [BATCH_SIZE, img_height, img_width, img_depth], name='real_images')
noise = tf.placeholder(tf.float32, [BATCH_SIZE, noise_size], name='noise')
g_loss, d_loss = get_loss(images, noise, label)
g_vars, g_train_opt, d_train_opt, clip = get_optimizer(g_loss, d_loss, beta, learning_rate)


with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    if param_file:
        print "loading neural network params......"
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

    for epoch in range(5):
        for i in range(int(50/BATCH_SIZE)):
            batch = np.array(random.sample(csi, BATCH_SIZE))
            batch_images = batch[0]
            batch_images = batch_images.reshape((BATCH_SIZE, img_height, img_width, img_depth))
            batch_labels = batch[1]
            batch_noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, noise_size))

            _,_ = sess.run([d_train_opt, clip], feed_dict={images: batch_images, noise: batch_noise, label: batch_labels})
            _ = sess.run([g_train_opt], feed_dict={images: batch_images, noise: batch_noise, label: batch_labels})

            if i % 1 == 0:
                errD = d_loss.eval(feed_dict={images: batch_images, label: batch_labels, noise: batch_noise})
                errG = g_loss.eval({noise: batch_noise, label: batch_labels})
                print("epoch:[%d], i:[%d]  d_loss: %.8f, g_loss: %.8f" % (epoch, i, errD, errG))

            # if i % 100 == 1:
            #     sample_noise = np.random.uniform(-1, 1, size=(batch_noise, noise_size))
            #     samples = sess.run(generator(noise, label, training=False), feed_dict={noise: sample_noise, label:sample_labels})
            #
            #     samples_path = './pics/'
            #     save_images(samples, [8, 8], samples_path + 'epoch_%d_i_%d.png' % (epoch, i))
            #     print('save image')

            # if i == (int(55000/BATCH_SIZE)-1):
            #     checkpoint_path = os.path.join('./check_point/DCGAN_model.ckpt')
            #     saver.save(sess, checkpoint_path, global_step=i+1)
            #     print('save check_point')
