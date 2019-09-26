import numpy as np
import tensorflow as tf
import os
import math
import time
from PIL import Image


# data=np.load('./vgg16_model.npy',allow_pickle=True,encoding='bytes')
# data_dic=data.item()
# # 查看网络层参数的键值
# print(data_dic.keys())
# # 查看卷积层1_1的参数w,b
# w,b=data_dic[b'conv1_1']
# print(w.shape,b.shape)          # (3, 3, 3, 64) (64,)
# # 查看全连接层的参数
# w,b=data_dic[b'fc8']
# print(w.shape,b.shape)          # (4096, 1000) (1000,)


class VGGNet:
    def __init__(self, data_dir):
        data = np.load(data_dir, allow_pickle=True, encoding='bytes')
        self.data_dic = data.item()

    def conv_layer(self, x, name):
        # 实现卷积操作
        with tf.name_scope(name):
            # 从模型文件中读取各卷积层的参数值
            weight = tf.constant(self.data_dic[name][0], name='conv')
            bias = tf.constant(self.data_dic[name][1], name='bias')
            # 进行卷积操作
            y = tf.nn.conv2d(x, weight, [1, 1, 1, 1], padding='SAME')
            y = tf.nn.bias_add(y, bias)
            return tf.nn.relu(y)

    def pooling_layer(self, x, name):
        # 实现池化操作
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def flatten_layer(self, x, name):
        # 实现展开层
        with tf.name_scope(name):
            # x_shape->[batch_size,image_width,image_height,chanel]
            x_shape = x.get_shape().as_list()
            dimension = 1
            # 计算x的最后三个维度积
            for d in x_shape[1:]:
                dimension *= d
            output = tf.reshape(x, [-1, dimension])
            return output

    def fc_layer(self, x, name, activation=tf.nn.relu):
        # 实现全连接层
        with tf.name_scope(name):
            # 从模型文件中读取各全连接层的参数值
            weight = tf.constant(self.data_dic[name][0], name='fc')
            bias = tf.constant(self.data_dic[name][1], name='bias')
            # 进行全连接操作
            y = tf.matmul(x, weight)
            y = tf.nn.bias_add(y, bias)
            if activation == None:
                return y
            else:
                return tf.nn.relu(y)

    def build(self, x_rgb):
        s_time = time.time()
        # 归一化处理，在第四维上将输入的图片的三通道拆分
        r, g, b = tf.split(x_rgb, [1, 1, 1], axis=3)
        # 分别将三通道上减去特定值归一化后再按bgr顺序拼起来
        VGG_MEAN = [103.939, 116.779, 123.68]
        x_bgr = tf.concat(
            [b - VGG_MEAN[0],
             g - VGG_MEAN[1],
             r - VGG_MEAN[2]],
            axis=3
        )
        # 判别拼接起来的数据是否符合期望，符合再继续往下执行
        assert x_bgr.get_shape()[1:] == [668, 668, 3]

        # 构建各个卷积、池化、全连接等层
        self.conv1_1 = self.conv_layer(x_bgr, b'conv1_1')
        self.conv1_2 = self.conv_layer(self.conv1_1, b'conv1_2')
        self.pool1 = self.pooling_layer(self.conv1_2, b'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, b'conv2_1')
        self.conv2_2 = self.conv_layer(self.conv2_1, b'conv2_2')
        self.pool2 = self.pooling_layer(self.conv2_2, b'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, b'conv3_1')
        self.conv3_2 = self.conv_layer(self.conv3_1, b'conv3_2')
        self.conv3_3 = self.conv_layer(self.conv3_2, b'conv3_3')
        self.pool3 = self.pooling_layer(self.conv3_3, b'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, b'conv4_1')
        self.conv4_2 = self.conv_layer(self.conv4_1, b'conv4_2')
        self.conv4_3 = self.conv_layer(self.conv4_2, b'conv4_3')
        self.pool4 = self.pooling_layer(self.conv4_3, b'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, b'conv5_1')
        self.conv5_2 = self.conv_layer(self.conv5_1, b'conv5_2')
        self.conv5_3 = self.conv_layer(self.conv5_2, b'conv5_3')
        self.pool5 = self.pooling_layer(self.conv5_3, b'pool5')

        # self.flatten = self.flatten_layer(self.pool5, b'flatten')
        # self.fc6 = self.fc_layer(self.flatten, b'fc6')
        # self.fc7 = self.fc_layer(self.fc6, b'fc7')
        # self.fc8 = self.fc_layer(self.fc7, b'fc8', activation=None)
        # self.prob = tf.nn.softmax(self.fc8, name='prob')

        print('模型构建完成，用时%d秒' % (time.time() - s_time))


vgg16_dir = './data/vgg16_model.npy'
style_img = './data/starry_night.jpg'
content_img = './data/city_night.jpg'
output_dir = './data'


def read_image(img):
    img = Image.open(img)
    img_np = np.array(img)  # 将图片转化为[668,668,3]数组
    img_np = np.asarray([img_np], )  # 转化为[1,668,668,3]的数组
    return img_np


# 输入风格、内容图像数组
style_img = read_image(style_img)
content_img = read_image(content_img)
# 定义对应的输入图像的占位符
content_in = tf.placeholder(tf.float32, shape=[1, 668, 668, 3])
style_in = tf.placeholder(tf.float32, shape=[1, 668, 668, 3])

# 初始化输出的图像
initial_img = tf.truncated_normal((1, 668, 668, 3), mean=127.5, stddev=20)
res_out = tf.Variable(initial_img)

# 构建VGG网络对象
res_net = VGGNet(vgg16_dir)
style_net = VGGNet(vgg16_dir)
content_net = VGGNet(vgg16_dir)
res_net.build(res_out)
style_net.build(style_in)
content_net.build(content_in)

# 计算损失，分别需要计算内容损失和风格损失
# 提取内容图像的内容特征
content_features = [
    content_net.conv1_2,
    content_net.conv2_2
    # content_net.conv2_2
]
# 对应结果图像提取相同层的内容特征
res_content = [
    res_net.conv1_2,
    res_net.conv2_2
    # res_net.conv2_2
]
# 计算内容损失
content_loss = tf.zeros(1, tf.float32)
for c, r in zip(content_features, res_content):
    content_loss += tf.reduce_mean((c - r) ** 2, [1, 2, 3])


# 计算风格损失的gram矩阵
def gram_matrix(x):
    b, w, h, ch = x.get_shape().as_list()
    features = tf.reshape(x, [b, w * h, ch])
    # 对features矩阵作内积，再除以一个常数
    gram = tf.matmul(features, features, adjoint_a=True) / tf.constant(w * h * ch, tf.float32)
    return gram


# 对风格图像提取特征
style_features = [
    # style_net.conv1_2
    style_net.conv4_3
]
style_gram = [gram_matrix(feature) for feature in style_features]
# 提取结果图像对应层的风格特征
res_features = [
    res_net.conv4_3
]
res_gram = [gram_matrix(feature) for feature in res_features]
# 计算风格损失
style_loss = tf.zeros(1, tf.float32)
for s, r in zip(style_gram, res_gram):
    style_loss += tf.reduce_mean((s - r) ** 2, [1, 2])

# 模型内容、风格特征的系数
k_content = 0.1
k_style = 500
# 按照系数将两个损失值相加
loss = k_content * content_loss + k_style * style_loss

# 进行训练
learning_steps = 100
learning_rate = 10
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(learning_steps):
        t_loss, c_loss, s_loss, _ = sess.run(
            [loss, content_loss, style_loss, train_op],
            feed_dict={content_in: content_img, style_in: style_img}
        )
        print('第%d轮训练，总损失：%.4f，内容损失：%.4f，风格损失：%.4f'
              % (i + 1, t_loss[0], c_loss[0], s_loss[0]))
        # 获取结果图像数组并保存
        res_arr = res_out.eval(sess)[0]
        res_arr = np.clip(res_arr, 0, 255)  # 将结果数组中的值裁剪到0~255
        res_arr = np.asarray(res_arr, np.uint8)  # 将图片数组转化为uint8
        img_path = os.path.join(output_dir, 'res_%d.jpg' % (i + 1))
        res_img = Image.fromarray(res_arr)
        res_img.save(img_path)
