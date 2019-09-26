import os
import jieba
import numpy as np
import tensorflow as tf
import math

train_dir = './data/text_category/train'
train_segment_dir = './data/text_category/train_segment.txt'
train_list_dir = './data/text_category/train_list.txt'


# 将文件中的句子通过jieba库拆分为单个词
def segment_word(input_file, output_file):
    # 循环遍历训练数据集的每一个文件
    for root, folders, files in os.walk(input_file):
        print('root:', root)
        for folder in folders:
            print('dir:', folder)
        for file in files:
            file_dir = os.path.join(root, file)
            with open(file_dir, 'rb') as in_file:
                # 读取文件中的文本
                sentence = in_file.read()
                # 通过jieba函数库将句子拆分为单个词组
                words = jieba.cut(sentence)
                # 文件夹路径最后两个字即为分类名
                content = root[-2:] + '\t'
                # 去除词组中的空格，排除为空的词组
                for word in words:
                    word = word.strip(' ')
                    if word != '':
                        content += word + ' '
            # 换行并将文本写入输出文件
            content += '\n'
            with open(output_file, 'a') as outfile:
                outfile.write(content.strip(' '))


# 统计每个词出现的频率
def get_list(segment_file, out_file):
    # 通过词典保存每个词组出现的频率
    word_dict = {}
    with open(segment_file, 'r') as seg_file:
        lines = seg_file.readlines()
        # 遍历文件的每一行
        for line in lines:
            line = line.strip('\r\n')
            # 将一行按空格拆分为每个词，统计词典
            for word in line.split(' '):
                # 如果这个词组没有在word_dict词典中出现过，则新建词典项并设为0
                word_dict.setdefault(word, 0)
                # 将词典word_dict中词组word对应的项计数加一
                word_dict[word] += 1
        # 将词典中的列表排序，关键字为列表下标为1的项，且逆序
        sorted_list = sorted(word_dict.items(), key=lambda d: d[1], reverse=True)
        with open(out_file, 'w') as outfile:
            # 将排序后的每条词典项写入文件
            for item in sorted_list:
                outfile.write('%s\t%d\n' % (item[0], item[1]))


# segment_word(train_dir, train_segment_dir)
# get_list(train_segment_dir,train_list_dir)

# 通过RNN网络进行文本分类训练
# 1、封装数据集类，提供接口next_batch用于连续批次数据读取
# 2、封装词表，将句子转化为词语id。类别封装，将输入词语类别转为id
# 3、构建网络模型，输入进行embedding，之后经过LSTM层，全连接层，然后进行训练

# 定义超参数
embedding_size = 32  # 每个词组向量的长度
max_words = 10  # 一个句子最大词组长度
lstm_layers = 2  # lstm网络层数
lstm_nodes = [64, 64]  # lstm每层结点数
fc_nodes = 64  # 全连接层结点数
batch_size = 100  # 每个批次样本数据
lstm_grads = 1.0  # lstm网络梯度
learning_rate = 0.001  # 学习率
word_threshold = 10  # 词表频率门限，低于该值的词语不统计
num_classes = 4  # 最后的分类结果有4类



class Word_list:
    def __init__(self, filename):
        # 用词典类型来保存需要统计的词组及其频率
        self._word_dic = {}
        with open(filename, 'r') as f:
            lines = f.readlines()
        for line in lines:
            res = line.strip('\r\n').split('\t')[0:2]
            if len(res) < 2:
                continue
            else:
                word, freq = res
            if freq.isdigit():
                freq = int(freq)
            else:
                continue
            # 如果词组的频率小于阈值，跳过不统计
            if freq < word_threshold:
                continue
            # 词组列表中每个词组都是不重复的，按序添加到word_dic中即可，下一个词组id就是当前word_dic的长度
            word_id = len(self._word_dic)
            self._word_dic[word] = word_id

    def sentence2id(self, sentence):
        # 将以空格分割的句子返回word_dic中对应词组的id，若不存在返回-1
        sentence_id = [self._word_dic.get(word, -1)
                       for word in sentence.split()]
        return sentence_id


train_list = Word_list(train_list_dir)


class TextData:
    def __init__(self, segment_file, word_list):
        self.inputs = []
        self.labels = []
        # 通过词典管理文本类别
        self.label_dic = {'体育': 0, '校园': 1, '女性': 2, '文学': 3}
        self.index = 0

        with open(segment_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # 文本按制表符分割，前面为类别，后面为句子
                res = line.strip('\r\n').split('\t')[0:2]
                if len(res) < 2:
                    continue
                else:
                    label, content = res
                self.content_size = len(content)
                # 将类别转换为数字id
                label_id = self.label_dic.get(label)
                # 将句子转化为embedding数组
                content_id = word_list.sentence2id(content)
                # 如果句子的词组长超过最大值，截取max_words长度以内的id值
                content_id = content_id[0:max_words]
                # 如果不够则填充-1，直到max_words长度
                padding_num = max_words - len(content_id)
                content_id = content_id + [-1 for i in range(padding_num)]
                self.inputs.append(content_id)
                self.labels.append(label_id)
        self.inputs = np.asarray(self.inputs, dtype=np.int32)
        self.labels = np.asarray(self.labels, dtype=np.int32)
        self._shuffle_data()

    # 对数据按照(input,label)对来打乱顺序
    def _shuffle_data(self):
        r_index = np.random.permutation(len(self.inputs))
        self.inputs = self.inputs[r_index]
        self.labels = self.labels[r_index]

    # 返回一个批次的数据
    def next_batch(self, batch_size):
        # 当前索引+批次大小得到批次的结尾索引
        end_index = self.index + batch_size
        # 如果结尾索引大于样本总数，则打乱所有样本从头开始
        if end_index > len(self.inputs):
            self._shuffle_data()
            self.index = 0
            end_index = batch_size
        # 按索引返回一个批次的数据
        batch_inputs = self.inputs[self.index:end_index]
        batch_labels = self.labels[self.index:end_index]
        self.index = end_index
        return batch_inputs, batch_labels

    # 获取词表数目
    def get_size(self):
        return self.content_size


# 训练数据集对象
train_set = TextData(train_segment_dir, train_list)
# print(data_set.next_batch(10))
# 训练数据集词组条数
train_list_size = train_set.get_size()
print('列表长度', train_list_size)


# 创建计算图模型
def create_model(list_size, num_classes):
    # 定义输入输出占位符
    inputs = tf.placeholder(tf.int32, (batch_size, max_words))
    outputs = tf.placeholder(tf.int32, (batch_size,))
    # 定义是否dropout的比率
    keep_prob = tf.placeholder(tf.float32, name='keep_rate')
    # 记录训练的总次数
    global_steps = tf.Variable(tf.zeros([], tf.float32), name='global_steps', trainable=False)

    # 将输入转化为embedding编码
    with tf.variable_scope('embedding',
                           initializer=tf.random_normal_initializer(-1.0, 1.0)):
        embeddings = tf.get_variable('embedding', [list_size, embedding_size], tf.float32)
        # 将指定行的embedding数值抽取出来
        embedded_inputs = tf.nn.embedding_lookup(embeddings, inputs)

    # 实现LSTM网络
    # 生成Cell网格所需参数
    def _generate_paramas(x_size, h_size, b_size):
        x_w = tf.get_variable('x_weight', x_size)
        h_w = tf.get_variable('h_weight', h_size)
        bias = tf.get_variable('bias', b_size, initializer=tf.constant_initializer(0.0))
        return x_w, h_w, bias

    scale = 1.0 / math.sqrt(embedding_size + lstm_nodes[-1]) / 3.0
    lstm_init = tf.random_uniform_initializer(-scale, scale)
    with tf.variable_scope('lstm_nn', initializer=lstm_init):
        '''
        # 通过库函数构建两层的lstm，每层结点数为lstm_nodes[i]
        cells = []
        for i in range(lstm_layers):
            cell = tf.contrib.rnn.BasicLSTMCell(lstm_nodes[i], state_is_tuple=True)
            # 实现Dropout操作
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
            cells.append(cell)
        # 合并两个lstm的cell
        cell = tf.contrib.rnn.MultiRNNCell(cells)
        # 将embedded_inputs输入到RNN中进行训练
        initial_state = cell.zero_state(batch_size, tf.float32)
        # runn_output:[batch_size,num_timestep,lstm_outputs[-1]
        rnn_output, _ = tf.nn.dynamic_rnn(cell, embedded_inputs, initial_state=initial_state)
        last_output = rnn_output[:, -1, :]
        '''

        # 输入门
        with tf.variable_scope('input'):
            x_in, h_in, b_in = _generate_paramas(
                x_size=[embedding_size, lstm_nodes[0]],
                h_size=[lstm_nodes[0], lstm_nodes[0]],
                b_size=[1, lstm_nodes[0]]
            )
        # 输出门
        with tf.variable_scope('output'):
            x_out, h_out, b_out = _generate_paramas(
                x_size=[embedding_size, lstm_nodes[0]],
                h_size=[lstm_nodes[0], lstm_nodes[0]],
                b_size=[1, lstm_nodes[0]]
            )
        # 遗忘门
        with tf.variable_scope('forget'):
            x_f, h_f, b_f = _generate_paramas(
                x_size=[embedding_size, lstm_nodes[0]],
                h_size=[lstm_nodes[0], lstm_nodes[0]],
                b_size=[1, lstm_nodes[0]]
            )
        # 中间状态
        with tf.variable_scope('mid_state'):
            x_m, h_m, b_m = _generate_paramas(
                x_size=[embedding_size, lstm_nodes[0]],
                h_size=[lstm_nodes[0], lstm_nodes[0]],
                b_size=[1, lstm_nodes[0]]
            )
        # 两个初始化状态，隐含状态state和初始输入h
        state = tf.Variable(tf.zeros([batch_size, lstm_nodes[0]]), trainable=False)
        h = tf.Variable(tf.zeros([batch_size, lstm_nodes[0]]), trainable=False)
        # 遍历每个词的输入过程
        for i in range(max_words):
            # 取出每轮输入，三维数组embedd_inputs的第二维代表训练的轮数
            embedded_input = embedded_inputs[:, i, :]
            # 将取出的结果reshape为二维
            embedded_input = tf.reshape(embedded_input, [batch_size, embedding_size])
            # 遗忘门计算
            forget_gate = tf.sigmoid(tf.matmul(embedded_input, x_f) + tf.matmul(h, h_f) + b_f)
            # 输入门计算
            input_gate = tf.sigmoid(tf.matmul(embedded_input, x_in) + tf.matmul(h, h_in) + b_in)
            # 输出门
            output_gate = tf.sigmoid(tf.matmul(embedded_input, x_out) + tf.matmul(h, h_out) + b_out)
            # 中间状态
            mid_state = tf.tanh(tf.matmul(embedded_input, x_m) + tf.matmul(h, h_m) + b_m)
            # 计算隐含状态state和输入h
            state = state * forget_gate + input_gate * mid_state
            h = output_gate + tf.tanh(state)
        # 最后遍历的结果就是LSTM的输出
        last_output = h

    # 构建全连接层
    fc_init = tf.uniform_unit_scaling_initializer(factor=1.0)
    with tf.variable_scope('fc', initializer=fc_init):
        fc1 = tf.layers.dense(last_output, fc_nodes, activation=tf.nn.relu, name='fc1')
        fc1_drop = tf.contrib.layers.dropout(fc1, keep_prob)
        logits = tf.layers.dense(fc1_drop, num_classes, name='fc2')

    # 定义评估指标
    with tf.variable_scope('matrics'):
        # 计算损失值
        softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=outputs)
        loss = tf.reduce_mean(softmax_loss)
        # 计算预测值，求第1维中最大值的下标，例如[1,1,5,3,2] argmax=> 2
        y_pred = tf.argmax(tf.nn.softmax(logits), 1, output_type=tf.int32)
        # 求准确率
        correct_prediction = tf.equal(outputs, y_pred)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 定义训练方法
    with tf.variable_scope('train_op'):
        train_var = tf.trainable_variables()
        # for var in train_var:
        #     print(var)
        # 对梯度进行裁剪防止梯度消失或者梯度爆炸
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, train_var), clip_norm=lstm_grads)
        # 将梯度应用到变量上去
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, train_var), global_steps)

    # 以元组的方式将结果返回
    return ((inputs, outputs, keep_prob),
            (loss, accuracy),
            (train_op, global_steps))


# 运行模型构建函数，并将返回的参数赋值给外部
placeholders, matrics, others = create_model(train_list_size, num_classes)
inputs, outputs, keep_prob = placeholders
loss, accuracy = matrics
train_op, global_steps = others

# 进行训练
init_op = tf.global_variables_initializer()
train_keep_prob = 0.8  # 训练集的dropout比率
train_steps = 10000

with tf.Session() as sess:
    sess.run(init_op)

    for i in range(train_steps):
        # 按批次获取训练集数据
        batch_inputs, batch_labels = train_set.next_batch(batch_size)
        # 运行计算图
        res = sess.run([loss, accuracy, train_op, global_steps],
                       feed_dict={inputs: batch_inputs, outputs: batch_labels,
                                  keep_prob: train_keep_prob})
        loss_val, acc_val, _, g_step_val = res
        if g_step_val % 20 == 0:
            print('第%d轮训练，损失：%3.3f,准确率：%3.5f' % (g_step_val, loss_val, acc_val))
