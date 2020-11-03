"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# np.random.seed(1)
# tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=50,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            cost_his=[]
    ):
        self.memory_counter = 0
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = 128
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.std = StandardScaler()
        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        my_file = Path('my_net')
        if my_file.exists():
            print("exist")
            self.restore_net()
        self.cost_his = cost_his

    # def sigmoid(self, x):
    #     return 1.0 / (1 + np.exp(5 - float(x)))

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 15, \
                tf.random_normal_initializer(0,0.01), tf.constant_initializer(0.3)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                self.w1 = tf.get_variable('evl_w1', [self.n_features, n_l], initializer=w_initializer, collections=c_names)
                self.b1 = tf.get_variable('evl_b1', [1, n_l], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, self.w1) + self.b1)

            with tf.variable_scope('l2'):
                self.w2 = tf.get_variable('evl_w2', [n_l, n_l], initializer=w_initializer, collections=c_names)
                self.b2 = tf.get_variable('evl_b2', [1, n_l], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, self.w2) + self.b2)
            # #
            # with tf.variable_scope('l3'):
            #     self.w3 = tf.get_variable('evl_w3', [n_l, n_l], initializer=w_initializer, collections=c_names)
            #     self.b3 = tf.get_variable('evl_b3', [1, n_l], initializer=b_initializer, collections=c_names)
            #     l3 = tf.nn.relu(tf.matmul(l2, self.w3) + self.b3)
            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l3'):
                self.w3 = tf.get_variable('evl_w3', [n_l, self.n_actions], initializer=w_initializer, collections=c_names)
                self.b3 = tf.get_variable('evl_b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l2, self.w3) + self.b3

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            tf.summary.scalar('loss', self.loss)
        with tf.variable_scope('train'):
            self._train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('target_w1', [self.n_features, n_l], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('target_b1', [1, n_l], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('target_w2', [n_l, n_l], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('target_b2', [1, n_l], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
            # with tf.variable_scope('l3'):
            #     w3 = tf.get_variable('target_w3', [n_l, n_l], initializer=w_initializer, collections=c_names)
            #     b3 = tf.get_variable('target_b3', [1, n_l], initializer=b_initializer, collections=c_names)
            #     l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('target_w3', [n_l, self.n_actions], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('target_b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l2, w3) + b3

    def max_min_normalization(self, x, max_element, min_element):
        x = (x - min_element) / (max_element - min_element)
        return x

    def restore_net(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, "my_net/save_net.ckpt")

    def reset_variables(self):
        self.learn_step_counter = 0
        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))
        self.memory_counter = 0
        # print(self.sess.run(self.w1))
        # print(self.sess.run(self.b1))
        # self.sess.run('b1')
        # self.sess.run('w2')
        # self.sess.run('b2')

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))
        # transition = self.std.fit_transform(transition.reshape(-1, 1))
        # transition = np.array(transition).squeeze()
        min_max = MinMaxScaler(feature_range=(0, 1))
        transition = min_max.fit_transform(transition.reshape(-1, 1))
        transition = np.array(transition).squeeze()
        # max_element = transition.max()
        # min_element = transition.min()
        # for i in range(0, len(transition)):
        #     transition[i] = self.sigmoid(transition[i])

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        if np.random.uniform() > self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        summary, _, self.cost = self.sess.run([self.merged, self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        # print(self.cost)
        self.cost_his.append(self.cost)
        self.writer.add_summary(summary, self.learn_step_counter)
        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def save_params(self):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, "my_net/save_net.ckpt")
        # print("save to path:", save_path)
        # print("evl_w:", self.sess.run(self.w1))

    def save_data_to_files(self):
        data = pd.DataFrame(self.memory)

        writer = pd.ExcelWriter("qos_training/" + self.data_file_name + ".xlsx")  # 写入Excel文件
        data.to_excel(writer, 'sheet_1', float_format='%.8f')  # ‘page_1’是写入excel的sheet名
        writer.save()
        writer.close()

    def get_eval(self, s):
        s = np.array(s).reshape(1, 14)
        q_eval = self.sess.run(
            self.q_eval,
            feed_dict={
                self.s: s
            })
        return q_eval





