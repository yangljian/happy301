import tensorflow as tf
import numpy as np


class MyNet:
    def __init__(self, n_actions, n_features):
        self.n_features = n_features
        self.n_actions = n_actions
        self._build_net()
        self.sess = tf.Session()

    # 定义网络
    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 5, \
                tf.random_normal_initializer(0, 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                self.w1 = tf.get_variable('evl_w1', [self.n_features, n_l], initializer=w_initializer,
                                          collections=c_names)
                self.b1 = tf.get_variable('evl_b1', [1, n_l], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, self.w1) + self.b1)

            # with tf.variable_scope('l2'):
            #     self.w2 = tf.get_variable('evl_w2', [n_l, n_l], initializer=w_initializer, collections=c_names)
            #     self.b2 = tf.get_variable('evl_b2', [1, n_l], initializer=b_initializer, collections=c_names)
            #     l2 = tf.nn.relu(tf.matmul(l1, self.w2) + self.b2)
            #
            # with tf.variable_scope('l3'):
            #     self.w3 = tf.get_variable('evl_w3', [n_l, n_l], initializer=w_initializer, collections=c_names)
            #     self.b3 = tf.get_variable('evl_b3', [1, n_l], initializer=b_initializer, collections=c_names)
            #     l3 = tf.nn.relu(tf.matmul(l2, self.w3) + self.b3)
            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                self.w2 = tf.get_variable('evl_w2', [n_l, self.n_actions], initializer=w_initializer,
                                          collections=c_names)
                self.b2 = tf.get_variable('evl_b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, self.w2) + self.b2

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('target_w1', [self.n_features, n_l], initializer=w_initializer,
                                     collections=c_names)
                b1 = tf.get_variable('target_b1', [1, n_l], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # with tf.variable_scope('l2'):
            #     w2 = tf.get_variable('target_w2', [n_l, n_l], initializer=w_initializer, collections=c_names)
            #     b2 = tf.get_variable('target_b2', [1, n_l], initializer=b_initializer, collections=c_names)
            #     l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
            # with tf.variable_scope('l3'):
            #     w3 = tf.get_variable('target_w3', [n_l, n_l], initializer=w_initializer, collections=c_names)
            #     b3 = tf.get_variable('target_b3', [1, n_l], initializer=b_initializer, collections=c_names)
            #     l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('target_w2', [n_l, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('target_b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    # 恢复网络
    def restore_net(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, "my_net/save_net.ckpt")
        print(self.sess.run(self.w1))

    def get_eval(self, s):
        s = np.array(s).reshape(1, 14)
        q_eval = self.sess.run(
            self.q_eval,
            feed_dict={
                self.s: s
            })
        return q_eval

