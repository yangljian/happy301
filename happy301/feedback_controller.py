import tensorflow as tf
from MyNet import MyNet
import numpy as np
from workload_env import WorkloadEnv
from sklearn.preprocessing import StandardScaler


std = StandardScaler()
# min_max = MinMaxScaler(feature_range=(0, 1))
env = WorkloadEnv(7, 11)
my_net = MyNet(6, 14)
my_net.restore_net()
# 获取当前环境状态
s = env.reset()


def sigmoid(x):
    return 1.0 / (1 + np.exp(5 - float(x)))


# 循环
while True:
    # res = std.fit_transform(s.reshape(-1, 1))
    # res = np.array(res).squeeze()
    # 将状态传入模型，获取所有动作Q值,获取最大Q值的动作
    print(my_net.get_eval(s))
    print(s)
    a = np.argmax(my_net.get_eval(s), axis=1)

    # 执行动作，返回新的状态，更新状态，继续循环
    s, _, Done = env.step(a)

    # 当Q值小于阈值T时，退出循环
    # if Done:
    #     break
# 循环结束
