from RL_brain import DeepQNetwork
from workload_env2 import WorkloadEnv as WorkloadEnv
from workload_env import WorkloadEnv as WorkloadEnv2
from MyNet import MyNet
import numpy as np
import pandas as pd


def run_this(env, RL, e):
    step = 0
    for episode in range(0, e):
        # 初始化云环境状态<WLs,VM,Qos>
        observation = env.reset()
        # 一轮迭代，完成一次模型训练过程
        while True:
            # RL通过ε-greedy选择一个动作
            action = RL.choose_action(observation)

            # 执行该动作，环境返回下一个状态
            observation_, reward, done = env.step(action)
            # 经验存放池存储经验
            RL.store_transition(observation, action, reward, observation_)

            if (step > 500) and (step % 30 == 0):
                # RL学习经验，更新参数
                RL.learn()

            # 更新状态
            observation = observation_

            # 判断虚拟机是否达到目标状态
            if done:
                break
            step += 1
    # end of game
    print('finish train')
    RL.reset_variables()
    # 保存网络参数


def cal_accuracy():
    ret = []
    for no in range(1, 13):
        # 1.获取指定步数差距的数据集
        datas = pd.read_csv('datas/vm_gap/vmGap' + str(no) + '_data.csv', header=None)
        length = len(datas)

        correct = 0
        # 2.循环体：循环所有数据
        for i in range(0, length):
            # 1）对每个类型数据，获取初始状态、目标状态，并通过初始状态调用一次DQN模型
            row = datas.iloc[i, 0:14]
            s_init = np.array(row, dtype=float).reshape(1, 14)
            # tmp = np.array(s_init).copy()
            # min_max = MinMaxScaler(feature_range=(0, 1))
            # tmp = min_max.fit_transform(tmp.reshape(-1, 1))
            # tmp = np.array(tmp).squeeze()
            # print(my_net.get_eval(s_init))
            a = np.argmax(RL.get_eval(s_init), axis=1)
            vm_next = WorkloadEnv2.step2(s_init, a)
            vm_obj = np.array(datas.iloc[i, 14:17], dtype=float).reshape(1, 3)
            vm_gap = np.absolute(vm_next - vm_obj)
            if np.sum(vm_gap) <= no:
                correct = correct + 1
            #     print('√')
            # else:
            #     print('X')
        # print('=============' + str(no) + '===============')
        # print(correct)
        # print(length)
        # print(correct / length)
        ret.append(correct / length)
    return ret


datas = pd.read_csv('datas/runtime_dataset.csv', header=None)
datas = np.array(datas)
length = len(datas)
np.random.shuffle(datas)
datas = pd.DataFrame(datas)
cost_his = []
RL = DeepQNetwork(6, 14,
                      learning_rate=0.01,
                      reward_decay=0.8,
                      e_greedy=0.9,
                      replace_target_iter=50,
                      memory_size=500,
                      output_graph=True,
                      cost_his=cost_his
                      )
for i in range(0, 10):
    row = datas.iloc[i, :]
    data = np.array(row, dtype=float).reshape(1, 17)
    env = WorkloadEnv(data)
    # vm_init = data[0][10:13]
    # vm_obj = data[0][14:17]
    # vm_init = np.array(vm_init, dtype=float).reshape(1, 3)
    # vm_obj = np.array(vm_obj, dtype=float).reshape(1, 3)
    # vm_gap = np.absolute(vm_init - vm_obj)
    # if np.sum(vm_gap) < 5:
    #     run_this(env, RL, 5)
    # else:
    # 训练agent
    run_this(env, RL, 2)
    # 一轮训练结束后，评估agent的准确性：分别计算12个步长测试数据的准确性
    ret = cal_accuracy()
    print(ret)
    # print(ret)

RL.save_params()
RL.plot_cost()
#
# if __name__ == "__main__":
#     env = WorkloadEnv(1, 5)
#     RL = DeepQNetwork(env.n_actions, env.n_features,
#                       learning_rate=0.0001,
#                       reward_decay=0.8,
#                       e_greedy=0.9,
#                       replace_target_iter=200,
#                       memory_size=2000,
#                       output_graph=True
#                       )
#     run_this()
#     RL.plot_cost()
#     # A = np.array([1, 2, 3])
#     # B = np.array([4, 5, 6])
#     # C = A - B
#     # print(C)
#     # print(np.absolute(C))
