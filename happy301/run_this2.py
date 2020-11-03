from RL_brain import DeepQNetwork
from workload_train_env import WorkloadEnv as WorkloadEnv
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


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

            if step > 500 and step % 10 == 0:
                # RL学习经验，更新参数
                RL.learn()

            # 更新状态
            observation = observation_

            # 判断虚拟机是否达到目标状态
            if done:
                break
            step += 1
    # end of game
    # print('finish train')
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
            tmp = np.array(s_init).copy()
            min_max = MinMaxScaler(feature_range=(0, 1))
            tmp = min_max.fit_transform(tmp.reshape(-1, 1))
            tmp = np.array(tmp).squeeze()
            # print(str(no) + ":" + str(i) + "----")
            print(RL.get_eval(tmp))
            q_value = RL.get_eval(tmp)
            a1 = np.argmax(q_value, axis=1)
            q_value[0][a1] = 0
            a2 = np.argmax(q_value, axis=1)
            q_value[0][a2] = 0
            a3 = np.argmax(q_value, axis=1)
            vm_next1 = WorkloadEnv.step2(s_init, a1)
            vm_next2 = WorkloadEnv.step2(s_init, a2)
            vm_next3 = WorkloadEnv.step2(s_init, a3)
            vm_obj = np.array(datas.iloc[i, 14:17], dtype=float).reshape(1, 3)
            vm_gap1 = np.absolute(vm_next1 - vm_obj)
            vm_gap2 = np.absolute(vm_next2 - vm_obj)
            vm_gap3 = np.absolute(vm_next3 - vm_obj)
            if no - 3 <= 0 and (np.sum(vm_gap1) <= no or np.sum(vm_gap2) <= no or np.sum(vm_gap3) <= no):
                correct = correct + 1
            elif no - 3 > 0 and (np.sum(vm_gap1) <= no or np.sum(vm_gap2) <= no):
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
                      learning_rate=0.00001,
                      reward_decay=0.99,
                      e_greedy=0.9,
                      replace_target_iter=100,
                      memory_size=500,
                      output_graph=True,
                      cost_his=cost_his
                      )
for i in range(0, 500):
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
    # if i % 10 == 0:
    # ret = cal_accuracy()
    # print(ret)
    # if min(ret) > 0.8:
    #     break
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
