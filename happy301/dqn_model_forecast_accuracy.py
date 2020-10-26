from MyNet import MyNet
import numpy as np
from workload_env import WorkloadEnv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

my_net = MyNet(6, 14)
my_net.restore_net()
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
        a = np.argmax(my_net.get_eval(s_init), axis=1)
        vm_next = WorkloadEnv.step2(s_init, a)
        vm_obj = np.array(datas.iloc[i, 14:17], dtype=float).reshape(1, 3)
        vm_gap = np.absolute(vm_next - vm_obj)
        if np.sum(vm_gap) <= no:
            correct = correct + 1
        #     print('√')
        # else:
        #     print('X')
    print('=============' + str(no) + '===============')
    print(correct)
    print(length)
    print(correct / length)
# 2）若执行动作后的方案靠近目标方案，则正确数+1，否则错误数+1
# 3.计算DQN算法准确率
