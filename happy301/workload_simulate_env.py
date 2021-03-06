import pandas as pd
import numpy as np
from svm import PredictedModel


def sigmoid(x):
    s = 1 / (1 + np.exp(3 * x - 6))
    return s


class WorkloadEnv:

    def __init__(self, a, b):
        self.svm = PredictedModel()
        self.action_space = ['a1', 'r1', 'a2', 'r2', 'a3', 'r3']
        self.n_actions = len(self.action_space)
        self.n_features = 14  # wls:5 * 2 + vm:3 + qos:1
        workload_env = pd.read_csv('datas/workload.csv', header=None).loc[a:b]
        self.initWorkloadState = np.array(workload_env, np.float32)[:, 0:2].reshape(1, 10)
        self.objectVmPlan = np.array(np.array(workload_env)[0, 2:5], np.float)

        # self.vmPlan = np.random.randint(0, 8, (1, 3)).squeeze()
        self.vmPlan = [5, 5, 7]
        workload_vm = np.insert(self.initWorkloadState.squeeze()[0:2], 2, self.vmPlan)
        qos = self.svm.predict(workload_vm.reshape(1, -1))
        qos = sigmoid(qos)
        self.initState = np.insert(self.initWorkloadState, 10, np.append(self.vmPlan, qos))
        self.currentState = np.array(self.initState, np.float32)

    def get_vm_obj(self):
        return self.objectVmPlan

    # 环境初始化，给状态赋值
    def reset(self):
        # self.vmPlan = np.random.randint(0, 8, (1, 3)).squeeze()
        # self.vmPlan = [0, 0, 0]
        # workload_vm = np.insert(self.initWorkloadState.squeeze()[0:2], 2, self.vmPlan)
        # qos = self.svm.predict(workload_vm.reshape(1, -1))
        # qos = sigmoid(qos)
        # self.initState = np.insert(self.initWorkloadState, 10, np.append(self.vmPlan, qos))
        self.currentState = np.array(self.initState, np.float32)
        return self.currentState

    # 动作执行，返回新的状态
    def step(self, action):
        s_ = np.array(self.currentState, np.float32)
        reward = -10
        done = False
        if action == 0:   # a1
            if s_[10] < 8:
                s_[10] = s_[10] + 1
                reward = 0
        elif action == 1:   # r1
            if s_[10] > 0:
                s_[10] = s_[10] - 1
                reward = 0
        elif action == 2:   # a2
            if s_[11] < 8:
                s_[11] = s_[11] + 1
                reward = 0
        elif action == 3:   # r2
            if s_[11] > 0:
                s_[11] = s_[11] - 1
                reward = 0
        elif action == 4:  # a3
            if s_[12] < 8:
                s_[12] = s_[12] + 1
                reward = 0
        elif action == 5:  # r3
            if s_[12] > 0:
                s_[12] = s_[12] - 1
                reward = 0
        workload_vm = np.insert(s_[0:2], 2, s_[10:13])
        qos = self.svm.predict(workload_vm.reshape(1, -1))
        qos = sigmoid(qos)
        s_[13] = qos
        if (s_[10:13] == self.objectVmPlan).all():
            reward = 10
            done = True
        # else:
        #     reward = 0
        #     done = False
        self.currentState = s_
        return s_, reward, done

