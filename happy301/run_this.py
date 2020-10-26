from RL_brain import DeepQNetwork
from workload_env import WorkloadEnv
from MyNet import MyNet
import numpy as np
from sklearn.preprocessing import StandardScaler
std = StandardScaler()


def run_this():
    step = 0
    flag = True
    episode = 1
    # for episode in range(10):
    while flag:
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

            if (step > 200) and (step % 5 == 0):
                # RL学习经验，更新参数
                RL.learn()

            # 更新状态
            observation = observation_

            # 判断虚拟机是否达到目标状态
            if done:
                break
            step += 1

        # 调用反馈控制算法，判断是否达到目标要求(设置训练50轮后，每五轮执行一次)
        # if episode >= 30 and episode % 2 == 0:
            # RL.save_params()
            # RL.save_data_to_files()
            # my_net = MyNet(6, 14)
            # my_net.restore_net()
            # 获取当前环境状态
            s = env.reset()
            vm_obj = env.get_vm_obj()
            count = 24
            # 循环
            while count > 0:
                # res = std.fit_transform(s.reshape(-1, 1))
                # res = np.array(res).squeeze()
                # 将状态传入模型，获取所有动作Q值,获取最大Q值的动作
                print(RL.get_eval(s))
                print(s)
                a = np.argmax(RL.get_eval(s), axis=1)
                # 执行动作，返回新的状态，更新状态，继续循环
                s, _, Done = env.step(a)
                if Done:
                    break
                count = count - 1
                vm_current = s[10:13]
                vm_gap = np.absolute(vm_obj - vm_current)
                accept = vm_gap[0] <= 2 and vm_gap[1] <= 2 and vm_gap[2] <= 2
                if accept:
                    break

            if Done or accept:
                print(s)
                flag = False
                break
        episode = episode + 1
    # end of game
    print('finish')
    # 保存网络参数
    # RL.save_params()
    # RL.save_data_to_files()


if __name__ == "__main__":
    env = WorkloadEnv(7, 11)
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.00001,
                      reward_decay=0.8,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=True
                      )
    run_this()
    RL.plot_cost()
