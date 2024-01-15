import numpy as np
import pandas as pd
import torch as T
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import math
#from google.colab import files 
import environment_simulation_move as env
from ReplayBuffer import ReplayBuffer,create_directory
from DDPG_agent import DDPG
import random
print(T.__version__)
for i_loop in range(10):
    numAPuser = 5
    numRU = 8
    numSenario = 1
    linkmode = 'uplink'
    ru_mode = 3
    episode = 20
    max_iteration = 2000
    test_env = env.environment_base(numAPuser,numRU,linkmode,ru_mode)

    DDPG_agent = DDPG(alpha=1e-4, beta=1e-4,numSenario=numSenario,numAPuser=numAPuser,numRU=numRU,
        actor_fc1_dim=2**6,actor_fc2_dim=2**7,actor_fc3_dim=2**7,
        actor_fc4_dim=2**6,
        critic_fc1_dim=2**6,critic_fc2_dim=2**7,critic_fc3_dim=2**7,
        critic_fc4_dim=2**6,
        ckpt_dir='./DDPG/',
        gamma=0.99,tau=0.001,action_noise=1e-5,max_size=10000,batch_size=128)
    create_directory('./DDPG/',sub_paths=['Actor', 'Target_actor', 'Critic', 'Target_critic'])

    system_bitrate_history_ave = []

    for i_episode in range(episode):
        system_bitrate_history = []
        test_env.change_seed(i_episode)
        x_init,y_init = test_env.senario_user_local_init()
        x,y = x_init,y_init
        userinfo = test_env.senario_user_info(x,y)
        channel_gain_obs = test_env.channel_gain_calculate()
        AP123_RU_mapper = test_env.n_AP_RU_mapper()
        for i_iteration in range(max_iteration):
            RU_mapper = np.zeros((numAPuser, numRU))
            RU_mapper_next = np.zeros((numAPuser, numRU))
            for i_step in range(numRU):
                if i_iteration <=1000:
                    action_pre = DDPG_agent.choose_action(RU_mapper.reshape(1,numAPuser,numRU), train=True)
                else:
                    action_pre = DDPG_agent.choose_action(RU_mapper.reshape(1,numAPuser,numRU), train=False)
                action_pre = action_pre.reshape(numAPuser,numRU)
                max_key = np.argmax(action_pre[:,i_step])
                RU_mapper_next[max_key][i_step] = 1
                action_map = np.zeros((numAPuser, numRU))
                action_map[max_key][i_step] = 1
                if i_step != numRU-1:
                    DDPG_agent.remember(RU_mapper.reshape(1,numAPuser,numRU), action_map.reshape(1,numAPuser,numRU), 0, RU_mapper_next.reshape(1,numAPuser,numRU), done=False)
                    DDPG_agent.learn()
                else:
                    final_mapper = np.vstack((AP123_RU_mapper, RU_mapper_next.reshape(1,numAPuser,numRU)))
                    system_bitrate = test_env.calculate_4_cells(final_mapper)
                    system_bitrate_history.append(system_bitrate)
                    reward = system_bitrate/(1e+6)
                    DDPG_agent.remember(RU_mapper.reshape(1,numAPuser,numRU), action_map.reshape(1,numAPuser,numRU), reward, RU_mapper_next.reshape(1,numAPuser,numRU), done=True)
                    DDPG_agent.learn()
                    print('loop =', i_loop,'episode =', i_episode,'iteration =', i_iteration,'system_bitrate =', system_bitrate)
                RU_mapper[max_key][i_step] = 1
            dataframe=pd.DataFrame({'bitrate':system_bitrate_history})
            dataframe.to_csv("./result/bitrate_single_wf_seed_"+str(i_loop)+"_"+str(i_episode)+".csv", index=False,sep=',')
        # print('episode =',i_episode,'average result =',np.mean(system_bitrate_history))
        # print('general result =',general_system_bitrate)
        system_bitrate_history_ave.append(np.mean(system_bitrate_history))
        dataframe=pd.DataFrame({'bitrate':system_bitrate_history_ave})
        dataframe.to_csv("./result/bitrate_single_wf_seed_0-19_"+str(i_loop)+".csv", index=False,sep=',')
    