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

for i_loop in range(1):
    numAPuser = 5
    numRU = 8
    numSenario = 1
    linkmode = 'uplink'
    ru_mode = 3
    episode = 20
    max_iteration = 2000
    test_env = env.environment_base(numAPuser,numRU,linkmode,ru_mode)

    agent_array = []
    for i_agent in range(4):
        DDPG_agent = DDPG(alpha=1e-4, beta=1e-4,numSenario=numSenario,numAPuser=numAPuser,numRU=numRU,
            actor_fc1_dim=2**6,actor_fc2_dim=2**7,actor_fc3_dim=2**7,
            actor_fc4_dim=2**6,
            critic_fc1_dim=2**6,critic_fc2_dim=2**7,critic_fc3_dim=2**7,
            critic_fc4_dim=2**6,
            ckpt_dir='./DDPG/',
            gamma=0.99,tau=0.001,action_noise=1e-5,max_size=10000,batch_size=128)
        create_directory('./DDPG_'+ str(i_agent) +'/',sub_paths=['Actor', 'Target_actor', 'Critic', 'Target_critic'])
        agent_array.append(DDPG_agent)
    
    system_bitrate_history_ave = []
    system_bitrate_history_max = []

    for i_episode in range(episode):
        system_bitrate_history = []
        test_env.change_seed(i_episode)
        x_init,y_init = test_env.senario_user_local_init()
        x,y = x_init,y_init
        userinfo = test_env.senario_user_info(x,y)
        channel_gain_obs = test_env.channel_gain_calculate()

        for i_iteration in range(max_iteration):
            action_array_array = []
            RU_mapper_array = []
            for i_agent in range(4):
                DDPG_agent = agent_array[i_agent]
                action_array = []   
                RU_mapper = np.zeros((numAPuser, numRU))
                for i_step in range(numRU):
                    action_pre = DDPG_agent.choose_action(RU_mapper.reshape(1,numAPuser,numRU),True)
                    action_pre = action_pre.reshape(numAPuser,numRU)
                    max_key = np.argmax(action_pre[:,i_step])
                    RU_mapper[max_key][i_step] = 1
                    action_map = np.zeros((numAPuser, numRU))
                    action_map[max_key][i_step] = 1
                    action_array.append(action_map)
                action_array_array.append(action_array)
                RU_mapper_array.append(RU_mapper)
            RU_mapper_final = np.vstack((RU_mapper_array[0].reshape(1,numAPuser,numRU), 
                                         RU_mapper_array[1].reshape(1,numAPuser,numRU),
                                         RU_mapper_array[2].reshape(1,numAPuser,numRU),
                                         RU_mapper_array[3].reshape(1,numAPuser,numRU)
                                         ))
            system_bitrate = test_env.calculate_4_cells(RU_mapper_final)
            reward = system_bitrate/(1e+4)
            system_bitrate_history.append(system_bitrate)
            #learning
            for i_agent in range(4):
                action_array = action_array_array[i_agent]
                RU_mapper = np.zeros((numAPuser, numRU))
                RU_mapper_next = np.zeros((numAPuser, numRU))
                for i_step in range(numRU):
                    RU_mapper_next = RU_mapper_next + action_array[i_step]
                    if i_step == numRU-1:
                        DDPG_agent.remember(RU_mapper.reshape(1, numAPuser, numRU),
                                            action_array[i_step].reshape(1,numAPuser,numRU),
                                            reward,
                                            RU_mapper_next.reshape(1,numAPuser,numRU),
                                            done = True
                                            )
                    else:
                        DDPG_agent.remember(RU_mapper.reshape(1, numAPuser, numRU),
                                            action_array[i_step].reshape(1,numAPuser,numRU),
                                            0,
                                            RU_mapper_next.reshape(1,numAPuser,numRU),
                                            done = False
                                            )
                    RU_mapper = RU_mapper + action_array[i_step]
            print('loop =', i_loop,'episode =', i_episode,'iteration =', i_iteration,'system_bitrate =', system_bitrate)
            dataframe=pd.DataFrame({'bitrate':system_bitrate_history})
            dataframe.to_csv("./result/bitrate_single_wf_seed_"+str(i_loop)+"_"+str(i_episode)+".csv", index=False,sep=',')
        system_bitrate_history_ave.append(np.mean(system_bitrate_history))
        dataframe=pd.DataFrame({'bitrate':system_bitrate_history_ave})
        dataframe.to_csv("./result/bitrate_single_wf_seed_0-19_"+str(i_loop)+".csv", index=False,sep=',')


                
