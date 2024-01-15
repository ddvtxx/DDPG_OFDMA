import pandas as pd

#from google.colab import files 
import environment_simulation_move as env


numAPuser = 5
numRU = 8
numSenario = 1
linkmode = 'uplink'
ru_mode = 4
episode = 20
test_env = env.environment_base(numAPuser,numRU,linkmode,ru_mode)

system_bitrate_history = []

for i_episode in range(episode):
    test_env.change_seed(i_episode)
    x_init,y_init = test_env.senario_user_local_init()
    x,y = x_init,y_init
    userinfo = test_env.senario_user_info(x,y)
    channel_gain_obs = test_env.channel_gain_calculate()
    RU_mapper = test_env.n_AP_RU_mapper()
    system_bitrate = test_env.calculate_4_cells(RU_mapper)
    system_bitrate_history.append(system_bitrate)
dataframe=pd.DataFrame({'bitrate':system_bitrate_history})
dataframe.to_csv("./result/bitrate_general_seed_0-19.csv", index=False,sep=',')