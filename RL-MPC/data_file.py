# This file has all the data/variables required in one place

import numpy as np

####### RL or otherwise? #######
# set rl_flag to 1 if RL and 0 otherwise
rl_flag = 1

rl_algo_opt = "DDPG" # available options: DDPG or MADDPG
####### RL or otherwise? #######

algo_option = "rl_modified_ddswa"

# simulation time limit
#max_sim_time = 500
max_sim_time = 500


real_time_spawning_flag = 1




#################################### what is the list ?


run_coord_on_captured_snap = 0



heuristic_dict_id = 0

heuristic_dict = {0: None, 1:"fifo", 2:"time_to_react", 3:"dist_react_time", 4:"conv_dist_react"}

used_heuristic = heuristic_dict[heuristic_dict_id]
# 'congestion'
#"congestion"
#"num_follow_veh"
#"dist_to_int"
#"norm_dist_and_vel"
#"time_to_react"
#"fifo"
#"conv_dist_react"
#"dist_react_time"
#"rl_modified_ddswa"



#################################### what is the list ?



arr_rates_to_simulate = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

#arr_rates_to_simulate = [0.06]


# [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
# [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]


# time step
dt = round(1.0, 1)

# weight in cost on velocity
W_pos = 1

# weight in cost on acceleration
W_acc = 0

# weight in cost on jerk
W_jrk = 0

# scaling factor for average demand in ddswa
w_l = 1

# upper-bound on velocity
vm = {0:1, 3:1, 6:1, 9:1, 1:1.5 ,2:1 ,4:1, 5:1.5, 7:1.5, 8:1, 10:1, 11:1.5}
#{0:1.5, 3:1.5, 6:1.5, 9:1.5, 1:1.5 ,2:1.5 ,4:1.5, 5:1.5, 7:1.5, 8:1.5, 10:1.5, 11:1.5}

# upper-bound on acceleration
u_max = {0:2, 3:2, 6:2, 9:2, 1:2, 2:2, 4:2, 5:2, 7:2, 8:2, 10:2, 11:2} #3.0

# lower-bound on acceleration
u_min = {0:-2, 3:-2, 6:-2, 9:-2, 1:-2, 2:-2, 4:-2, 5:-2, 7:-2, 8:-2, 10:-2, 11:-2} #-3.0

# Priorities
lane_priorities = {0:1, 3:1, 6:1, 9:1, 1:3, 2:1, 4:2, 5:2, 7:1, 8:2, 10:3, 11:2}

# maximum number of vehicles in a lane related to RL
max_vehi_per_lane = 15

# RL DDPG replay-buffer-size
rl_ddpg_buff_size = 256

# RL DDPG sample-size !!!! important: keep this an even number !!!!
rl_ddpg_samp_size = 64

# RL Rpioritized experience replay (PER) flag. 0 means no PER
rl_per_flag = 0

# lane numbers
lanes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
#lanes = [0, 1, 2, 3]

# maximum number of lanes
lane_max = len(lanes)

# number of lanes per direction
lane_per_dir = lane_max/4

# Length of a vehicle
L = 0.75

# Width of vehicle
B = 0.7

# Number of lanes per branch
num_lanes = 2

num_veh = 10 * 4 * num_lanes

# length of the intersection (actual intersection)
int_bound = 2 * num_lanes * B

#lanes = [1, 2, 4]

# start of region of interest on different lanes
int_start = {0:-7, 3:-7, 6:-7, 9:-7, 1:-7 ,2:-7, 4:-7, 5:-7, 7:-7, 8:-7, 10:-7, 11:-7}

# distance between lanes
dist_bw_lanes = int_bound/(2*lane_per_dir)

# scheduling time
T_sc = 30

# prescheduling time
# T_ps = 20

# optimization interval T_c
t_opti = 6

# number of additional time steps to plan for after end of provisional phase 
# to maintain trajectory continuity in coordinated phase
prov_extra_steps = 2 # keep this value greater than or equal to 2

prov_roll_hor = 6 # plannig horizon

prov_impl_hor = 1  # number of steps taken in realtime
#prov_impl_hor = 3


# lenght of path inside intersection
intersection_path_length = [B / np.sqrt(2), int_bound, np.sqrt(((2.5 * B) ** 2) + ((2.5 * B) ** 2))]
#intersection_path_length = [int_bound]


# incompatibility dictionary
incompdict = {0:[],3:[],6:[],9:[],1:[4,10,8,11],2:[4,7,5,11],4:[1,7,2,11],5:[7,10,2,8],7:[4,10,2,5],8:[1,10,5,11],10:[1,7,8,5],11:[4,1,8,2]}
#incompdict = {0:[],3:[],1:[4],2:[4,5],4:[1,2],5:[2]}

# arrival rates for each branch (**for RL. not for ddswa!** )
# arr_rate_branch = 0.5*np.ones(len(branches))


# # arrival rate
# arr_rate = 0.01
# # array of arrival rates to be simulated for
# arr_rate_array = {0:0, 1:arr_rate, 2:arr_rate, 3:0, 4:arr_rate, 5:arr_rate, 6:0, 7:arr_rate, 8:arr_rate, 9:0, 10:arr_rate, 11:arr_rate} #arr_rate*np.ones(len(lanes)) #


# initial spawning position
p_init = int_start

#### data for main.py file ####
num_sim = 100 # number of simulations to be done
veh_in_lane = 1000
no_veh_per_lane = {0:0, 1:veh_in_lane, 2:veh_in_lane, 3:0, 4:veh_in_lane, 5:veh_in_lane, 6:0, 7:veh_in_lane, 8:veh_in_lane, 9:0, 10:veh_in_lane, 11:veh_in_lane} # number of vehicles per lane to be considered in each simulation

#### features of pseudo vehicle ####
#!!!!!!!! CAUTION !!!!!!!!#
######## DO NOT CHANGE THE ORDER OF THE FEATURES/STATE VARIABLES #######
d_since_arr = -7
feat_vel = 0
t_since_arr = -6
no_v_follow = 0
avg_sep = -10
avg_arr_rate = 0
min_wait_time = 60
lane = -1
vel_bound = 0
acc_bound = -10
priority_ = -10

# number of fetures
num_features = 10 #no arr_rate feature
num_dem_param = 5

#### weights for ddswa in the same order as features above ####

## weight vector with Wcomf = 0
#!!!!!!!! CAUTION !!!!!!!!#
######## DO NOT CHANGE THE ORDER #######
weights_Wc0_Wv1 = {'d_since_arr': 0.1, 'feat_vel': 5, 't_since_arr': 3, 'no_v_follow': 4, 'avg_sep': 5, 'avg_arr_rate': 40, 'min_wait_time': -0.5}

## weight vector with Wcomf = 1
#!!!!!!!! CAUTION !!!!!!!!#
######## DO NOT CHANGE THE ORDER #######
weights_Wc1_Wv1 = {'d_since_arr': 0.8, 'feat_vel': 7, 't_since_arr': 2, 'no_v_follow': 5, 'avg_sep': 6, 'avg_arr_rate': 40, 'min_wait_time': -4}

## weight vector for inhomogeneous traffic
#!!!!!!!! CAUTION !!!!!!!!#
######## DO NOT CHANGE THE ORDER #######
weights_inho = {'d_since_arr': 0.5, 'feat_vel': 4, 't_since_arr': 3, 'no_v_follow': 4, 'avg_sep': 6, 'avg_arr_rate': 30, 'min_wait_time': -1}

rand_weights = {'d_since_arr': np.random.uniform(0, 10), 'feat_vel': np.random.uniform(0, 10), 't_since_arr': np.random.uniform(0, 10), 'no_v_follow': np.random.uniform(0, 10), 'avg_sep': np.random.uniform(0, 10), 'avg_arr_rate': np.random.uniform(0, 50), 'min_wait_time': np.random.uniform(0, 10) }
current_used_weights = rand_weights


#current_used_weights = weights_Wc0_Wv1

colours = {0:'black', 1:'blue', 2:'brown', 3:'cyan', 4:'green', 5:'navy', 6:'grey', 7:'violet', 8:'olive', 9:'magenta', 10:'red', 11:'fuchsia'}




