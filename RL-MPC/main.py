# this is the main file which calls all the remaining functions and
# keeps track of simulation time

import os ; os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
import sys
#from numba import jit, cuda 
import time
import copy
import numpy as np
import random
import math
from collections import deque
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import csv
import pickle
import time
import data_file
import vehicle
import gen_vehi
import prov_phase
import coord_phase
import functions
import set_class
import vehicle
import prece_dem
import delete_data

from multiprocessing import Pool

def func(_args):

	train_iter = _args[0] # If this value is -1 learning_flag set to zero
	sim = _args[1]     # Argument for calling RL agent, default zero
	arr_rate_array = _args[2]
	arr_rate_ = _args[3]
	train_sim = _args[4]    # what is this  ?????

	### algorithm option if data_file.rl_flag is 0 ###

	###### available algorithm options ######

			# algo_option = "comb_opt" -> for combined optimization
			# conda listalgo_option = "ddswa" -> for ddswa

	###### available algorithm options ######

	algo_option = data_file.algo_option # "comb_opt"

	### algorithm option if data_file.rl_flag is 0 ###


####????????????????????????
	# ### flag to switch between real-time vehicle generation and generating all vehicles before simulation ###

	# real_time_spawning_flag = 1

	# ### flag to switch between real-time vehicle generation and generating all vehicles before simulation ###

####????????????????????????
	capture_snapshot_flag = 0

	learning_flag = 0

	max_rep_sim = 1

	time_to_capture = 100

	#captured_snapshots = []


	### flag to run combined optimization for testing RL performance ###

	comb_test_probability = 0

	baseline_test_flag = 0

	baseline_test_freq = 10

	### flag to run combined optimization for testing RL performance ###



	#### rl_flag part ####
	if data_file.rl_flag:

		import tensorflow as tf

        ##### Selection of  ddpg and MADDPG   ######
		if data_file.rl_algo_opt == "DDPG":
			from ddpg_related_class import DDPG as Agent

		elif data_file.rl_algo_opt == "MADDPG":
			from maddpg import DDPG as Agent

		 ##### Selection of  ddpg and MADDPG   ######	


		#### initializing variables to store values ####

		rl_ret_collection = []
		comb_opt_test = []
		moving_avg_ret = []
		ddswa_comp_ret = []
		comb_opt_ret_collection = []
		rl_explore_data = []

		#### initializing variables to store values ####


		#### rl_ddswa(RL found inside coord phase for multuiple times ) vs rl_modified _ddswa: RL for finding index peformend only once 
        ### current seq_optimisation IS RL_modified_DDSWA
       
		### algorithm option if data_file.rl_flag is 1 ###

		###### available algorithm options ######

				# algo_option = "rl_ddswa" -> for rl based ddswa (not implemented)
				# algo_option = "rl_modified_ddswa" -> for rl based modified ddswa

		###### available algorithm options ######

		algo_option = "rl_modified_ddswa"

		### algorithm option if data_file.rl_flag is 1 ###

		if train_iter == -1:    # FOR input "--train" loop 
			learning_flag = 1  

		else:
			learning_flag = 0


		### flag to switch between stream learning and snapshot learning ####

		stream_rl_flag = 1
		
		### flag to switch between stream learning and snapshot learning ####

		#sim = 1
		ss = [5000, 64] # buffer size and sample size
		actor_lr = 0.0001
		critic_lr = 0.001
		p_factor = 0.0001
		d_factor = 0

		agent = None  ## define it as NONE so not empty/random value
		


		#### RL agent object creation ####
		if data_file.rl_algo_opt == "DDPG":
			if algo_option == "rl_modified_ddswa":
				agent = Agent(sim, samp_size=ss[1], buff_size=ss[0], act_lr=actor_lr, cri_lr=critic_lr, polyak_factor=p_factor, disc_factor=d_factor)
				

				if learning_flag:          # FOR input "--train" loop 

					# agent.actor_model_.load_weights(f"../data/init_weights/train_sim_{sim}/actor_weights_itr_0")
					# agent.critic_model_.load_weights(f"../data/init_weights/train_sim_{sim}/critic_weights_itr_0")

					curr_state = None
					prev_state = None

					curr_act = None
					prev_act = None

					curr_rew = None
					prev_rew = None

					max_rep_sim = 61 # for CML 61 episodes of data, each of 500 sec [data collection phase]


			elif algo_option == "rl_ddswa":
				agent = Agent(algo_opt=algo_option, state_size=data_file.num_features*data_file.lane_max, action_size=2*data_file.lane_max)
		
		elif data_file.rl_algo_opt == "MADDPG":
			if algo_option == "rl_modified_ddswa":
				agent = Agent(algo_opt=algo_option, num_of_agents=data_file.max_vehi_per_lane*data_file.lane_max, state_size=data_file.num_features, action_size=2)

			elif algo_option == "rl_ddswa":
				agent = Agent(algo_opt=algo_option, state_size=data_file.num_features*data_file.lane_max, action_size=2*data_file.lane_max)
		#### RL agent object creation ####


		## load trained model
		if (not learning_flag) and (data_file.used_heuristic == None):
			agent.actor_model_.load_weights(f"../data/merged_replay_buffer_with_next_state/train_sim_{train_sim}/trained_weights/actor_weights_itr_{train_iter}")
			max_rep_sim = 1

		#### RL agent object creation ####
		#if train_iter == 0:
		#	agent.actor_model_.load_weights(f"../data/arr_{arr_rate_}/multi_snap_train_rand_500/init_weights/sim_{sim}/actor_itr_0")

		#else:
		#	agent.actor_model_.load_weights(f"../data/arr_{arr_rate_}/multi_snap_train_rand_500/sim_{sim}/trained_weights/actor_itr_{train_iter*500}")


    #### rl_flag part ####

	
	
	#### sanpshot based ####
	if data_file.run_coord_on_captured_snap:
		with open(f"../data/{data_file.algo_option}/arr_{arr_rate_}/coord_phase_info_{sim}.csv", "a", newline="") as f:
			writer = csv.writer(f)
			writer.writerows([["sim_time", "num veh in Vs", "num veh in Vp",  "num veh crossed", "RL like cost", "actual coord cost", "prov phase time", "total coord phase time", "coord phase solve only time per veh", "coord phase object construction plus solve time per veh"]])		

	

	############## streamline ######################
	if not data_file.run_coord_on_captured_snap:

		#################### QUERIES #########################
	    # Is ddpg used before CML and MADDPG after cml --- yes
		# Is this train module in this file only for collecting data for CML -- yes
		#################### QUERIES #########################

		############################################################################
			# real-time spawning: generate robots through poisson distribution
			# NOT real-time spawning: to use the robots already generated and stored in a file
			# UNSPAWNED SET: IN NOT real-time spawning setting, THE SET OF ROBOTS STILL LEFT TO BE SPAWNED YET(yet to be intoduced inside the ROI)
			# functions.get_num_of_objects  provides the sum of lenght of list of a perticular set in all lanes
        ############################################################################
		



		for rep_sim in range(0, max_rep_sim): # through DDPG it will 61

			time_track = 0  # time tracking variable

			cumulative_throuput_count = 0  # cumulative throuput

			throughput_id_in_lane = [0 for _ in data_file.lanes] 
			# variable to help track last vehicle in each lane, a list of zeros, change to 1 if particular one crosses the intersection


			sim_obj = set_class.sets() 
			# a class of sets to classify vehicles whether they are generated, unspawned, in provisional phase, in coordinated phase or have crossed the region of interest

		
			if not data_file.real_time_spawning_flag:  #default 1 #NOT real-time spawning
				file = open(f"../data/compare_files/homogeneous_traffic/arr_{arr_rate_}/sim_obj_num_{sim}", 'rb')
				sim_obj = pickle.load(file)
				file.close()
				total_veh_in_simulation = functions.get_num_of_objects(sim_obj.unspawned_veh) 


			if data_file.real_time_spawning_flag:
				veh_id_var = 0
				#dep_veh_id  = [(_iter_+1)*10+(_iter_+1) for _iter_ in data_file.lanes] 
				dep_veh_id  = [(100*lane) for lane in data_file.lanes] 

                 
				# why is the next_spawn_time updated by  100+ max.time--- so that they never spawn
				next_spawn_time = [100 + data_file.max_sim_time for _ in data_file.lanes]  
				for lane in data_file.lanes:
					if not (arr_rate_array[lane] == 0):   #if only the arrival rate not zero we add poisson spawning else the previous values not changed
						next_spawn_time[lane] = round(time_track + float(np.random.exponential(1/arr_rate_array[lane],1)),1)

			#####
			### wait time for reaser end safety ??
			#####
			wait_till_time_on_lane = {} 
			for lane in data_file.lanes:
				wait_till_time_on_lane[lane] = 0     

			
			### start of simulation ###
			while (time_track < data_file.max_sim_time):  

				curr_time = time.time()

				########################
				# why break if done_veh > unspawned_veh--- in already genereated vehicle, when don_veh > total . work done
				# round(time_track, 1) >= round(next_spawn_time[lane] ----- same velocity constraint
				# (round(v.sp_t,1) < round(time_track,1)) --- velocity constraint
				# in the loop of unspwan iteration does it run till all the unspwan recieves a prov phase ---- yes
				#######################
				
				if (not data_file.real_time_spawning_flag):
					if (functions.get_num_of_objects(sim_obj.done_veh) >= total_veh_in_simulation):
						break

				### tracking time in whole numbers for batches of learning  ####
				if data_file.rl_algo_opt == "DDPG" and learning_flag and (agent.buffer.buffer_counter > 0) and ((time_track % 1) == 0):
						agent.buffer.learn()
						agent.update_target(agent.target_actor_.variables, agent.actor_model_.variables, agent.tau_)
						agent.update_target(agent.target_critic_.variables, agent.critic_model_.variables, agent.tau_)

				if learning_flag and (time_track % 100) == 0:
					agent.actor_model_.save_weights(f"../data/arr_{arr_rate_}/train_homo_stream/train_sim_{sim}/sim_data/trained_weights/actor_weights_itr_{int(time_track)}")
			
				#################### QUERIES #########################
	   			# unspawned_veh emptied - yes
				# spawned_veh emptied --  yes 
				# prior_coord_veh emptied -- yes 
				# prior_prov_veh emptied - yes
				# prov_veh emptied --- YES
				# coord_veh emptied  ---- YES
				#################### QUERIES ########################


				sp_list=[]
				prov_list=[]
				coord_list=[]
		
				
				#dep_veh_id  = [(_iter_+1)*10+(_iter_+1) for _iter_ in data_file.lanes] 
				##### spawning & rear end safety validation ##### 
				for lane in data_file.lanes: 


					if (data_file.real_time_spawning_flag) and (round(time_track, 1) >= round(next_spawn_time[lane], 1)) and (len(sim_obj.unspawned_veh[lane]) == 0) and (not (arr_rate_array[lane] == 0)):
						next_spawn_time[lane] = round(time_track + float(np.random.exponential(1/arr_rate_array[lane],1)),1)  #NEW SPAWN TIME
						new_veh = vehicle.Vehicle(lane, data_file.int_start[lane], 0, data_file.vm[lane], data_file.u_min[lane], data_file.u_max[lane], data_file.L, arr_rate_array)
						
						
						############### veh.ID ####################
						## for lane dependent ID
						#new_veh.id = copy.deepcopy(dep_veh_id[lane])
						#dep_veh_id[lane] += lane  #making the lane dependent ID
						# continuous ID
						new_veh.id = copy.deepcopy(veh_id_var)
						veh_id_var += 1
						############################################
						### update the list of spwan 
						#sp_list.append(new_veh.id)
						new_veh.sp_t = copy.deepcopy(time_track)
						# new_veh.arr = copy.deepcopy(arr_rate_array[lane])
						#print( veh_id_var)

						#[lane] += (50*lane +(lane+1))

						# UNSPAWNED SET: HERE UNSPAWNED SET DIFFERENT, the new robots generated are added to the unspawned set before PROV/COORD phase.
						#the unspawned set is called the set of robots the have been generated and yet to enter ROI
						#### the new_veh OBJECT appended to unspawned_veh   #####
						sim_obj.unspawned_veh[lane].append(copy.deepcopy(new_veh))
						#print(veh_id_var)	
						#print(new_veh.id, veh_id_var,sim_obj.unspawned_veh[lane][-1].id)
			    	##### end of spawning  ##### 	
					

			    	##### rear end safety validation  ##### 	
					sim_obj.spawned_veh[lane] = copy.deepcopy(sim_obj.prov_veh[lane]) # spawned_veh with prov_veh
					n = len(sim_obj.unspawned_veh[lane])
					v_itr = 0
					while v_itr < n:
						v = sim_obj.unspawned_veh[lane][v_itr]
						pre_v = None
						if len(sim_obj.prov_veh[lane]) > 0:       
							pre_v = sim_obj.prov_veh[lane][-1]
						elif len(sim_obj.coord_veh[lane]) > 0:
							pre_v = sim_obj.coord_veh[lane][-1]

						if (round(v.sp_t,1) < round(time_track,1)) and (functions.check_init_config(v, pre_v, time_track)): # CHECKS REAR END SAFETY
							v.sp_t = round(time_track,1)	# step at whcih the particular vehicle spawned 	
							#v_temp = sim_obj.unspawned_veh[lane].popleft()
							sim_obj.unspawned_veh[lane].popleft()
							sim_obj.spawned_veh[lane].append(v)
							n = len(sim_obj.unspawned_veh[lane])
							#print(functions.get_num_of_objects(sim_obj.unspawned_veh))
							#print(functions.get_num_of_objects(sim_obj.spawned_veh))
						else : break	
						#elif  data_file.real_time_spawning_flag: 

						#### how this helps ???
						#next_spawn_time[lane] = round(next_spawn_time[lane] + data_file.dt, 1)  # move to next SIM step for spawning 	
						#sim_obj.spawned_veh[lane].append(sim_obj.prov_veh[lane][v_itr])	

				##### END - spawning & rear end safety validation ##### 
				#print(len(sim_obj.spawned_veh))							
				####### ********************LOGIC ANALYSIS NEEDED***********************
				sim_obj.prov_veh = [deque([]) for _ in range(len(data_file.lanes))] # ONLY AFTER SPAWNING
				####### ********************************************
				
					
				### RL decision ###
				###################
				# both prov and coord veh will be provided with prec_s
				# spawn_set consists of both prov and coords phase veh   ?????????( WHY COORDS  INSIDE IT)
				# spawn_set updated with prec/dem/tac_flag
				#### why coord phase sent to prece_dem ???
				###################
				coord_set = copy.deepcopy(sim_obj.coord_veh)
				spawn_set = copy.deepcopy(sim_obj.spawned_veh)
				if learning_flag:
						prev_state = curr_state
						prev_rew = curr_rew
				sim_obj.spawned_veh, pre_s, tc_flag, state_t, action_t, explore_flag = prece_dem.get_prece_tcflag(time_track, spawn_set, coord_set, agent, algo_option, learning_flag)
				if learning_flag:
						curr_state = state_t
						prev_act = curr_act
						curr_act = action_t
						curr_rew = 0
				### END - RL decision ###

				#### DEBUG #####
				# by deafult id spawned in proper order -- yes made to be sorted by default
				# check coord and prov phase ---- yes priors have been feeded with robot
				# WRITE MODULE TO PRINT current prov/coord veh id in a file    - 
				# PRINT TRAJE AND VERIFY FOR THE LANES
				# CREATE PRINT MODULE TO PRINT ALL OF THEM AND USE EFFECTIVELY
				# should number of robots spawned be equal for lanes ??????
				# why the vehicle files different ?????
				################


				##### sort of spawn_veh #####
				for le in data_file.lanes:  
					#splen = len(sim_obj.spawned_veh[le]) 
					#print("splen value",splen)
					#print(functions.get_num_of_objects(sim_obj.spawned_veh))
					#for ind in range(splen):
					ind = 0
					n= len(sim_obj.spawned_veh[le])
					#####for _ in range(n):print("lane no:",le,"spwan ID",sim_obj.spawned_veh[le][_].id,_,"time", time_track,"spawn_size:",n)
					while ind <n: 
						if sim_obj.spawned_veh[le][ind].tc_flag > 0.5: 
							sim_obj.prior_coord_veh[le].append(sim_obj.spawned_veh[le][ind])
							sim_obj.spawned_veh[le].popleft()
							n= len(sim_obj.spawned_veh[le])
						elif sim_obj.spawned_veh[le][ind].tc_flag <= 0.5: 
							ind__ = 0
							#for _ in range(ind,n): 
							while ind__ < n:  # if a preceeding vehicle is prov phase, succeeding veh shd be in prov phase
								########print("ind:",ind,"n:",n,"index:",ind__)
								########  sim_obj.spawned_veh[le][ind].tc_flag = 0.5
								sim_obj.prior_prov_veh[le].append(sim_obj.spawned_veh[le][ind__])
								sim_obj.spawned_veh[le].popleft()
								n= len(sim_obj.spawned_veh[le]) 
								#########print("lane no:",le,"prov ID",sim_obj.prior_prov_veh[le][ind__].id,_,"time", time_track,"prov_size:",len(sim_obj.prior_prov_veh[le]))
					
				############
				#  change the code to print it in a better format
				#############
				
				""" 				
				print("complete-list")
				for lle in data_file.lanes:  
					print("lane no:",lle,"\n")
					for _iter in range(len(sim_obj.unspawned_veh[lle])): print("unspawn ID-index",sim_obj.unspawned_veh[lle][_iter].id,_iter,"time:",time_track,end=" ")
					print("\n")
					for _iter in range(len(sim_obj.spawned_veh[lle])): print("spawn ID-index",sim_obj.spawned_veh[lle][_iter].id,_iter,"time:",time_track,end=" ")
					print("\n")
					for __iter in range(len(sim_obj.prior_prov_veh[lle])): print("prov ID-index",sim_obj.prior_prov_veh[lle][__iter].id,__iter,"time:",time_track,end=" ")
					print("\n")
					for ___iter in range(len(sim_obj.prior_coord_veh[lle])): print("coord ID-index",sim_obj.prior_coord_veh[lle][___iter].id,___iter,"time:",time_track,end=" ")
				print("END-sort") """


				#for iter_ in range(len(sim_obj.prior_prov_veh[le])):print("lane no:",le,"prov ID:",sim_obj.prov_veh[le][iter_].id,iter_,"time:", time_track,n,"\n")
				#for iter__ in range(len(sim_obj.prior_coord_veh[le])):print("lane no:",le,"coord ID:",sim_obj.coord_veh[le][_].id,iter__,"time:", time_track,n,"\n")

				### coordinated phase  ###
				success = False
				if data_file.rl_flag:						
					num_of_veh = functions.get_num_of_objects(sim_obj.prior_coord_veh)
					prior_coord_veh_copy = copy.deepcopy(sim_obj.prior_coord_veh)
					coord_veh_copy = copy.deepcopy(sim_obj.coord_veh)
					len_lane_prior_coord_set = [0 for _ in data_file.lanes]
					len_lane_coord_set = [0 for _ in data_file.lanes]
					for _l in data_file.lanes:
						len_lane_prior_coord_set[_l] = copy.deepcopy(len([_l]))
						len_lane_coord_set[_l] = copy.deepcopy(len(coord_veh_copy[_l]))						
					sim_obj.coord_veh, cp_cost, coord_cost_with_comb_opt_para,success = coord_phase.coord_algo(time_track, prior_coord_veh_copy, coord_veh_copy,algo_option, learning_flag, 1,sim, train_iter, train_sim, pre_s, dem_s, tc_flag)
					#sim_obj.coord_veh, cp_cost, coord_cost_with_comb_opt_para, state_t, action_t, success = coord_phase.coord_algo(time_track, prov_veh_copy__copy, coord_veh_copy__copy, algo_option, agent, learning_flag, 1, sim, train_iter, train_sim)
					
					### update the list of spwan 
					#for iter in data_file.lanes:
					#	for j in range(len(sim_obj.coord_veh[iter])):
					#		coord_list.append(sim_obj.coord_veh[iter][j].id)

					if (num_of_veh > 0):
						#rl_ret = coord_cost_with_comb_opt_para / num_of_veh
						rl_ret = coord_cost_with_comb_opt_para #/ num_of_veh
						curr_rew = rl_ret
						rl_ret_collection.append(rl_ret)
						#if comb_test_flag:	comb_opt_test.append(comb_cost_test / num_of_veh)
						#else: comb_opt_test.append(0)
					else:
						rl_ret = None

                                         
					### storing data in buffer
					if (algo_option == "rl_modified_ddswa") and (learning_flag):
						if (not (prev_rew == None)) and (not (len(prev_state) == 0)):
							agent.buffer.remember((prev_state, prev_act, prev_rew, curr_state))
							qwe = {}
							qwe["state_buffer"] = agent.buffer.state_buffer
							qwe["action_buffer"] = agent.buffer.action_buffer
							qwe["reward_buffer"] = agent.buffer.reward_buffer
							qwe["next_state_buffer"] = agent.buffer.next_state_buffer
							dbfile = open(f'../data/arr_{arr_rate_}/train_homo_stream/train_sim_{sim}/replay_buffer_sim_{sim}', 'wb')
							pickle.dump(qwe, dbfile)
							dbfile.close()

					#sim_obj.prov_veh = [deque([]) for _ in range(len(data_file.lanes))]
					sim_obj.prior_coord_veh = [deque([]) for _ in range(len(data_file.lanes))]


					##### query ##########
					# does coord_phase file remove objects from prior_coord_veh when assigned with coord trajectories --- yes
					# prov_veh_copy__copy objects that don't cross intersection in coord phase-----(so we perform a prov phase again for them)
					##### query ##########
					
					### wht if a new robot comes to a stage without prior prov_phase ??? 
					
					if functions.get_num_of_objects(prior_coord_veh_copy) != 0:
						for lane in data_file.lanes:
							for v in prior_coord_veh_copy[lane]:
								#################
								## prov_veh made zero at start then ?????
								## prov_veh = copy.deepcopy(v) ---- direct copies delete the existing vehicle in prov_veh ??????	
								################
								pre_v = None
								if len(sim_obj.prov_veh[lane]) > 0:
									pre_v = sim_obj.prov_veh[lane][-1]
								elif len(sim_obj.coord_veh[lane]) > 0:
									pre_v = sim_obj.coord_veh[lane][-1]
										
								prov_veh = copy.deepcopy(v)
								prov_veh, prov_sucess = prov_phase.prov_phase(prov_veh, pre_v, time_track, wait_till_time_on_lane)

								assert len(prov_veh.p_traj) > len(v.p_traj)

								sim_obj.prov_veh[lane].append(copy.deepcopy(prov_veh))
				### END - coordinate phase end ###
					
						
				### provisional phase start ####
				for lane in data_file.lanes:
					n = len(sim_obj.prior_prov_veh[lane])
					iter = 0
					#################
					## prov_veh made zero at start then ?????
					################
					while iter<n:
						v = sim_obj.prior_prov_veh[lane][iter]
						pre_v = None
						if len(sim_obj.prov_veh[lane]) > 0: 
							pre_v = sim_obj.prov_veh[lane][-1]
						elif len(sim_obj.coord_veh[lane]) > 0: 	
							pre_v = sim_obj.coord_veh[lane][-1]
						prov_sucess = False
						prov_veh = copy.deepcopy(v)
						prov_veh, prov_sucess = prov_phase.prov_phase(prov_veh, pre_v, time_track, wait_till_time_on_lane)
						sim_obj.prov_veh[lane].append(copy.deepcopy(prov_veh))
						sim_obj.prior_prov_veh[lane].popleft()
						n = len(sim_obj.prior_prov_veh[lane])
				### provisional end #### """

				### update current time###
				time_track = round((time_track + data_file.dt), 1)

				if learning_flag:
					...
					print(f"arr_rate: {arr_rate_}, rep: {rep_sim}", "current time:", time_track, "sim:", sim, "train_iter:", train_iter)#, end="\r")
				else:
					print("current time:", time_track, "sim:", sim, "train_sim: ", train_sim, "train_iter:", train_iter, "arr_rate: ", arr_rate_, "heuristic:", data_file.used_heuristic, "................")#, end="\r")
				
				### update current time###

				### throuput calculation ###
				for l in data_file.lanes:
					for v in sim_obj.coord_veh[l]:
						t_ind = functions.find_index(v, time_track)

						if  ((t_ind == None) or (v.p_traj[t_ind] > (v.intsize + data_file.L))) and (v.id >= throughput_id_in_lane[l]):

							throughput_id_in_lane[l] = copy.deepcopy(v.id)
							cumulative_throuput_count += 1
						else:
							break

				# with open(f"../data/arr_{arr_rate_}/train_homo_stream/train_sim_{sim}/sim_data/throughput.csv", "a", newline="") as f:
				# 	writer = csv.writer(f)
				# 	#for j in wri:
				# 	writer.writerows([[time_track, cumulative_throuput_count]])

				### throuput calculation ###

				### removing vehicles which have crossed the region of interest ###
				for l in data_file.lanes:
					n_in_coord = len(sim_obj.coord_veh[l])
					v_ind = 0
					while v_ind < n_in_coord:
						coord_while_flag = 0
						v = sim_obj.coord_veh[l][v_ind]
						t_ind = functions.find_index(v, time_track)

						if (t_ind == None) or (v.p_traj[t_ind] > (v.intsize + v.length - v.int_start)):
									
							sim_obj.done_veh[v.lane].append(v)
							sim_obj.coord_veh[v.lane].popleft()
							#del v
							n_in_coord -= 1

						else:
							break
				### removed vehicles which have crossed the region of interest ###

			### end of simulation ###
		
			
    #### streamline ####






if __name__ == '__main__':

	arr_rates_to_sim = data_file.arr_rates_to_simulate  #The 10 diff values from 0.01 to 0.1

	args = []


	if data_file.used_heuristic == None:
	
		if data_file.rl_flag:

			train_or_test = str(sys.argv[1])


#
			if train_or_test == "--train":
				for _train_iter in range(1):
					for _sim_num in range(1, 2):
						for _arr_rate_ in arr_rates_to_sim:
							arr_rate_array_ = {0:0, 1:_arr_rate_, 2:_arr_rate_, 3:0, 4:_arr_rate_, 5:_arr_rate_, 6:0, 7:_arr_rate_, 8:_arr_rate_, 9:0, 10:_arr_rate_, 11:_arr_rate_} #arr_rate*np.ones(len(lanes)) #
							args.append([-1, _sim_num, arr_rate_array_, _arr_rate_, 0])
							# func(args[-1])
				# pool = Pool(10)
				pool = Pool(10)
				pool.map(func, args)
# pool(10) parallel procesing of diff arrival rate, how parametres for each  processing split? ?				

			elif train_or_test == "--test":
				delete_data.rem()  # deletes previous veh objects

				if not data_file.run_coord_on_captured_snap:

					_train_iter_list = [int(sys.argv[2])]

					for _train_iter in _train_iter_list:
						for _sim_num in range(1, 11):   # each policy run at speicific arrival rate for 10 times to increase the samples.
							for _train_sim in list(range(1, 11)):
								for _arr_rate_ in arr_rates_to_sim:
									arr_rate_array_ = {0:0, 1:_arr_rate_, 2:_arr_rate_, 3:0, 4:_arr_rate_, 5:_arr_rate_, 6:0, 7:_arr_rate_, 8:_arr_rate_, 9:0, 10:_arr_rate_, 11:_arr_rate_} #arr_rate*np.ones(len(lanes)) #
									
									# file_path = f"../data/arr_{_arr_rate_}/test_homo_stream/train_sim_{_train_sim}/train_iter_{_train_iter}"								

									file_path = f"../data/arr_{_arr_rate_}/test_homo_stream/train_sim_{_train_sim}/train_iter_{_train_iter}/sim_{_sim_num}/sim_{_sim_num}_train_iter_{_train_iter}.png"

									try:
										with open(f"{file_path}") as f:
											f.close()

									except:
										# if len(list(os.listdir(f"{file_path}/pickobj_sim_{_sim_num}"))) == 0:
										args.append([_train_iter, _sim_num, arr_rate_array_, _arr_rate_, _train_sim])
										print(f"train_sim: {_train_sim}, train_iter: {_train_iter}, sim: {_sim_num}")

										# args.append([_train_iter, _sim_num, arr_rate_array_, _arr_rate_])
										# func(args[-1])
					# pool = Pool(18)
					pool = Pool(18)
					pool.map(func, args)

				else:
					_arr_rate_ = 0.08
					for _sim_num in range(1,4):
						args.append([5000,_sim_num,0,_arr_rate_,8])
						func(args[-1])				




		elif not data_file.run_coord_on_captured_snap:
			for _arr_rate_ in arr_rates_to_sim:
				arr_rate_array_ = {0:0, 1:_arr_rate_, 2:_arr_rate_, 3:0, 4:_arr_rate_, 5:_arr_rate_, 6:0, 7:_arr_rate_, 8:_arr_rate_, 9:0, 10:_arr_rate_, 11:_arr_rate_} #arr_rate*np.ones(len(lanes)) #
				for _sim_num in range(1, 101):
					args.append([0, _sim_num, arr_rate_array_, _arr_rate_, 0])
					func(args[-1])


		else:
			_arr_rate_ = 0.08
			for _sim_num in range(1,4):
				args.append([0,_sim_num,0,_arr_rate_,0])
				func(args[-1])


	else:

		for _train_iter in [0]:
			for _sim_num in range(1, 101): # 100 diff simulations
				for _train_sim in list(range(1)):
					for _arr_rate_ in arr_rates_to_sim:

						arr_rate_array_ = {0:0, 1:_arr_rate_, 2:_arr_rate_, 3:0, 4:_arr_rate_, 5:_arr_rate_, 6:0, 7:_arr_rate_, 8:_arr_rate_, 9:0, 10:_arr_rate_, 11:_arr_rate_} #arr_rate*np.ones(len(lanes)) #
						heuristics_pickobj_save_path = f"../data/{data_file.used_heuristic}/arr_{_arr_rate_}/pickobj_sim_{_sim_num}"

						# if len(os.listdir(f"{heuristics_pickobj_save_path}")) < 290:
						args.append([_train_iter, _sim_num, arr_rate_array_, _arr_rate_, _train_sim])
						
						# else:
						# 	...

						# print(f"train_sim: {_train_sim}, train_iter: {_train_iter}, sim: {_sim_num}")

		pool = Pool(18)
		pool.map(func, args)








