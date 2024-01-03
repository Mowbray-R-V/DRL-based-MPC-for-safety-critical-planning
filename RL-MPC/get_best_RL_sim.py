import os
import numpy as np
import csv
from matplotlib import pyplot as plt
import pickle

import data_file


def stream_norm_dist_and_vel_test_using_pickobj(_tr_iter_, arr_rate):

	avg_time_to_cross_norm_dist_and_vel_all_arr = []

	avg_time_to_cross_comb_opt_all_arr = []

	avg_obj_fun_norm_dist_and_vel_all_arr = []

	avg_obj_fun_comb_opt_all_arr = []

	sp_t_limit = 200


	avg_true_arr_rate = []

	

	
	# arr_rate = [] #round(0.1, 1)			

	train_iter_list = [_tr_iter_]

	num_train_iter = len(train_iter_list)

	sim_list = list(range(1, 11)) # + list(range(6, 11)) # list(range(3, 5)) + list(range(6, 9))

	train_sim_list = [2] #list(range(1, 11)) #list(range(1,4)) + [5] + list(range(8,11))

	num_sim = len(sim_list)

	test_data = {}
	throughput = {}
	ttc = {}
	all_ttc = []
	exit_vel = {}

	percentage_comparison_dict = {}
	throughput_ratio_dict = {}


	total_comb_opt_veh = {}

	comb_opt_veh_dict = {}
	
	total_veh_num = {}
	heuristic_veh_dict = {}

	train_sim = 1
	train_iter = 5000	

	for heuristic_ind in range(0, 2):
		heuristic = data_file.heuristic_dict[heuristic_ind]
		if heuristic == None:
			heuristic = 'RL'
		test_data[heuristic] = {}
		for sim in sim_list:

			if heuristic == 'RL':
				test_data_file_path = f"../data/arr_{arr_rate}/test_homo_stream/train_sim_{train_sim}/train_iter_{train_iter}"

			else:
				test_data_file_path = f"../data/{heuristic}/arr_{arr_rate}" #/train_sim_{train_sim}/train_iter_{train_iter}"
			
			test_data[heuristic][sim] = 0
			
			veh_num = 0
			temp = 0
			temp_ttc = 0
			temp_exit_vel = 0

			for c in os.listdir(f"{test_data_file_path}/pickobj_sim_{sim}"):
				try:
					file = open(f"{test_data_file_path}/pickobj_sim_{sim}/{c}",'rb')
					object_file = pickle.load(file)
					file.close()
				except:

					continue

				if (object_file[int(c)].sp_t > sp_t_limit): # or (object_file[int(c)].sp_t < 90.5):
					continue
				
				else:

					try:
						temp += object_file[int(c)].priority * (object_file[int(c)].p_traj[int(data_file.T_sc/data_file.dt) -1] - object_file[int(c)].p0)
						index_var = 0
						veh_num += 1

						for time, pos in zip(object_file[int(c)].t_ser, object_file[int(c)].p_traj):
							if pos >= object_file[int(c)].length + object_file[int(c)].intsize:
								temp_ttc += (time - object_file[int(c)].t_ser[0]) # object_file[int(c)].priority * 
								# all_ttc.append(time - object_file[int(c)].t_ser[0])
								# throughput[train_iter][train_sim][sim] += 1
								# temp_exit_vel += object_file[int(c)].v_traj[index_var]
								break

							index_var += 1
					
					except IndexError:
						continue

			
			print(f"heuristic: {heuristic}, arr_rate: {arr_rate}, sim: {sim}, veh_num: {veh_num}, avg_ttc: {temp_ttc/veh_num} ...................") #, end="\r") # 
			test_data[heuristic][sim] += temp

	
	for heuristic_ind in [1]: #, 2, 3, 4]:
		temp_obj_list = []
		heuristic = data_file.heuristic_dict[heuristic_ind]
		if heuristic == None:
			heuristic = 'RL'
		for sim in sim_list:
			temp_obj_list.append(((test_data['RL'][sim] / test_data[heuristic][sim]) - 1)*100)
		print(f"\nheirustic:{heuristic}\tgaps:{temp_obj_list}\n")


		
if __name__ == "__main__":

	args_in = []
	iter_list = [5000]
	arr_rate_list = data_file.arr_rates_to_simulate
	for tr_iter in iter_list:
		for arr_rate_ in arr_rate_list:
			args_in.append([tr_iter, arr_rate_])
			stream_norm_dist_and_vel_test_using_pickobj(tr_iter, arr_rate_)