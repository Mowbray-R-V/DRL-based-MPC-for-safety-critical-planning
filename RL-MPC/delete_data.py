import os
import glob
import data_file

#arr_rate = 0.06
#train_sim = 1
train_iter = 100000 
#sim_num = 1

train_sim_list = [i for i in range(1,11) ]
sim_num = [i for i in range(1,11) ]


""" #path = f"/home/user/mowbrayr/AIM-code/data/arr_{arr_rate}/test_homo_stream/train_sim_{train_sim}/train_iter_{train_iter}/pickobj_sim_{sim_num}/"
print(os.getcwd())
os.chdir(path)
print(os.getcwd()) """


def rem():
    for _ in data_file.arr_rates_to_simulate:
        for __ in train_sim_list:
            for ___ in sim_num:
                files = glob.glob(f"/home/user/mowbrayr/AIM-code/data/arr_{_}/test_homo_stream/train_sim_{__}/train_iter_{train_iter}/pickobj_sim_{___}/*")
                for f in files:
                    os.remove(f)


