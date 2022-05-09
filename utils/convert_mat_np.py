import os
from dataset import SpectrogramDataset

path = '../data/speech_commands_v1'

# num_list = [1,4,8,12,16,20,24,28,36,40,44,48,52,56,60,64]

# for n in num_list:
#     dataset_name = 'mats/n'+str(n)+'-q3-a1-100-4000'
#     save_path = 'imgs/n'+str(n)+'-q3-a1-100-4000'
#     print("Converting Dataset:", dataset_name[5:])
#     SpectrogramDataset.mat2np(path, dataset_name, save_path)
#     # print(os.path.join(path, dataset_name), os.path.join(path, save_path))

# num_list = [0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 2.5, 3]
# num_list = [3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 15, 20, 30]
# num_list = [3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 30]
num_list = [15, 20]

for q in num_list:
    dataset_name = 'mats/n32-q'+str(q)+'-a1-100-4000'
    save_path = 'imgs/n32-q'+str(q)+'-a1-100-4000'
    print("Converting Dataset:", dataset_name[5:])
    SpectrogramDataset.mat2np(path, dataset_name, save_path)
    # print(os.path.join(path, dataset_name), os.path.join(path, save_path))