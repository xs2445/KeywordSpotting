from utils.dataset import SpectrogramDataset

num_list = [1000,2000,3000,5000,6000,7000,8000,9000,10000,15000,20000]

path = './data/speech_commands_v1'

for f_max in num_list:
    dataset_name = 'mats/n32-q3-a1-100-'+str(f_max)
    save_path = 'imgs/n32-q3-a1-100-'+str(f_max)
    print("Converting Dataset:", dataset_name[5:])
    SpectrogramDataset.mat2np(path, dataset_name, save_path)
    # print(os.path.join(path, dataset_name), os.path.join(path, save_path))