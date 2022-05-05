from dataset import SpectrogramDataset

num_list = [1,4,8,12,16,20,24,28,36,40,44,48,52,56,60,64]

path = '../data/speech_commands_v1'

for n in num_list:
    dataset_name = 'mats/n'+str(n)+'-q3-a1-100-4000'
    save_path = 'imgs/n'+str(n)+'-q3-a1-100-4000'
    print("Converting Dataset:", dataset_name[5:])
    SpectrogramDataset.mat2np(path, dataset_name, save_path)
    # print(os.path.join(path, dataset_name), os.path.join(path, save_path))