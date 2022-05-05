import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

from torchsummary import summary

from utils.dataset import SpectrogramDataset
from utils.model import SpeechResModel
from utils.train import train
from utils.utils import train_val_plot
from utils.logger import logger


path = "/home/jupyter/6692/project/data/speech_commands_v1/imgs"

# f_c_max
# num_list = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,15000,20000]
# DATASET_NAME_LIST = []
# for f_max in num_list:
#     DATASET_NAME_LIST.append('n32-q3-a1-100-'+str(f_max))
# LOG_FOLDER = './logs/' + datetime.today().strftime("%Y-%m-%d")

# n_filter
# num_list = [1,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64]
num_list = [40,44,48,52,56,60,64]

DATASET_NAME_LIST = []
for n in num_list:
    DATASET_NAME_LIST.append('n'+str(n)+'-q3-a1-100-4000')
LOG_FOLDER = './logs/' + datetime.today().strftime("%Y-%m-%d")
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

# print(DATASET_NAME_LIST)

# training parameters
batch_size = 64
epochs = 20
lr = [1e-3]
subset_frac = None
device = "cuda"
log_interval = 50
weight_decay = 1e-5

for dataset_name in DATASET_NAME_LIST:

    model_name = 'res15'
    dataset_log_folder = os.path.join(LOG_FOLDER, model_name + '-' + dataset_name)
    if not os.path.exists(dataset_log_folder):
        os.makedirs(dataset_log_folder)
    log_file_name = os.path.join(dataset_log_folder, model_name + '-' + dataset_name + '_log.txt')
    save_path = os.path.join(dataset_log_folder, model_name + '-' + dataset_name + '.pt')
    
    # genetate dataset
    train_set, test_set, val_set = SpectrogramDataset.split_dataset(path, dataset_name, split_rate=[0.8,0.1,0.1])
    data_shape = train_set.__getitem__(0)[0].numpy().shape
    model = SpeechResModel(len(train_set.class_names), model_name)

    # start training
    logger(log_file_name, console_print=True)
    print("Date:",datetime.today(), '\n')
    print("Model name:", model_name)
    print("Dataset:", dataset_name)
    print("Input shape:", data_shape)
    # print(model)
    summary(model.cuda(), data_shape)

    train_loss_history, val_loss_history, train_acc_history, val_acc_history = train(
        model, 
        train_set, val_set, test_set,
        batch_size, epochs, lr, weight_decay,
        save_path, subset_frac, device, log_interval)

    print("Best validation acc:{:4f}".format(np.max(val_acc_history)))

    np.save(os.path.join(dataset_log_folder, 'train_loss_hist.npy'), train_loss_history)
    np.save(os.path.join(dataset_log_folder, 'val_loss_hist.npy'), val_loss_history)
    np.save(os.path.join(dataset_log_folder, 'train_acc_hist.npy'), train_acc_history)
    np.save(os.path.join(dataset_log_folder, 'val_acc_hist.npy'), val_acc_history)

    train_val_plot(train_loss_history, val_acc_history, save_path=os.path.join(dataset_log_folder, model_name+'_loss.jpg'))
    train_val_plot(train_acc_history, val_acc_history, 
               save_path=os.path.join(dataset_log_folder, model_name+'_acc.jpg'), 
               title='Training Accuracy and Validation Accuracy', 
               y_label_1='Training Accuracy')
