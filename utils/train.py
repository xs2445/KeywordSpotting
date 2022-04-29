import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

import os
import numpy as np
import copy
import time


def train(
    model, 
    train_dataset, val_dataset, test_dataset, 
    batch_size, epochs, lr, weight_decay=1e-5,
    save_path=None, subset_frac=None, device=None, log_interval=100):

    train_loader, val_loader, test_loader = make_frac_dataloader(
                            train_dataset, val_dataset, test_dataset, 
                            subset_frac, batch_size)
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    acc_high = 0

    criterion = torch.nn.CrossEntropyLoss()
    # criterion = F.nll_loss()

    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr = lr[0],
    #     weight_decay = weight_decay
    # )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = lr[0],
        weight_decay = weight_decay
    )

    if device:
        torch.cuda.set_device(0)
        print("Using gpu:", torch.cuda.get_device_name(0))
        model.cuda()
        torch.cuda.empty_cache()
    else:
        print("Using cpu")

    print("Training epoches:", epochs)
    print("Training batches:", train_loader.__len__())

    for epoch in range(epochs):
        epc_start = time.time()
        print("\nEpoch:", epoch+1)
        # training iteration
        for batch_idx, (data_in, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            
            if device:
                data_in = data_in.cuda()
                labels = labels.cuda()

            logits = model(data_in)
            loss = criterion(logits, labels)
            # loss = F.nll_loss(logits, labels)
            loss.backward()
            optimizer.step()
            # print the training loss and acc
            avg_train_acc = cal_eval(logits, labels)
            # avg_train_acc = print_eval("train step #{}/{}".format(batch_idx, len(train_loader)), logits, labels, loss)
            train_loss_history.append(loss.item())
            train_acc_history.append(avg_train_acc)
            if batch_idx % log_interval == 0:
                print_eval("train step #{}/{}".format(batch_idx, len(train_loader)), logits, labels, loss)
        
        # validation iteration
        model.eval()
        val_running_loss = []
        val_running_acc = []
        for batch_idx, (data_in, labels) in enumerate(val_loader): 
            if device:
                data_in = data_in.cuda()
                labels = labels.cuda()
            logits = model(data_in)
            loss = criterion(logits, labels)
            val_running_loss.append(loss.item())
            val_running_acc.append(cal_eval(logits, labels))
        val_loss_history.append(np.mean(val_running_loss))
        val_acc_history.append(np.mean(val_running_acc))

        print("Validation acc: {:5f}, loss: {:5f}".format(
            val_acc_history[-1],
            val_loss_history[-1]))

        if val_acc_history[-1] >= acc_high:
            acc_high = val_acc_history[-1]
            if save_path:
                print("saving best model ...")
                model.save(save_path)
            best_model = copy.deepcopy(model)
        
        evaluate(model, test_loader, device)
        epc_end = time.time()
        print("Cost time:{:2f}s".format(epc_end-epc_start))

    print()
    evaluate(best_model, test_loader, device)

    return train_loss_history, val_loss_history, train_acc_history, val_acc_history


def evaluate(model, test_loader, device=None):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    test_acc_list = []
    test_loss_list = []
    for data_in, labels in test_loader:
        if device:
            data_in = data_in.cuda()
            labels = labels.cuda()
        logits = model(data_in)
        loss = criterion(logits, labels)
        acc = cal_eval(logits, labels)
        test_acc_list.append(acc)
        test_loss_list.append(loss.item())
    avg_acc = np.mean(test_acc_list)
    avg_loss = np.mean(test_loss_list)
    print("Test acc: {:5f}, loss: {:5f}".format(
        avg_acc, avg_loss))


def print_eval(name, scores, labels, loss, end="\n"):
    """
    print the evaluation result
    """
    accuracy = cal_eval(scores, labels)
    loss = loss.item()
    print("{} acc: {:5f}, loss: {:5f}".format(name, accuracy, loss), end=end)


    return accuracy

def cal_eval(scores, labels):
    """
    calculate the evaluation result
    """
    batch_size = labels.size(0)
    accuracy = (torch.max(scores, 1)[1].view(batch_size).data == labels.data).float().sum() / batch_size
    return accuracy.item()


def make_frac_dataloader(train_dataset, val_dataset, test_dataset, 
                subset_frac, batch_size):

    if subset_frac:
        # use a subset of dataset
        train_indices = list(np.random.randint(0, len(train_dataset), 
            size=int(len(train_dataset)*subset_frac)))
        val_indices = list(np.random.randint(0, len(val_dataset), 
            size=int(len(val_dataset)*subset_frac)))
        test_indices = list(np.random.randint(0, len(test_dataset), 
            size=int(len(test_dataset)*subset_frac)))
        # subset samplers
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)    
        print("traning sample:{}".format(len(train_indices)))
        print("validation sample:{}".format(len(val_indices)))
        print("testing sample:{}\n".format(len(test_indices)))
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
        print("traning sample:{}".format(len(train_dataset)))
        print("validation sample:{}".format(len(val_dataset)))
        print("testing sample:{}\n".format(len(test_dataset)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False, 
        drop_last=True, 
        sampler=train_sampler)
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(len(val_dataset), batch_size),
        shuffle=False,
        sampler=val_sampler)
    test_loader = DataLoader(
        test_dataset,
        batch_size=min(len(test_dataset), batch_size),
        shuffle=False,
        sampler=test_sampler)
    
    return train_loader, val_loader, test_loader

