import matplotlib.pyplot as plt
import numpy as np





def train_val_plot(train_history, val_history, save_path=None):
    """
    Plotting training loss and validation accuracy togather.

    - train_history: training loss (len = n_batch*n_epoch)
    - val_history: validation accuracy (len = n_epoch)
    - save_path: path to save the image
    """
    n_epoch = len(val_history)
    n_sample = len(train_history)
    n_batch = n_sample//n_epoch

    # create figure and axis objects with subplots()
    fig,ax = plt.subplots()
    # title
    plt.title('Training Loss and Validation Accuracy')

    # training loss
    ax.plot(range(len(train_history)), train_history, color="steelblue")
    # set x-axis label
    ax.set_xlabel("Training Batches",fontsize=14)
    # set y-axis label
    ax.set_ylabel("Training Loss",color="steelblue",fontsize=14)

    # x axis of validation accuracy
    val_x = [n_batch*i for i in range(n_epoch)]
    # for i in range(n_epoch):
    #     val_x += [x*log_interval+i*len(train_loader) for x in range(1+len(train_loader)//log_interval)]

    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(val_x, val_history,color="darkorange")
    ax2.set_ylabel("Val Accuracy (%)",color="darkorange",fontsize=14)

    plt.show()

    # save the plot as a file
    if save_path:
        fig.savefig(save_path,
                    format='jpeg',
                    dpi=100,
                    bbox_inches='tight')