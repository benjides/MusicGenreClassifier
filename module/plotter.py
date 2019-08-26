import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap='Greens'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()




def confussion_plot(y_true, y_pred, labels):

    ncols = 8
    nrows = ceil(len(labels) / ncols)
    _, axs = plt.subplots(nrows=nrows, ncols=ncols, constrained_layout=True)

    for (index, ax) in zip(range(len(labels)), axs.flatten()):

        y_true_label = y_true[:, index]
        y_pred_label = y_pred[:, index]

        cm = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label)

        sns.heatmap(
            cm, 
            annot=True, 
            ax=ax, 
            fmt='g', 
            cmap='Greens', 
            cbar=False, 
            yticklabels=False, 
            xticklabels=False
        )
        ax.set_title(labels[index])
        

    plt.show()

def genre_distribution(genres, labels):

    _, axs = plt.subplots(nrows=1, ncols=len(labels), constrained_layout=True)
    
    if len(labels) != 1:
        axs = axs.flatten()
    else :
        axs = [axs]

    for (distribution, label, ax) in zip(genres, labels, axs):
        sns.catplot(
            x='count',
            y="_id",
            kind='bar',
            ax=ax, 
            palette="ch:.25",
            data=distribution, 
        )
        ax.set_title(label)
        
    plt.show()