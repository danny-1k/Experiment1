import matplotlib.pyplot as plt


def save_metrics(total_train,total_test, total_accuracy, header):
    x = list(range(1, len(total_train)+1))

    plt.plot(x, total_train, label='Train loss over time')
    plt.plot(x, total_test, label='Test loss over time')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()

    plt.savefig(f'{header}_loss_plot.png')
    
    plt.close('all')

    plt.plot(x, total_accuracy, label='Accuracy over time (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(f'{header}_accuracy_plot.png')

    plt.close('all')

