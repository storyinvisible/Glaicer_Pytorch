import matplotlib.pyplot as plt


def plot_loss(train_loss, test_loss, show=False):
    plt.figure()
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.title("Loss plot")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(['Train', 'Test'], loc='upper left')
    if show:
        plt.show()
    return plt
