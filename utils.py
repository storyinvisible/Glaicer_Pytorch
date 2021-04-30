import matplotlib.pyplot as plt
import numpy as np


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


def plot_actual(actual, pred_test, pred_train, show=False):
    plt.figure()
    plt.plot(actual, label="Actual")
    plt.plot(pred_train, label="Train_pred")
    zeros = np.zeros_like(pred_train)
    test = np.append(zeros, pred_test)
    plt.plot(test, label="Test_pred")
    plt.legend(loc="upper left")
    if show:
        plt.show()
    return plt
