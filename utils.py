import numpy as np
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


def plot_smb(actual_train, actual_test, pred_train, pred_test, train_start_year, test_start_year, show=False):
    plt.figure()
    start, end = train_start_year, test_start_year + len(actual_test)
    train_year_range = np.arange(start, start + len(actual_train))
    test_year_range = np.arange(test_start_year, end)
    plt.plot(train_year_range, actual_train, color="red", linewidth=2)
    plt.plot(train_year_range, pred_train, color="blue", linestyle='--')
    actual, = plt.plot(test_year_range, actual_test, color="red", linewidth=2)
    predict, = plt.plot(test_year_range, pred_test, color="blue", linestyle='--')
    plt.ylabel("dm/dt")
    plt.xlabel("year")
    plt.legend([actual, predict], ["Actual", "Predict"], loc="upper left")
    if show:
        plt.show()
    return plt
