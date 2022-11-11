from matplotlib import pyplot as plt


def plot_2D_scatter(X_train, X_test):
    
    print("Plotting is starting...")
    X_test = X_test.cpu().detach().numpy()
    X_train = X_train.cpu().detach().numpy()
    
    fig, ax = plt.subplots()
    ax.scatter(X_train[:, 0], X_train[:, 1], c="blue", label="train", alpha=0.5)
    ax.scatter(X_test[:, 0], X_test[:, 1], c="red", label="test", alpha=0.5)
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    plt.legend(loc="upper left")
    plt.savefig("plot.png")   
    print("Plotting is saved as plot.png.")