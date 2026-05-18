from __future__ import print_function
from os.path import join
from get_datasets import prefix_data
import matplotlib.pyplot as plt


def plot_graphs(
    opt,
    test_recalls,
    train_recall_at_1,
    all_loss,
    start_epoch,
    curr_epoch,
    unique_string,
):
    epochs = list(range(start_epoch, curr_epoch))
    # Plot for all_loss
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
    plt.plot(epochs, all_loss, label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
    plt.plot(epochs, test_recalls[0], label="Test Recall@1")
    plt.plot(epochs, test_recalls[1], label="Test Recall@5")
    plt.plot(epochs, test_recalls[2], label="Test Recall@10")
    plt.plot(epochs, train_recall_at_1, label="Train Recall@1")

    plt.xlabel("Epochs")
    plt.ylabel("Recall")
    plt.title(f"{opt.dataset},{opt.seqL}Recall over Epochs")
    plt.legend()

    plt.tight_layout()
    print("unique string: ", unique_string)
    plt.savefig(
        join(
            prefix_data, "run_graphs/{}_skip{}_plot.png".format(unique_string, opt.skip)
        )
    )
    plt.show()