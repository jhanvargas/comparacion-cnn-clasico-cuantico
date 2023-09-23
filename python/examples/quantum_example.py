# External libraries
import matplotlib.pyplot as plt
import torch
from qiskit.utils import algorithm_globals
from torch import manual_seed

# Own libraries
from python.metadata.path import Path
from python.utils.mnist_pytorch import generate_data, plot_data
from python.utils.qnn_qiskit import Net, create_qnn, test_model, train_model

# Set seed for random generators
algorithm_globals.random_seed = 42


if __name__ == '__main__':
    # Set train shuffle seed (for reproducibility)
    manual_seed(42)

    # Loading data
    train_loader = generate_data(n_samples=1000, batch_size=1)
    test_loader = generate_data(n_samples=100, batch_size=1, train=False)

    plot_data(train_loader, n_samples_show=6)

    # Define and create QNN
    qnn = create_qnn()
    model = Net(qnn)

    loss_list, loss_func, total_loss = train_model(model, train_loader)

    # Plot loss convergence
    plt.plot(loss_list)
    plt.title("Hybrid NN Training Convergence")
    plt.xlabel("Training Iterations")
    plt.ylabel("Neg. Log Likelihood Loss")
    plt.show()

    # Evaluate
    qnn_test = create_qnn()
    model_test = Net(qnn_test)
    model_test.load_state_dict(torch.load(Path.qnn_model_example))

    test_model(
        model_test=model_test,
        data_loader=test_loader,
        total_loss=total_loss,
        loss_func=loss_func,
        batch_size=1,
    )
