# External libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from torch import cat, no_grad
from torch.nn import Conv2d, Dropout2d, Linear, Module, NLLLoss

# Own libraries
from python.metadata.path import Path


class Net(Module):
    """A neural network model that combines classical and quantum components.

    This class defines a neural network architecture that combines classical
    deep learning layers with a Quantum Neural Network (QNN) using Qiskit.
    The QNN is integrated into the model using the TorchConnector from Qiskit's
    machine learning module.

    Args:
        qnn (EstimatorQNN): A Quantum Neural Network (QNN) from Qiskit.

    Attributes:
        conv1 (Conv2d): The first 2D convolutional layer.
        conv2 (Conv2d): The second 2D convolutional layer.
        dropout (Dropout2d): A 2D dropout layer.
        fc1 (Linear): The first fully connected layer.
        fc2 (Linear): The second fully connected layer for QNN input.
        qnn (TorchConnector): The Quantum Neural Network (QNN) connector.
        fc3 (Linear): The fully connected layer for QNN output.

    """

    def __init__(self, qnn: EstimatorQNN):
        super().__init__()
        self.conv1 = Conv2d(1, 2, kernel_size=5)
        self.conv2 = Conv2d(2, 16, kernel_size=5)
        self.dropout = Dropout2d()
        self.fc1 = Linear(256, 64)
        self.fc2 = Linear(64, 2)  # 2-dimensional input to QNN
        self.qnn = TorchConnector(qnn)  # Apply torch connector, weights chosen
        # uniformly at random from interval [-1,1].
        self.fc3 = Linear(1, 1)  # 1-dimensional output from QNN

    def forward(self, x: np.array) -> np.array:
        """Forward pass through the neural network.

        Args:
            x (Tensor): Input data tensor.

        Returns:
            Tensor: Output tensor after passing through the network.

        """
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.qnn(x)  # apply QNN
        x = self.fc3(x)
        return cat((x, 1 - x), -1)


def create_qnn() -> EstimatorQNN:
    """Creates and configures a Quantum Neural Network (QNN) using Qiskit.

    This function constructs a QNN by combining a quantum feature map and a
    quantum ansatz.
    The QNN is configured with input parameters and weight parameters,
    and input gradients are enabled for quantum gradient-based optimization.

    Returns:
         A configured QNN for quantum machine learning.

    Example:
        qnn = create_qnn()

    """
    # circuito cuantico preestablecido. Validar que tipo de circuito es exactamente
    # validar simulador
    # validar consumo de recursos
    # menor margen de error
    # 5 Qbits ibm bogota - no bogota no belen
    # metricas de validación - matrix de confusión! F1, necesario 4 metrica
    # sistemas de auto-atención
    # transferencia de aprendizaje
    feature_map = ZZFeatureMap(2)
    ansatz = RealAmplitudes(2, reps=1)
    qc = QuantumCircuit(2)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
    )
    return qnn


def train_model(model: Module, data_loader: DataLoader) -> tuple:
    """Train a PyTorch model using the specified data loader.

    This function takes a PyTorch model, a data loader, and performs the
     training process. It defines the optimizer, loss function, and the number
     of training epochs. The training progress is printed, and the model's state
     dictionary is saved to a specified path.

    Args:
        model: The PyTorch model to be trained.
        data_loader: The DataLoader containing training data.

    Returns:
        tuple: A tuple containing the following elements:
            - loss_list (list): A list of training losses over epochs.
            - loss_func (torch.nn.Module): The loss function used for training.
            - total_loss (list): A list of individual losses for each training
             batch.

    Example:
        loss_list, loss_func, total_loss = train_model(my_model, train_loader)

    """
    # Define model, optimizer, and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = NLLLoss()

    # Start training
    epochs = 10  # Set number of epochs
    loss_list = []  # Store loss history
    model.train()  # Set model to training mode

    for epoch in range(epochs):
        total_loss = []
        for batch_idx, (data, target) in enumerate(data_loader):
            optimizer.zero_grad(set_to_none=True)  # Initialize gradient
            output = model(data)  # Forward pass
            loss = loss_func(output, target)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize weights
            total_loss.append(loss.item())  # Store loss
        loss_list.append(sum(total_loss) / len(total_loss))
        print(
            "Training [{:.0f}%]\tLoss: {:.4f}".format(
                100.0 * (epoch + 1) / epochs, loss_list[-1]
            )
        )

    torch.save(model.state_dict(), Path.qnn_model_example)

    return loss_list, loss_func, total_loss


def test_model(
    model_test: Module,
    data_loader: DataLoader,
    total_loss: list,
    loss_func: Module,
    batch_size: int,
) -> None:
    """Test a PyTorch model's performance on a test dataset.

    This function evaluates the performance of a PyTorch model on a test dataset
     using the specified data loader.
     It calculates the test loss and accuracy, and also plots a subset of
     predicted labels for visualization.

    Args:
        model_test: The PyTorch model to be tested.
        data_loader: The DataLoader containing test data.
        total_loss: A list to store individual test losses.
        loss_func: The loss function used for testing.
        batch_size: The batch size used in the DataLoader.

    Returns:
        None

    Example:
        test_model(
            my_test_model,
            test_loader,
            test_loss_list,
            my_loss_func,
            batch_size=64,
        )

    """
    model_test.eval()  # set model to evaluation mode
    with no_grad():
        correct = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            output = model_test(data)
            if len(output.shape) == 1:
                output = output.reshape(1, *output.shape)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            loss = loss_func(output, target)
            total_loss.append(loss.item())

        print(
            "Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%".format(
                sum(total_loss) / len(total_loss),
                correct / len(data_loader) / batch_size * 100,
            )
        )

    # Plot predicted labels
    n_samples_show = 6
    count = 0
    fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

    model_test.eval()
    with no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if count == n_samples_show:
                break
            output = model_test(data[0:1])
            if len(output.shape) == 1:
                output = output.reshape(1, *output.shape)

            pred = output.argmax(dim=1, keepdim=True)

            axes[count].imshow(data[0].numpy().squeeze(), cmap="gray")

            axes[count].set_xticks([])
            axes[count].set_yticks([])
            axes[count].set_title("Predicted {}".format(pred.item()))

            count += 1
