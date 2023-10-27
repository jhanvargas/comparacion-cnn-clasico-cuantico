# External libraries
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import Aer
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.opflow import AerPauliExpectation
from qiskit.utils import QuantumInstance
from qiskit.visualization import circuit_drawer
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import TwoLayerQNN

# Own libraries
from python.ibm_quantum.utils.connect import get_ibm_quantum


class HybridCNN(nn.Module):
    """Clase que define una red neuronal convolucional (CNN) en PyTorch.

    Args:
        Ninguno.

    Attributes:
        conv0 (nn.Conv2d): Capa de convolución 2D.
        bn0 (nn.BatchNorm2d): Capa de normalización por lotes.
        max_pool (nn.MaxPool2d): Capa de max-pooling.
        conv1 (nn.Conv2d): Capa de convolución 2D.
        bn1 (nn.BatchNorm2d): Capa de normalización por lotes.
        conv2 (nn.Conv2d): Capa de convolución 2D.
        bn2 (nn.BatchNorm2d): Capa de normalización por lotes.
        conv3 (nn.Conv2d): Capa de convolución 2D.
        bn3 (nn.BatchNorm2d): Capa de normalización por lotes.
        dropout (nn.Dropout): Capa de dropout para regularización.
        fc0 (nn.Linear): Capa de conexión completamente conectada.
        fc1 (nn.Linear): Capa de conexión completamente conectada.
        fc2 (nn.Linear): Capa de conexión completamente conectada.

    Methods:
        forward(x): Define el flujo hacia adelante de la red.

    Examples:
        model = TorchCNN()

    """

    def __init__(self, qnn: TwoLayerQNN):
        """Inicializa una instancia de la clase TorchCNN."""

        super().__init__()
        self.conv0 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.bn0 = nn.BatchNorm2d(num_features=16)
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv1 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=32)
        # self.max_pool

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(num_features=64)
        # self.max_pool

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(num_features=128)
        # self.max_pool

        self.dropout = nn.Dropout(p=0.5)
        self.fc0 = nn.Linear(in_features=128 * 2 * 2, out_features=64)
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=2)
        self.qnn = TorchConnector(qnn)
        self.fc3 = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define el flujo hacia adelante de la red.

        Args:
            x: Tensor de entrada.

        Returns:
            Tensor de salida.

        """
        x = F.relu(self.bn0(self.conv0(x)))
        x = self.max_pool(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.max_pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.max_pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.max_pool(x)

        x = x.reshape(x.shape[0], -1)

        x = self.dropout(x)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.qnn(x)
        x = F.sigmoid(self.fc3(x))

        return x


def create_qnn(backend: bool = False) -> TwoLayerQNN:
    """Creates and configures a Quantum Neural Network (QNN) using Qiskit.

    This function constructs a QNN by combining a quantum feature map and a
    quantum ansatz.
    The QNN is configured with input parameters and weight parameters,
    and input gradients are enabled for quantum gradient-based optimization.

    Args:
        backend

    Returns:
         A configured QNN for quantum machine learning.

    Example:
        qnn = create_qnn()

    """
    # https://quantum-computing.ibm.com/services/resources?tab=yours&system=ibm_nairobi&view=table

    if backend:
        backend = get_ibm_quantum()
    else:
        backend = QuantumInstance(Aer.get_backend('statevector_simulator'))

    feature_map = ZZFeatureMap(2)
    ansatz = RealAmplitudes(2, reps=1, entanglement='linear')

    print(ansatz.parameters)

    # param_dict = {param: np.pi / 4 for param in ansatz.parameters}
    # ansatz.assign_parameters(param_dict)

    qnn = TwoLayerQNN(
        2, feature_map, ansatz,
        exp_val=AerPauliExpectation(),
        quantum_instance=backend,
    )

    return qnn


def save_q_circuit(qnn: TwoLayerQNN, path_save: str = None) -> None:
    """Guarda un circuito cuántico en una representación gráfica.

    Args:
        qnn: El Quantum Neural Network (QNN) cuyo circuito se va a guardar.
        path_save: La ruta donde se guardará la imagen del circuito.

    Example:
        save_q_circuit(qnn, "circuit.png")

    """
    circuit_drawer(qnn.circuit, output='mpl')

    if path_save:
        plt.savefig(path_save)

    plt.show()
