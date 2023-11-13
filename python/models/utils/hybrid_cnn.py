# External libraries
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import Aer
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap, TwoLocal
from qiskit.opflow import AerPauliExpectation
from qiskit.utils import QuantumInstance
from qiskit.visualization import circuit_drawer
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import TwoLayerQNN
import pennylane as qml
import numpy as np
# Own libraries
from python.ibm_quantum.utils.connect import get_ibm_quantum
from qiskit.circuit import QuantumCircuit, Parameter


def custom_ansatz(num_qubits):
    circuit = QuantumCircuit(num_qubits)
    params = [Parameter(f"θ{i}") for i in range(3 * num_qubits)]
    param_iter = iter(params)

    for qubit in range(num_qubits):
        circuit.rx(next(param_iter), qubit)
        circuit.ry(next(param_iter), qubit)
        circuit.rz(next(param_iter), qubit)

    # Agregar más operaciones si lo deseas, como entrelazamiento entre los qubits
    # circuit.cz(0, 1) # Por ejemplo, una puerta CZ entre qubits 0 y 1

    return circuit, params


n_qubits = 2
n_layers = 6
dev = qml.device("default.qubit", wires=n_qubits)
# weight_shapes = {"weights": (n_layers, n_qubits)}
# weight_shapes = {"angles": (n_qubits, 3)}
weight_shapes = {"weights_ry": (n_layers, n_qubits), "weights_rz": (n_layers, n_qubits)}


@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Z')
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]


@qml.qnode(dev)
def circuit(inputs, weights_ry, weights_rz):
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    for layer in range(n_layers):
        for i, weight in enumerate(weights_ry[layer]):
            qml.RY(weight, wires=i)
        for i, weight in enumerate(weights_rz[layer]):
            qml.RZ(weight, wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]


class HybridCNNPenny(nn.Module):
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

    def __init__(self):
        super(HybridCNNPenny, self).__init__()

        self.conv0 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=3, padding=1
        )
        self.bn0 = nn.BatchNorm2d(num_features=64)
        self.conv1 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_features=128)
        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(num_features=256)
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(num_features=512)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)

        self.fc0 = nn.Linear(in_features=512 * 2 * 2, out_features=4096)
        self.fc1 = nn.Linear(in_features=4096, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=2)
        self.qnn1 = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.qnn2 = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.fc3 = nn.Linear(in_features=4, out_features=2)
        self.fc4 = nn.Linear(in_features=2, out_features=1)

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = self.max_pool(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.max_pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.max_pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.max_pool(x)
        x = self.dropout(x)
        x = x.view(
            x.size(0), -1
        )  # Aplanar el tensor para la capa completamente conectada
        x = F.relu(self.fc0(x))
        # x = self.dropout(x)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        # x = self.qnn(x)
        # x_1, x_2 = torch.split(x, 2, dim=1)
        # x_1 = self.qnn1(x_1)
        # x_2 = self.qnn2(x_2)
        # x = torch.cat([x_1, x_2], axis=1)
        x = self.qnn1(x)
        # x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))

        return x


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
        super(HybridCNN, self).__init__()

        self.conv0 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=3, padding=1
        )
        self.bn0 = nn.BatchNorm2d(num_features=64)
        self.conv1 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_features=128)
        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(num_features=256)
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(num_features=512)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)

        self.fc0 = nn.Linear(in_features=512 * 2 * 2, out_features=4096)
        self.fc1 = nn.Linear(in_features=4096, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=2)
        self.qnn = TorchConnector(qnn)
        self.fc3 = nn.Linear(1, 1)

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = self.max_pool(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.max_pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.max_pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.max_pool(x)
        x = self.dropout(x)
        x = x.view(
            x.size(0), -1
        )  # Aplanar el tensor para la capa completamente conectada
        x = F.relu(self.fc0(x))
        # x = self.dropout(x)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
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

    feature_map = ZZFeatureMap(2, entanglement='full')
    #ansatz = RealAmplitudes(2, reps=1, entanglement='full')
    
    custom_ansatz_circuit, custom_params = custom_ansatz(2)
    ansatz = custom_ansatz_circuit

    qnn = TwoLayerQNN(
        2,
        feature_map,
        ansatz,
        input_gradients=True,
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
