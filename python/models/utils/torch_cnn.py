# External libraries
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from typing import Tuple

# Own libraries
from python.metadata.path import Path


class LoadDataset(Dataset):
    """Clase para cargar un conjunto de datos desde un archivo CSV.

    Args:
        csv_file: Ruta al archivo CSV que contiene los datos.
        transform: Transformación opcional a aplicar a las imágenes.

    Attributes:
        data_frame (pandas.DataFrame): El DataFrame que contiene los datos del
            archivo CSV.
        transform (callable): Función de transformación aplicada a las imágenes.

    Methods:
        __len__(): Retorna la cantidad de muestras en el conjunto de datos.
        __getitem__(idx): Retorna una muestra específica del conjunto de datos.

    Examples:
        dataset = LoadDataset('datos.csv', transform=transforms)

    """

    def __init__(self, csv_file: str, transform: callable = None) -> None:
        """Inicializa una instancia de la clase LoadDataset.

        Args:
            csv_file: Ruta al archivo CSV que contiene los datos.
            transform: Transformación opcional a aplicar a las imágenes.

        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self) -> int:
        """Retorna la cantidad de muestras en el conjunto de datos.

        Returns:
            int: Cantidad de muestras en el conjunto de datos.

        """
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> tuple:
        """Retorna una muestra específica del conjunto de datos.

        Args:
            idx: Índice de la muestra a recuperar.

        Returns:
            Tupla que contiene la imagen y su etiqueta correspondiente.

        """
        img_name = os.path.join(self.data_frame.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.data_frame.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label


class TorchCNN(nn.Module):
    """Clase que define una red neuronal convolucional en PyTorch.

    Atributos:
        conv0 (nn.Conv2d): Capa de convolución 2D con 16 filtros de 3x3 
         y padding de 1.
        bn0 (nn.BatchNorm2d): Capa de normalización por lotes con 16 canales.
        conv1 (nn.Conv2d): Capa de convolución 2D con 32 filtros de 3x3 
         y padding de 1.
        bn1 (nn.BatchNorm2d): Capa de normalización por lotes con 32 canales.
        conv2 (nn.Conv2d): Capa de convolución 2D con 64 filtros de 3x3 
         y padding de 1.
        bn2 (nn.BatchNorm2d): Capa de normalización por lotes con 64 canales.
        conv3 (nn.Conv2d): Capa de convolución 2D con 128 filtros de 3x3  
         y padding de 1.
        bn3 (nn.BatchNorm2d): Capa de normalización por lotes con 128 canales.
        max_pool (nn.MaxPool2d): Capa de pooling máximo 2D con kernel de 2x2 
         y stride de 2.
        dropout (nn.Dropout): Capa de dropout con probabilidad de 0.5.
        fc0 (nn.Linear): Capa de red neuronal completamente conectada con 
         128*2*2 entradas y 1024 salidas.
        fc1 (nn.Linear): Capa de red neuronal completamente conectada con 
         1024 entradas y 32 salidas.
        fc2 (nn.Linear): Capa de red neuronal completamente conectada con 
         32 entradas y 1 salida.

    Métodos:
        forward(x): Realiza una pasada hacia adelante por la red neuronal.

    """
    def __init__(self):
        super(TorchCNN, self).__init__()

        self.conv0 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, padding=1
        )
        self.bn0 = nn.BatchNorm2d(num_features=16)
        self.conv1 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.bn3 = nn.BatchNorm2d(num_features=128)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)

        self.fc0 = nn.Linear(in_features=128 * 2 * 2, out_features=1024)
        self.fc1 = nn.Linear(in_features=1024, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=1)

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
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc0(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.sigmoid(self.fc2(x))

        return x


def image_generator(data: str) -> transforms.Compose:
    """Crea y devuelve un transformador de datos de imágenes configurado según
        el modo especificado.

    Args:
        data: El modo de los datos, 'train' para datos de entrenamiento o
            'test' para datos de prueba.

    Returns:
        Transformador de datos de imágenes configurado según el
            modo especificado.

    Raises:
        NotImplementedError: Si el valor de 'data' no es 'train' ni 'test'.

    """
    if data == 'train':
        transform = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()]
        )
    elif data == 'test':
        transform = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()]
        )
    else:
        raise NotImplementedError('Solo se acepta train o test')

    return transform


def show_batch(data_loader: DataLoader) -> None:
    """Muestra un lote de imágenes y etiquetas en una cuadrícula utilizando
        Matplotlib.

    Args:
        data_loader: DataLoader que contiene los lotes de imágenes y etiquetas.

    """
    for images, labels in data_loader:
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break
    plt.show()


def predict_data(
    model: nn.Module, test_loader: DataLoader, loss_function: callable
) -> Tuple[float, float]:
    """Realiza predicciones en el conjunto de datos de prueba y calcula la
        pérdida y precisión.

    Args:
        model: El modelo de red neuronal utilizado para las predicciones.
        test_loader: DataLoader que contiene los datos de prueba.
        loss_function: Función de pérdida utilizada para calcular la pérdida.

    Returns:
        Un par de valores que representan la pérdida y la precisión en el
            conjunto de prueba.

    """
    num_correct = 0
    num_samples = 0

    model.eval()

    with torch.no_grad():
        for batch, (x_test, y_test) in enumerate(test_loader):
            x_test = x_test.float()
            y_test = y_test.float()

            y_test = y_test.view(-1, 1)

            test_predict = model(x_test)
            test_loss = loss_function(test_predict, y_test)

            rounded_test_predict = torch.round(test_predict)
            num_correct += torch.sum(rounded_test_predict == y_test)
            num_samples += len(y_test)

    model.train()

    test_acc = num_correct / num_samples

    return test_loss, test_acc


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_function: callable,
    optimizer: callable,
    epochs: int = 10,
    structure: str = 'classic',
    patience: int = 10,
) -> dict:
    """Entrena un modelo de red neuronal de PyTorch.

    Args:
        model: Modelo de red neuronal a entrenar.
        train_loader: DataLoader de entrenamiento.
        val_loader: DataLoader de validación.
        loss_function: Función de pérdida para el entrenamiento.
        optimizer: Optimizador para el entrenamiento.
        epochs: Número de épocas de entrenamiento. Por defecto, 10.
        structure:

    Returns:
        dict: Un diccionario que contiene históricos de pérdida y precisión.

    """
    train_losses, test_losses, train_accuracy, test_accuracy = [], [], [], []

    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(epochs):
        num_correct_train = 0
        num_samples_train = 0
        for batch, (x_train, y_train) in tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):
            x_train = x_train.float()
            y_train = y_train.float()

            y_train = y_train.view(-1, 1)

            # Forward propagation
            train_predict = model(x_train)
            train_loss = loss_function(train_predict, y_train)

            # Calculate train accuracy
            with torch.no_grad():
                rounded_train_predict = torch.round(train_predict)
                num_correct_train += torch.sum(rounded_train_predict == y_train)
                num_samples_train += len(y_train)

            # Backward propagation
            optimizer.zero_grad()
            train_loss.backward()

            # Gradient descent
            optimizer.step()

        train_acc = num_correct_train / num_samples_train
        test_loss, test_acc = predict_data(model, val_loader, loss_function)

        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        train_accuracy.append(train_acc.item())
        test_accuracy.append(test_acc.item())

        if test_loss < best_val_loss:
            best_val_loss = test_loss
            early_stopping_counter = 0
            # Aquí también podrías guardar el modelo si es el mejor hasta ahora
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f'Parada temprana después de {epoch + 1} épocas.')
                break

        print(
            f'Epoch {epoch + 1} '
            f'| Train Loss: {train_loss.item():.4f} '
            f'| Test Loss: {test_loss.item():.4f} '
            f'| Train Acc: {train_acc.item() * 100:.2f}% '
            f'| Test Acc: {test_acc.item() * 100:.2f}%'
        )

    hist = {
        'accuracy': train_accuracy,
        'val_accuracy': test_accuracy,
        'loss': train_losses,
        'val_loss': test_losses,
    }

    if structure == 'classic':
        torch.save(model.state_dict(), Path.classic_model_torch)
    elif structure == 'hybrid':
        torch.save(model.state_dict(), Path.hybrid_model_torch)
    else:
        raise NotImplementedError

    return hist


def plot_generate(hist: dict, path_save: str = None) -> None:
    """Visualiza las curvas de precisión y pérdida del historial de
        entrenamiento y validación.

    Args:
        hist: Diccionario que contiene el historial de entrenamiento.
        path_save: Ruta para guardar la imagen.

    """
    # Datos del historial de entrenamiento
    train_accuracy = hist['accuracy']
    val_accuracy = hist['val_accuracy']
    train_loss = hist['loss']
    val_loss = hist['val_loss']

    # Crear una figura con un subplot de 1x2
    plt.figure(figsize=(12, 5))

    # Subplot para la precisión (accuracy)
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracy, label='Train Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Subplot para la pérdida (loss)
    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()

    if path_save:
        plt.savefig(path_save)

    plt.show()
