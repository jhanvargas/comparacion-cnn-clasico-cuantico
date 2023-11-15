# External libraries
import torch

from torch.utils.data.dataloader import DataLoader
from torchinfo import summary

# Own libraries
from python.metadata.path import Path
from python.models.utils.tf_cnn import plot_confusion_matrix
from python.models.utils.torch_cnn import (
    LoadDataset,
    TorchCNN,
    image_generator,
    plot_generate,
    show_batch,
    predict_data,
    fit_model,
)
from python.utils.readers import read_yaml


def torch_model() -> None:
    """Pipeline de modelo de CNN clásica con pyTorch.

    Esta función crea y entrena un modelo CNN clásico utilizando PyTorch. 
    La función carga la configuración desde un archivo YAML,
    crea los conjuntos de datos de entrenamiento y validación, 
    y entrena el modelo utilizando el optimizador Adam y la función de pérdida 
    de entropía cruzada binaria. La función también genera una gráfica de la 
    pérdida y precisión de entrenamiento y validación, e imprime la precisión 
    del modelo en el conjunto de datos de prueba.

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'PyTorch está utilizando la {device}.')

    config = read_yaml(Path.config)['cnn_models']['torch_cnn_classic']
    train = config['train']
    test = config['test']

    batch_size = config['batch_size']
    target = tuple(config['input_target'])
    epochs = config['epochs']
    learning_rate = config['learning_rate']

    loss_function = torch.nn.BCELoss()

    if train:
        train_transform = image_generator('train')
        test_transform = image_generator('test')

        train_dataset = LoadDataset(Path.train, train_transform)
        val_dataset = LoadDataset(Path.val, test_transform)

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size)

        # show_batch(train_loader)

        model = TorchCNN().to(device)
        summary(model, input_size=target)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        hist = fit_model(
            model, train_loader, val_loader, loss_function, optimizer, epochs
        )

        plot_generate(hist=hist, path_save=Path.cnn_torch_plot)

    if test:
        test_transform = image_generator('test')
        test_dataset = LoadDataset(csv_file=Path.test, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        model = TorchCNN().to(device)
        model.load_state_dict(torch.load(Path.classic_model_torch))

        model.eval()

        predicciones = []
        etiquetas = []

        with torch.no_grad():
            for data in test_loader:
                inputs, label = data 
                inputs = inputs.to(device)

                outputs = model(inputs)

                predicted = (outputs > 0.5).int() 
                predicted = [pred[0].item() for pred in predicted]

                predicciones.extend(predicted)
                etiquetas.extend(label.numpy()) 

        plot_confusion_matrix(
                etiquetas, predicciones, Path.confusion_matrix_torch
            )

        _, test_acc = predict_data(model, test_loader, loss_function)

        print(f'Accuracy: {test_acc}')
