# External libraries
import torch
from torch.utils.data.dataloader import DataLoader
from torchinfo import summary

# Own libraries
from python.metadata.path import Path
from python.models.utils.tf_cnn import plot_confusion_matrix
from python.models.utils.hybrid_cnn import (
    HybridCNN, HybridCNNPenny, create_qnn, save_q_circuit
)
from python.models.utils.torch_cnn import (
    LoadDataset,
    fit_model,
    image_generator,
    plot_generate,
    predict_data,
)
from python.utils.readers import read_yaml
import pennylane as qml


def hybrid_model():
    """Pipeline de modelo de CNN clásica con hybrid pyTorch."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'PyTorch está utilizando {device}.')

    config = read_yaml(Path.config)['cnn_models']['hybrid_cnn']
    train = config['train']
    test = config['test']

    batch_size = config['batch_size']
    target = tuple(config['input_target'])
    epochs = config['epochs']
    learning_rate = config['learning_rate']
    backend = config['backend']
    structure = config['structure']

    loss_function = torch.nn.BCELoss()

    if train:
        train_transform = image_generator('train')
        test_transform = image_generator('test')

        train_dataset = LoadDataset(Path.train, train_transform)
        val_dataset = LoadDataset(Path.val, test_transform)

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size)

        if structure == 'qiskit':
            qnn = create_qnn(backend)
            print(qnn.operator)
            model = HybridCNN(qnn).to(device)

            save_q_circuit(qnn, Path.q_circuit)

            summary(model, input_size=target)
        else:
            model = HybridCNNPenny().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        hist = fit_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            epochs=epochs,
            structure='hybrid',
        )

        plot_generate(hist=hist, path_save=Path.cnn_hybrid_plot)

    if test:
        test_transform = image_generator('test')
        test_dataset = LoadDataset(csv_file=Path.test, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        if structure == 'qiskit':
            qnn = create_qnn()
            model = HybridCNN(qnn).to(device)
        else:
            model = HybridCNNPenny().to(device)

        model.load_state_dict(torch.load(Path.hybrid_model_torch))

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
                etiquetas, predicciones, Path.confusion_matrix_hybrid
            )

        _, test_acc = predict_data(model, test_loader, loss_function)

        print(f'Accuracy: {test_acc}')