# External libraries
import torch

from torch.utils.data.dataloader import DataLoader
from torchinfo import summary
from tqdm import tqdm

# Own libraries
from python.metadata.path import Path
from python.models.utils.torch_cnn import (
    LoadDataset,
    image_generator,
    plot_generate,
    show_batch,
    predict_data,
    fit_model,
)
from python.models.utils.hybrid_cnn import HybridCNN, create_qnn, save_q_circuit
from python.utils.readers import read_yaml
from python.ibm_quantum.utils.connect import get_ibm_quantum

from qiskit.utils import QuantumInstance


def hybrid_model():
    """Pipeline de modelo de CNN clásica con hybrid pyTorch."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'PyTorch está utilizando la {device}.')

    config = read_yaml(Path.config)['cnn_models']['hybrid_cnn']
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

        qnn = create_qnn()
        print(qnn.operator)
        model = HybridCNN(qnn).to(device)

        save_q_circuit(qnn, Path.q_circuit)

        summary(model, input_size=target)

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

        qnn = create_qnn()
        model = HybridCNN(qnn).to(device)
        model.load_state_dict(torch.load(Path.hybrid_model_torch))

        test_loss, test_acc = predict_data(model, test_loader, loss_function)

        print(f'Accuracy: {test_acc}')
