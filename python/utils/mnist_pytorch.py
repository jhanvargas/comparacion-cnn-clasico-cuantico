# External libraries
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def generate_data(
    n_samples: int, batch_size: int, train: bool = True
) -> DataLoader:
    """Generate a filtered MNIST dataset for binary classification (0 or 1).

    This function downloads and prepares the MNIST dataset from torchvision,
     then filters the data to only include samples with labels 0 and 1 for
     binary classification.

    Args:
        n_samples: The number of samples per class (0 and 1) to include in the
         dataset.
        batch_size: The batch size for the DataLoader.
        train: If True, use the training dataset; if False, use the test
         dataset.

    Returns:
        A DataLoader containing the filtered MNIST dataset.

    Example:
        train_loader = generate_data(n_samples=500, batch_size=64, train=True)

    """
    data = datasets.MNIST(
        root="./data",
        train=train,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    # Filter out labels (originally 0-9), leaving only labels 0 and 1
    idx = np.append(
        np.where(data.targets == 0)[0][:n_samples],
        np.where(data.targets == 1)[0][:n_samples],
    )
    data.data = data.data[idx]
    data.targets = data.targets[idx]

    # Define torch dataloader with filtered data
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return data_loader


def plot_data(data_loader: DataLoader, n_samples_show: int) -> None:
    """Plot a subset of images from a data loader.

    This function takes a data loader containing image data and labels,
     extracts a subset of images, and plots them in a row of subplots.
      It also displays the corresponding labels below each image.

    Args:
        data_loader: The DataLoader containing the image data.
        n_samples_show: The number of samples to display in the plot.

    Returns:
        None

    Example:
        plot_data(train_loader, n_samples_show=5)

    """
    data_iter = iter(data_loader)
    fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

    while n_samples_show > 0:
        images, targets = data_iter.__next__()

        axes[n_samples_show - 1].imshow(
            images[0, 0].numpy().squeeze(), cmap="gray"
        )
        axes[n_samples_show - 1].set_xticks([])
        axes[n_samples_show - 1].set_yticks([])
        axes[n_samples_show - 1].set_title(
            "Labeled: {}".format(targets[0].item())
        )

        n_samples_show -= 1

    plt.show()
