# External libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns

from PIL import Image
from skimage import io
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def show_random_image(path: str) -> None:
    """Muestra una imagen aleatoria de una carpeta especificada.

    Args:
        path: La ruta de la carpeta que contiene las imágenes.

    """
    files = os.listdir(path)

    random_image = random.choice(files)

    im = io.imread(os.path.join(path, random_image))
    print(im.shape)

    plt.imshow(im, cmap='gray')
    plt.title(f'Imagen aleatoria: {random_image}\nTamaño: {im.shape}')
    plt.axis('off')
    plt.show()


def plot_images_from_path(
    path: str,
    num_images: int = 10,
    cols: int = 5,
    rows: int = 2,
    save_path: str = None,
) -> None:
    """Muestra una grilla de imágenes a partir de un directorio.

    Args:
        path: Ruta del directorio que contiene las imágenes.
        num_images: Número de imágenes a mostrar. Por defecto 10.
        cols: Número de columnas de la grilla. Por defecto 5.
        rows: Número de filas de la grilla. Por defecto 2.
        save_path: Ruta para guardar la grilla. Por defecto None.

    Raises:
        ValueError: Si cols * rows es menor que num_images.

    """
    if cols * rows < num_images:
        raise ValueError("cols * rows es menor que num_images")

    image_files = [os.path.join(path, file) for file in os.listdir(path)]

    # image_files = image_files[:num_images]
    image_files = np.random.choice(image_files, num_images, replace=False)

    fig, axes = plt.subplots(
        nrows=rows, ncols=cols, figsize=(cols * 3, rows * 3)
    )

    # fig.suptitle(title, fontsize=16)

    axes = axes.flatten()

    for ax, image_file in zip(axes, image_files):
        with Image.open(image_file) as img:
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            # ax.set_title(os.path.basename(image_file))

    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


def plot_label_distribution(
    df: pd.DataFrame, label_column: str, save_path: str = None
) -> None:
    """Grafica la distribución de las etiquetas de una columna en un DataFrame.

    Args:
        df: DataFrame que contiene la columna de etiquetas.
        label_column: Nombre de la columna de etiquetas.
        save_path: Ruta para guardar la gráfica. Por defecto None.

    """
    label_names = {1: 'Retrato', 0: 'Otro'}
    df['Label Name'] = df[label_column].map(label_names)

    label_counts = df['Label Name'].value_counts()

    colors = sns.color_palette('pastel')[0 : len(label_counts)]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        label_counts,
        labels=None,
        colors=colors,
        autopct='%.1f%%',
        startangle=90,
    )

    legend_labels = [
        f'{name}: {count:,}'
        for name, count in zip(label_counts.index, label_counts)
    ]
    ax.legend(
        wedges,
        legend_labels,
        title="Etiquetas",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=14,
    )

    plt.title('Distribución de Etiquetas', fontsize=14)
    plt.setp(autotexts, size=14, weight="bold")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


def analyze_image_statistics_(image_paths: list) -> None:
    """Calcula el histograma de color y las estadísticas de brillo para una
     lista de imágenes.

    Args:
        image_paths: Lista de rutas de las imágenes.

    """
    for path in image_paths:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        colors = ('r', 'g', 'b')
        for i, color in enumerate(colors):
            histogram = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(histogram, color=color)
            plt.xlim([0, 256])

        plt.title('Histograma de Color')
        plt.xlabel('Intensidad del Pixel')
        plt.ylabel('Cantidad de Pixels')

        plt.show()

        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mean_val = np.mean(gray_image)
        std_val = np.std(gray_image)
        print(f"Estadísticas de brillo para {path}:")
        print(f"Media: {mean_val:.2f}, Desviación estándar: {std_val:.2f}\n")


def analyze_image_statistics(image_path: str, save_path: str = None) -> None:
    """Analiza las estadísticas de brillo de una imagen en escala de grises y muestra
     su histograma.

    Args:
        image_path: Ruta de la imagen a analizar.
        save_path: Ruta para guardar la gráfica. Por defecto None.

    """
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(gray_image, cmap='gray')
    ax1.set_title('Imagen en Escala de Grises')
    ax1.axis('off')

    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    ax2.plot(histogram, color='black')
    ax2.set_xlim([0, 256])
    ax2.set_title('Histograma de Intensidad de Grises')
    ax2.set_xlabel('Intensidad del Pixel')
    ax2.set_ylabel('Cantidad de Pixels')

    mean_val = np.mean(gray_image)
    std_val = np.std(gray_image)
    ax2.text(
        0.1,
        0.9,
        f"Media: {mean_val:.2f}\nDesviación estándar: {std_val:.2f}",
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment='top',
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()

    print(f"Estadísticas de brillo para {image_path}:")
    print(f"Media: {mean_val:.2f}, Desviación estándar: {std_val:.2f}\n")


def check_image_quality(file_paths: list) -> None:
    """Verifica la calidad de las imágenes en una lista de rutas de archivo.

    Args:
        file_paths : Una lista de rutas de archivo.

    Returns:
        Una lista de problemas encontrados en las imágenes. Si no se encontraron
         problemas, la lista estará vacía.

    """
    problems = []

    for path in file_paths:
        if not os.path.isfile(path):
            problems.append(f"Archivo no encontrado: {path}")
            continue

        try:
            with Image.open(path) as img:
                if img.size != (128, 128):
                    problems.append(
                        f"Dimensiones inesperadas en {path}: {img.size}"
                    )

                pixel_values = np.array(img)
                if pixel_values.min() < 0 or pixel_values.max() > 255:
                    problems.append(
                        f"Valores de píxeles fuera de rango en {path}"
                    )

        except (IOError, SyntaxError) as e:
            problems.append(f"No se pudo leer la imagen {path}: {e}")

    return problems


def images_to_pixel_vectors(file_paths: list) -> np.array:
    """Convierte una lista de imágenes en una matriz de vectores de píxeles.

    Args:
        file_paths (list): Una lista de rutas de archivo que contienen imágenes.

    Returns:
        numpy.ndarray: Una matriz de vectores de píxeles de las imágenes.

    """
    pixel_vectors = []
    for path in file_paths:
        with Image.open(path) as img:
            pixel_vector = np.array(img).flatten()
            pixel_vectors.append(pixel_vector)
    return np.array(pixel_vectors)


def apply_pca(data: pd.DataFrame, save_path: str = None) -> None:
    """Realiza PCA en un conjunto de datos y genera un subplot con el scree
    plot y la visualización de las dos primeras componentes principales.

    Args:
        data: Un DataFrame que contiene una columnas de pixeles.
        vectores de píxeles aplanados y una columna con etiquetas
        binarias.
        save_path: Ruta para guardar la gráfica. Por defecto None.

    """
    pixel_vectors = data.drop('label', axis=1)
    labels = data['label'].values

    scaler = StandardScaler()
    pixel_vectors_scaled = scaler.fit_transform(pixel_vectors)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(pixel_vectors_scaled)

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        principal_components[:, 0],
        principal_components[:, 1],
        c=labels,
        alpha=0.5,
    )

    legend1 = ax.legend(*scatter.legend_elements(), title="Clases")
    ax.add_artist(legend1)
    ax.set_xlabel('Componente principal 1')
    ax.set_ylabel('Componente principal 2')
    ax.set_title('PCA - Primeras dos componentes principales')
    ax.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


def apply_tsne(data: pd.DataFrame, save_path: str = None) -> None:
    """Realiza PCA en un conjunto de datos y genera un subplot con el scree
    plot y la visualización de las dos primeras componentes principales.

    Args:
        data: Un DataFrame que contiene una columnas de pixeles.
        vectores de píxeles aplanados y una columna con etiquetas
        binarias.
        save_path: Ruta para guardar la gráfica. Por defecto None.

    """
    pixel_vectors = data.drop('label', axis=1)
    labels = data['label'].values

    scaler = StandardScaler()
    pixel_vectors_scaled = scaler.fit_transform(pixel_vectors)

    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(pixel_vectors_scaled)

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        tsne_results[:, 0],
        tsne_results[:, 1],
        c=labels,
        alpha=0.5,
        cmap='viridis',
    )

    legend1 = ax.legend(*scatter.legend_elements(), title="Clases")
    ax.add_artist(legend1)
    ax.set_xlabel('t-SNE característica 1')
    ax.set_ylabel('t-SNE característica 2')
    ax.set_title('Visualización con t-SNE')
    ax.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()
