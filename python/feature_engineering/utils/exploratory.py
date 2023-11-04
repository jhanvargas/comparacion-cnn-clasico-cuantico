# External libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random

from PIL import Image
from skimage import io
import seaborn as sns
import seaborn as sns


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
    title: str,
    num_images: int = 10,
    cols: int = 5,
    rows: int = 2,
    save_path: str = None,
) -> None:
    """Muestra una grilla de imágenes a partir de un directorio.

    Args:
        path: Ruta del directorio que contiene las imágenes.
        title: Título de la grilla.
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

    image_files = image_files[:num_images]

    fig, axes = plt.subplots(
        nrows=rows, ncols=cols, figsize=(cols * 3, rows * 3)
    )

    fig.suptitle(title, fontsize=16)

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

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        label_counts,
        labels=None,
        colors=colors,
        autopct='%.1f%%',
        startangle=90,
    )

    legend_labels = [
        f'{name}: {count}'
        for name, count in zip(label_counts.index, label_counts)
    ]
    ax.legend(
        wedges,
        legend_labels,
        title="Etiquetas",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
    )

    plt.title('Distribución de Etiquetas')
    plt.setp(autotexts, size=10, weight="bold")

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


def analyze_image_statistics(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises

    # Crear un subplot con 1 fila y 2 columnas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Mostrar la imagen en escala de grises
    ax1.imshow(gray_image, cmap='gray')
    ax1.set_title('Imagen en Escala de Grises')
    ax1.axis('off')  # Desactivar los ejes

    # Calcular el histograma para la imagen en escala de grises
    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    
    # Dibujar el histograma
    ax2.plot(histogram, color='black')
    ax2.set_xlim([0, 256])
    ax2.set_title('Histograma de Intensidad de Grises')
    ax2.set_xlabel('Intensidad del Pixel')
    ax2.set_ylabel('Cantidad de Pixels')

    # Mostrar el gráfico
    plt.tight_layout()  # Ajusta automáticamente los subplots para que encajen en la figura
    plt.show()

    # Calcular estadísticas de brillo
    mean_val = np.mean(gray_image)
    std_val = np.std(gray_image)
    print(f"Estadísticas de brillo para {image_path}:")
    print(f"Media: {mean_val:.2f}, Desviación estándar: {std_val:.2f}\n")
