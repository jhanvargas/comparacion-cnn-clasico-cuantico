# Especialización en analítica y ciencia de datos UdeA

## Evaluación del impacto de la computación cuántica en la clasificación de imágenes mediante modelos de CNN

Este proyecto explora la integración de la computación cuántica en la clasificación de imágenes mediante Redes Neuronales Convolucionales (CNN). Se comparan modelos tradicionales de CNN desarrollados con TensorFlow y PyTorch frente a un modelo híbrido que incorpora un circuito cuántico, utilizando un conjunto de datos de retratos para la clasificación binaria. Los resultados muestran que, aunque los modelos clásicos presentan un rendimiento ligeramente superior, el modelo híbrido cuántico muestra un potencial prometedor. El estudio subraya la viabilidad de la computación cuántica en el aprendizaje profundo y abre caminos para futuras investigaciones en este campo emergente

## Funcionamiento del repositorio

El repositorio contiene varios componentes esenciales para la ejecución y evaluación de modelos de redes neuronales, tanto convencionales como cuánticos:

`api`: Incluye herramientas de API, presumiblemente para la interacción con modelos entrenados o servicios externos.
`output`: Almacena los resultados de los modelos, especialmente útil para el modelo híbrido.
`python`: Contiene scripts de Python para diferentes propósitos, incluyendo ejemplos de uso, ingeniería de características, y ejemplos específicos de IBM Quantum.
`.gitignore`: Especifica qué archivos y carpetas deben ignorarse en las subidas de git.
`README.md`: Ofrece una explicación detallada del proyecto y sus componentes.
`config.py` y config.yaml: Contienen configuraciones, posiblemente para modelos o la API.
`main.py`: El script principal para ejecutar el API del proyecto.
`requirements.txt`: Lista las dependencias necesarias para el proyecto.
`run.py`: Script para ejecutar ciertas tareas o modelos.
Dentro del directorio python, hay más carpetas para organización modular, como jupyter_notebooks para exploración y experimentación, models para almacenar los modelos entrenados, y utils para funciones auxiliares.


## Instalación Qiskit

Para poder utilizar el modelo de CNN cuántico, es necesario instalar la librería Qiskit en tu entorno de desarrollo.
Se recomienda la instalación a través de `pip`, utilizando el siguiente comando:


```python
pip install qiskit
```

## Instalación PennyLane

Para poder utilizar el modelo de CNN cuántico, es necesario instalar la librería Qiskit en tu entorno de desarrollo.
Se recomienda la instalación a través de `pip`, utilizando el siguiente comando:


```python
pip install pennylane
```

![Alt text](/output/api.png)