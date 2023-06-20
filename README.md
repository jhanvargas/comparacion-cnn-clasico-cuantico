# Especialización en analítica y ciencia de datos UdeA

## Evaluación del impacto de la computación cuántica en la clasificación de imágenes mediante modelos de CNN

El repositorio tiene como objetivo comparar el desempeño de modelos de redes neuronales convolucionales (CNN) aplicados 
al conjunto de datos CIFAR-10, utilizando tanto computación clásica como cuántica. En el repositorio se incluyen los 
códigos implementados en Python para entrenar y evaluar los modelos en ambas plataformas, así como también los 
resultados obtenidos y una breve documentación explicando los pasos necesarios para replicar el experimento.

Además, se incluirá una sección con una breve introducción a la computación cuántica y su relación con el aprendizaje
automático, así como una revisión bibliográfica de trabajos previos relacionados con la comparación de modelos de 
aprendizaje automático entre computación clásica y cuántica. El objetivo final es proporcionar una comparación 
práctica y replicable de la eficacia y eficiencia de los modelos de CNN en ambas plataformas y ayudar a establecer 
un punto de partida para futuras investigaciones en el área.


## Funcionamiento del repositorio

Dentro del directorio `docs` se encuentra el entregable II de seminario.
El notebbook con la exploración se encuentra en la ruta `python/jupyer_notebooks/`.


## Instalación Qiskit

Para poder utilizar el modelo de CNN cuántico, es necesario instalar la librería Qiskit en tu entorno de desarrollo.
Se recomienda la instalación a través de `pip`, utilizando el siguiente comando:


```python
pip install qiskit
```

También es posible realizar la instalación utilizando `conda`, mediante el siguiente comando:

```python
conda install -c conda-forge qiskit
```
