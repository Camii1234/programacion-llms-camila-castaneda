# Soluciones — Fase 2: myanswers

Repositorio de las soluciones asignadas en la Fase 2 del ejercicio **Programación con LLMs** del curso **Modelos y Simulación 1** de la **Universidad de Antioquia**.

## 👩‍🎓 Datos de la estudiante

| Campo | Valor |
|---|---|
| Nombre | María Camila Castañeda Piedrahita |
| Correo | maria.castanedap@udea.edu.co |
| Curso | Modelos y Simulación 1 |
| Universidad | Universidad de Antioquia |

## 🧾 Propósito

Esta carpeta contiene las funciones solución asignadas en la Fase 2. Cada archivo `answer-XXXX.py` responde a una pregunta creada por otro compañero y fue resuelto usando librerías estándar de análisis de datos y aprendizaje automático.

## 🛠️ Tecnologías usadas

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white)

## 📂 Contenido de `myanswers/`

| ID | Archivo | Función | Tema | Librerías principales |
|---|---|---|---|---|
| 0166 | `answer-0166.py` | `calcular_curva_roc(y_true, y_scores)` | Curva ROC y AUC | NumPy, scikit-learn |
| 0464 | `answer-0464.py` | `resumir_por_grupo_y_normalizar(df, columna_grupo, columna_valor)` | Resumen estadístico por grupo | Pandas, NumPy |
| 0327 | `answer-0327.py` | `clasificar_tickets_soporte(textos, etiquetas, nuevo_mensaje)` | Clasificación de texto con TF-IDF y Naive Bayes | NumPy, scikit-learn |
| 0210 | `answer-0210.py` | `detectar_outliers(df)` | Detección de outliers con IQR | Pandas, NumPy |

## 🔗 Enlaces originales

| ID | Archivo | Enunciado original | Generador original | Repositorio |
|---|---|---|---|---|
| 0166 | `answer-0166.py` | [Ver enunciado](https://github.com/VSofia-1/LLMs-VSofia-1/blob/main/myquestions/question-0002.txt) | [Ver generador](https://github.com/VSofia-1/LLMs-VSofia-1/blob/main/myquestions/question-0002-usecase-generator.py) | [Repositorio](https://github.com/VSofia-1/LLMs-VSofia-1) |
| 0464 | `answer-0464.py` | [Ver enunciado](https://github.com/isaacmgz/AI4ENG-GenAI-ISAAC/blob/main/myquestions/question-0004.txt) | [Ver generador](https://github.com/isaacmgz/AI4ENG-GenAI-ISAAC/blob/main/myquestions/question-0004-usecase-generator.py) | [Repositorio](https://github.com/isaacmgz/AI4ENG-GenAI-ISAAC) |
| 0327 | `answer-0327.py` | [Ver enunciado](https://github.com/alejandrohenaoa/Alejandro/blob/main/myquestions/question-0003.txt) | [Ver generador](https://github.com/alejandrohenaoa/Alejandro/blob/main/myquestions/question-0003-usecase-generator.py) | [Repositorio](https://github.com/alejandrohenaoa/Alejandro) |
| 0210 | `answer-0210.py` | [Ver enunciado](https://github.com/Valentina-Garro/lab-programacion-LLMs/blob/main/myquestions/question-0002.txt) | [Ver generador](https://github.com/Valentina-Garro/lab-programacion-LLMs/blob/main/myquestions/question-0002-usecase-generator.py) | [Repositorio](https://github.com/Valentina-Garro/lab-programacion-LLMs) |

## 1) `answer-0166.py` — Curva ROC y AUC

**Enunciado original:** [question-0002.txt](https://github.com/VSofia-1/LLMs-VSofia-1/blob/main/myquestions/question-0002.txt)  
**Generador original:** [question-0002-usecase-generator.py](https://github.com/VSofia-1/LLMs-VSofia-1/blob/main/myquestions/question-0002-usecase-generator.py)  
**Repositorio original:** [VSofia-1/LLMs-VSofia-1](https://github.com/VSofia-1/LLMs-VSofia-1)

### Objetivo

Medir qué tan bien separa un modelo de clasificación las clases usando la curva ROC y el área bajo la curva (AUC).

### Entradas

- `y_true`: etiquetas reales binarios.
- `y_scores`: puntuaciones o probabilidades predichas por el modelo.

### Salida

Una tupla `(fpr, tpr, auc)` donde:

- `fpr` es la tasa de falsos positivos.
- `tpr` es la tasa de verdaderos positivos.
- `auc` es el área bajo la curva ROC.

### Lógica

La función convierte las entradas a arreglos de NumPy, calcula la curva ROC con `roc_curve` y luego obtiene el AUC con `roc_auc_score`. Finalmente retorna los tres valores para que puedan analizarse o graficarse después.

### Código completo

```python
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


def calcular_curva_roc(y_true, y_scores):
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = float(roc_auc_score(y_true, y_scores))

    return fpr, tpr, auc
```

## 2) `answer-0464.py` — Resumen por grupo y normalización

**Enunciado original:** [question-0004.txt](https://github.com/isaacmgz/AI4ENG-GenAI-ISAAC/blob/main/myquestions/question-0004.txt)  
**Generador original:** [question-0004-usecase-generator.py](https://github.com/isaacmgz/AI4ENG-GenAI-ISAAC/blob/main/myquestions/question-0004-usecase-generator.py)  
**Repositorio original:** [isaacmgz/AI4ENG-GenAI-ISAAC](https://github.com/isaacmgz/AI4ENG-GenAI-ISAAC)

### Objetivo

Construir una tabla resumen por grupo, calculando estadísticas básicas de una variable numérica y luego normalizar una de las métricas para facilitar la comparación entre grupos.

### Entradas

- `df`: DataFrame con los datos.
- `columna_grupo`: nombre de la columna que define el grupo.
- `columna_valor`: nombre de la columna numérica a resumir.

### Salida

Un DataFrame con una fila por grupo y las columnas:

- `sum`
- `mean`
- `std`
- `indice_normalizado`

### Lógica

Primero se agrupa el DataFrame por `columna_grupo` y se aplican agregaciones sobre `columna_valor`: suma, media y desviación estándar. Después se crea `indice_normalizado` dividiendo cada media entre la media máxima, de forma que el grupo con mayor promedio quede en 1.

### Código completo

```python
import pandas as pd
import numpy as np


def resumir_por_grupo_y_normalizar(df, columna_grupo, columna_valor):
    resumen = (
        df.groupby(columna_grupo)[columna_valor]
        .agg(["sum", "mean", "std"])
        .reset_index()
    )

    resumen["indice_normalizado"] = resumen["mean"] / resumen["mean"].max()

    return resumen
```

## 3) `answer-0327.py` — Clasificación automática de tickets de soporte

**Enunciado original:** [question-0003.txt](https://github.com/alejandrohenaoa/Alejandro/blob/main/myquestions/question-0003.txt)  
**Generador original:** [question-0003-usecase-generator.py](https://github.com/alejandrohenaoa/Alejandro/blob/main/myquestions/question-0003-usecase-generator.py)  
**Repositorio original:** [alejandrohenaoa/Alejandro](https://github.com/alejandrohenaoa/Alejandro)

### Objetivo

Clasificar automáticamente un nuevo ticket de soporte a partir de ejemplos de texto ya etiquetados.

### Entradas

- `textos`: lista de textos de entrenamiento.
- `etiquetas`: etiquetas asociadas a cada texto.
- `nuevo_mensaje`: texto nuevo que se desea clasificar.

### Salida

Un `numpy.ndarray` con la predicción de clase para el nuevo mensaje.

### Lógica

La función transforma los textos con `TfidfVectorizer`, que convierte el lenguaje en una matriz numérica. Luego entrena un modelo `MultinomialNB` con las etiquetas disponibles y usa ese modelo para predecir la clase del nuevo mensaje.

### Código completo

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def clasificar_tickets_soporte(textos, etiquetas, nuevo_mensaje):
    vectorizador = TfidfVectorizer()
    X = vectorizador.fit_transform(textos)

    modelo = MultinomialNB()
    modelo.fit(X, etiquetas)

    X_nuevo = vectorizador.transform([nuevo_mensaje])
    prediccion = modelo.predict(X_nuevo)

    return prediccion
```

## 4) `answer-0210.py` — Detección de outliers con IQR

**Enunciado original:** [question-0002.txt](https://github.com/Valentina-Garro/lab-programacion-LLMs/blob/main/myquestions/question-0002.txt)  
**Generador original:** [question-0002-usecase-generator.py](https://github.com/Valentina-Garro/lab-programacion-LLMs/blob/main/myquestions/question-0002-usecase-generator.py)  
**Repositorio original:** [Valentina-Garro/lab-programacion-LLMs](https://github.com/Valentina-Garro/lab-programacion-LLMs)

### Objetivo

Detectar valores atípicos en un DataFrame usando el método del rango intercuartílico (IQR).

### Entradas

- `df`: DataFrame que contiene la columna `valores`.

### Salida

Un DataFrame filtrado con únicamente las filas consideradas outliers.

### Lógica

Se calculan los cuartiles `Q1` y `Q3` de la columna `valores`, luego el `IQR = Q3 - Q1`. Con esos valores se obtienen los límites inferior y superior. Toda fila con un valor menor al límite inferior o mayor al límite superior se devuelve como outlier.

### Código completo

```python
import pandas as pd
import numpy as np


def detectar_outliers(df):
    q1 = df["valores"].quantile(0.25)
    q3 = df["valores"].quantile(0.75)
    iqr = q3 - q1

    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    outliers = df[
        (df["valores"] < limite_inferior) |
        (df["valores"] > limite_superior)
    ]

    return outliers
```

## 📋 Requisitos

- Python 3.10 o superior recomendado
- pandas
- numpy
- scikit-learn

## ⚙️ Instalación

```powershell
python -m venv .venv
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& .\.venv\Scripts\Activate.ps1)
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn
```

## ▶️ Ejecución de soluciones

```powershell
python myanswers/answer-0166.py
python myanswers/answer-0464.py
python myanswers/answer-0327.py
python myanswers/answer-0210.py
```

Si no imprimen nada, está bien: estos archivos solo contienen funciones y sirven para ser importados o validados desde pruebas.

## ✅ Validación rápida

```powershell
python validate_answers.py
```

### Resultado esperado

```text
answer-0166 -> OK
answer-0464 -> OK
answer-0327 -> OK
answer-0210 -> OK
```

## 🏁 Conclusión

En esta carpeta quedaron implementadas las 4 soluciones asignadas en la Fase 2, cada una asociada a su pregunta original y a su generador correspondiente. Las funciones pueden ejecutarse con entradas como las que producen los generadores de los compañeros y se verificaron localmente con `validate_answers.py`, cumpliendo el objetivo de crear soluciones claras, reutilizables y comprobables.

*Última actualización por María Camila Castañeda Piedrahita*