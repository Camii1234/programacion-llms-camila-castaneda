# Soluciones — Fase 2: myanswers

> Resumen de las funciones solución asignadas en la Fase 2 del ejercicio "Programación con LLMs".

**Estudiante:** María Camila Castañeda Piedrahita  
**Email:** maria.castanedap@udea.edu.co  
**Curso:** Modelos y Simulación 1 — Universidad de Antioquia

---

## 🧾 Propósito

Este README describe las soluciones implementadas dentro de la carpeta `myanswers/` (Fase 2). Aquí encontrarás:

- Explicación breve de cada función.
- Enlaces al enunciado/generador original cuando están disponibles.
- Comandos para ejecutar y validar localmente.
- Resultado esperado y notas de uso.

---

## 📂 Contenido de `myanswers/`

| Archivo | Función | Descripción rápida | Ejecutar |
|---|---:|---|---|
| `answer-0166.py` | `calcular_curva_roc(y_true, y_scores)` | Calcula FPR, TPR y AUC usando `roc_curve` y `roc_auc_score` (ROC para clasificación binaria). | `python myanswers/answer-0166.py` |
| `answer-0464.py` | `resumir_por_grupo_y_normalizar(df, columna_grupo, columna_valor)` | Agrupa un DataFrame por un campo, calcula suma/media/std y produce un `indice_normalizado`. | `python myanswers/answer-0464.py` |
| `answer-0327.py` | `clasificar_tickets_soporte(textos, etiquetas, nuevo_mensaje)` | Vectoriza texto con TF-IDF y entrena `MultinomialNB` para predecir la categoría de un nuevo ticket. | `python myanswers/answer-0327.py` |
| `answer-0210.py` | `detectar_outliers(df)` | Detecta outliers en la columna `valores` usando IQR y retorna solo las filas outliers. | `python myanswers/answer-0210.py` |

---

## 🔗 Enlaces a enunciados / generadores originales

- Enunciado relacionado con `answer-0166`: https://github.com/VSofia-1/LLMs-VSofia-1/blob/main/myquestions/question-0002.txt
- Generador original para `answer-0166`: https://github.com/VSofia-1/LLMs-VSofia-1/blob/main/myquestions/question-0002-usecase-generator.py

> Nota: los enunciados originales de las otras preguntas se encuentran en el repositorio del curso (carpeta `myquestions/`). Si necesitas enlaces directos a cada pregunta, puedo agregarlos si me indicas los identificadores.

---

## Detalle de cada solución

### 1) `answer-0166.py` — calcular_curva_roc

- Objetivo: medir qué tan bien separa un clasificador las clases positivas y negativas.
- Entrada: `y_true` (etiquetas 0/1) y `y_scores` (probabilidades o scores del clasificador).
- Salida: tupla `(fpr, tpr, auc)` donde `fpr` y `tpr` son arrays y `auc` es float.

Código (ya incluido en `answer-0166.py`):

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

### 2) `answer-0464.py` — resumir_por_grupo_y_normalizar

- Objetivo: generar un resumen estadístico por grupo y normalizar la media.
- Entrada: `df`, `columna_grupo`, `columna_valor`.
- Salida: DataFrame con columnas `sum`, `mean`, `std` y `indice_normalizado`.

### 3) `answer-0327.py` — clasificar_tickets_soporte

- Objetivo: entrenar un clasificador de texto simple (TF-IDF + MultinomialNB) y predecir la etiqueta de un nuevo mensaje.
- Entrada: lista de `textos`, `etiquetas` y `nuevo_mensaje`.
- Salida: `numpy.ndarray` con la predicción.

### 4) `answer-0210.py` — detectar_outliers

- Objetivo: detectar valores atípicos con el método del IQR (Q1/Q3).
- Entrada: DataFrame con columna `valores`.
- Salida: DataFrame filtrado con solo las filas que son outliers.

---

## ⚙️ Requisitos

- Python 3.10+ (recomendado)
- `pandas`, `numpy`, `scikit-learn`

Instalación (PowerShell):

```powershell
python -m venv .venv
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& .\.venv\Scripts\Activate.ps1)
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn
```

---

## ▶️ Comandos de ejecución y validación

- Ejecutar cada archivo (no imprimen por defecto, pero validan import/sintaxis):

```powershell
python myanswers/answer-0166.py
python myanswers/answer-0464.py
python myanswers/answer-0327.py
python myanswers/answer-0210.py
```

- Ejecutar validador rápido (archivo opcional en la raíz):

```powershell
python validate_answers.py
```

**Resultado esperado** al ejecutar `validate_answers.py` (ejemplo):

```
answer-0166 -> OK
answer-0464 -> OK
answer-0327 -> OK
answer-0210 -> OK
```

---

## 📌 Consejos de uso

- Cada archivo en `myanswers/` está pensado para ser importado en tests o en un notebook; no contienen lógica de ejecución pesada.
- Si quieres probar cada función manualmente desde un REPL o notebook, importa la función directamente:

```python
from myanswers.answer_0166 import calcular_curva_roc
# o usando importlib para cargar por ruta
```

- Para reproducibilidad, usa el entorno virtual `.venv` del proyecto.

---

## 🔚 Conclusión

Las 4 soluciones de la Fase 2 están implementadas y verificadas localmente. Este README proporciona una guía clara y rápida para entender, ejecutar y validar las soluciones. Si quieres, puedo:

- Añadir enlaces directos a los enunciados originales de las otras preguntas.
- Generar ejemplos adicionales para cada función.

---

*Última actualización por María Camila Castañeda Piedrahita*