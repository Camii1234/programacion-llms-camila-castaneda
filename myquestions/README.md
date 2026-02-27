# Programación con LLMs — Entrega 1 (Fase 1) | Modelos y Simulación 1 (UdeA)

**María Camila Castañeda Piedrahita**  
📧 maria.castanedap@udea.edu.co

---

## 🎯 Objetivo de esta entrega

Este repositorio cumple con la **Entrega 1 (Fase 1)** del ejercicio *Programación con LLMs*.  
El objetivo es proponer **4 preguntas** (funciones a implementar) y, para cada una, construir un **generador de casos de uso aleatorios** que produzca:

- **Input:** diccionario con los argumentos de la función solución.
- **Output esperado (Ground Truth):** el resultado que debería devolver la función solución para ese input.

Con esto se puede **automatizar la validación** de soluciones de forma reproducible y rigurosa.

---

## ✅ Requisitos / Dependencias

Los generadores usan:

- **Python 3.10+** (recomendado)
- `pandas`
- `numpy`
- `scikit-learn`

---

## 🧪 Instalación (Windows PowerShell)

Desde la raíz del repo:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install pandas numpy scikit-learn
```

> Si estás en macOS/Linux, cambia la activación por: `source .venv/bin/activate`

---

## 📁 Estructura del repositorio

```text
README.md
myquestions/
  question-0001.txt
  question-0001-usecase-generator.py
  question-0002.txt
  question-0002-usecase-generator.py
  question-0003.txt
  question-0003-usecase-generator.py
  question-0004.txt
  question-0004-usecase-generator.py
myanswers/
```

- En **`myquestions/`** están los enunciados y los generadores.
- La carpeta **`myanswers/`** se usa en la Entrega 2 (Fase 2), por eso aquí puede estar vacía.

---

## 📝 Preguntas incluidas (resumen)

### `question-0001` (Pandas)
**Función:** `detectar_solapamientos(df, paciente_col, fecha_col, hora_inicio_col, duracion_min_col)`  
Detecta citas médicas que se **traslapan** por paciente, calculando `inicio_dt`, `fin_dt` y un booleano `solapada`.

### `question-0002` (Pandas)
**Función:** `matriz_transicion(df, user_col, time_col, state_col)`  
Construye una **matriz de transición (Markov)** entre estados/pantallas a partir de secuencias de eventos por usuario.

### `question-0003` (Sklearn)
**Función:** `mejor_umbral_f1(y_true, y_proba, step=0.01)`  
Encuentra el **umbral** que maximiza el **F1-score** evaluando múltiples thresholds (con regla de desempate).

### `question-0004` (Sklearn)
**Función:** `mejor_k_kmeans(X, k_values, random_state=42)`  
Selecciona el mejor **K** para KMeans usando **silhouette_score** sobre datos estandarizados.

---

## ▶️ Cómo ejecutar los generadores

> Asegúrate de estar en el entorno virtual: `(.venv)` en la consola.

### Ejecutar uno por uno

```powershell
python myquestions/question-0001-usecase-generator.py
python myquestions/question-0002-usecase-generator.py
python myquestions/question-0003-usecase-generator.py
python myquestions/question-0004-usecase-generator.py
```

### Ejecutar los 4 en un solo comando (PowerShell)

```powershell
python myquestions/question-0001-usecase-generator.py; `
python myquestions/question-0002-usecase-generator.py; `
python myquestions/question-0003-usecase-generator.py; `
python myquestions/question-0004-usecase-generator.py
```

---

## ✅ Resultado esperado al ejecutar

Cada generador imprime:

- **INPUT:** una muestra del diccionario de entrada (data generado aleatoriamente).
- **OUTPUT ESPERADO:** el resultado que debería producir la función solución.
- En algunos casos incluye verificaciones rápidas, por ejemplo:
  - cantidad de solapamientos detectados (pregunta 0001)
  - suma por fila ≈ 1.0 en la matriz de transición (pregunta 0002)
  - `best_threshold` y `best_f1` (pregunta 0003)
  - `best_k` y scores por cada k (pregunta 0004)

Esto confirma que los generadores:
- ejecutan sin errores,
- producen casos no triviales,
- y generan el ground truth de forma consistente.

---

## 🧾 Conclusión

Con esta entrega se logró:

- Proponer **4 ejercicios** (2 basados en pandas y 2 en sklearn).
- Implementar **4 generadores aleatorios** que producen inputs y outputs esperados.
- Dejar lista la base para que en la **Entrega 2** se puedan validar soluciones automáticamente
  usando estos casos de uso.

---
