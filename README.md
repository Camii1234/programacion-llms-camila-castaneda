# Programación con LLMs

Repositorio del ejercicio **Programación con LLMs** del curso **Modelos y Simulación 1** de la **Universidad de Antioquia**.

## 👩‍🎓 Datos de la estudiante

- **Nombre:** María Camila Castañeda Piedrahita
- **Correo:** maria.castanedap@udea.edu.co
- **Curso:** Modelos y Simulación 1
- **Universidad:** Universidad de Antioquia

## 🧭 Descripción del proyecto

Este proyecto busca aprender a usar modelos de lenguaje (LLMs) de forma rigurosa para:

- Diseñar preguntas de programación bien definidas.
- Crear generadores de casos de uso para validar soluciones.
- Implementar funciones solución correctas y verificables.

El trabajo está organizado en dos fases: **Fase 1 (creación)** y **Fase 2 (resolución)**.

## 🛠️ Tecnologías usadas

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

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
  answer-0166.py
  answer-0464.py
  answer-0327.py
  answer-0210.py
validate_answers.py
```

## 1) Fase 1: Preguntas propias y generadores

En la **Fase 1** se diseñaron 4 preguntas propias y 4 scripts para generar casos de uso aleatorios en la carpeta `myquestions/`.

### Archivos de Fase 1

- `myquestions/question-0001.txt`
- `myquestions/question-0001-usecase-generator.py`
- `myquestions/question-0002.txt`
- `myquestions/question-0002-usecase-generator.py`
- `myquestions/question-0003.txt`
- `myquestions/question-0003-usecase-generator.py`
- `myquestions/question-0004.txt`
- `myquestions/question-0004-usecase-generator.py`

### Resumen de preguntas propias

1. **question-0001:** Detectar solapamientos de citas médicas por paciente usando pandas.
2. **question-0002:** Construir una matriz de transición entre estados de navegación usando pandas.
3. **question-0003:** Encontrar el mejor umbral para maximizar F1-score usando scikit-learn.
4. **question-0004:** Seleccionar el mejor K para KMeans usando silhouette score con scikit-learn.

## 2) Fase 2: Soluciones asignadas

En la **Fase 2** se resolvieron 4 preguntas asignadas por compañeros. Cada solución se encuentra en `myanswers/` y conserva su identificador.

### Archivos de Fase 2

- `myanswers/answer-0166.py`
- `myanswers/answer-0464.py`
- `myanswers/answer-0327.py`
- `myanswers/answer-0210.py`

### Funciones implementadas

1. **answer-0166.py**
  - `calcular_curva_roc(y_true, y_scores)`
  - Usa `roc_curve` y `roc_auc_score`.
  - Retorna FPR, TPR y AUC.

2. **answer-0464.py**
  - `resumir_por_grupo_y_normalizar(df, columna_grupo, columna_valor)`
  - Agrupa por columna, calcula suma, media y desviación estándar.
  - Crea `indice_normalizado`.

3. **answer-0327.py**
  - `clasificar_tickets_soporte(textos, etiquetas, nuevo_mensaje)`
  - Usa `TfidfVectorizer` y `MultinomialNB`.
  - Retorna la predicción del nuevo mensaje.

4. **answer-0210.py**
  - `detectar_outliers(df)`
  - Detecta outliers con el método IQR.
  - Retorna solo las filas outliers.

## ✅ Validador opcional

El archivo `validate_answers.py` ejecuta pruebas rápidas para las 4 funciones de Fase 2 con datos de ejemplo.

## ⚙️ Instalación de dependencias (PowerShell)

```powershell
python -m venv .venv
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& .\.venv\Scripts\Activate.ps1)
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn
```

## ▶️ Ejecutar generadores de Fase 1

```powershell
python myquestions/question-0001-usecase-generator.py
python myquestions/question-0002-usecase-generator.py
python myquestions/question-0003-usecase-generator.py
python myquestions/question-0004-usecase-generator.py
```

## ▶️ Verificar soluciones de Fase 2

```powershell
python myanswers/answer-0166.py
python myanswers/answer-0464.py
python myanswers/answer-0327.py
python myanswers/answer-0210.py
```

## ▶️ Ejecutar validador de respuestas

```powershell
python validate_answers.py
```

## 📌 Resultado esperado

Al ejecutar `validate_answers.py` deben aparecer mensajes similares a:

- `answer-0166 -> OK`
- `answer-0464 -> OK`
- `answer-0327 -> OK`
- `answer-0210 -> OK`

## 🏁 Conclusión

Se cumplió el objetivo del ejercicio: crear preguntas propias, construir generadores de casos de uso y resolver preguntas asignadas con funciones verificables. Este repositorio deja una base clara y reproducible para practicar programación asistida con LLMs desde etapas tempranas del curso.
