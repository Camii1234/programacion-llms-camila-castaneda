import numpy as np
from sklearn.metrics import f1_score


def generar_caso_de_uso_mejor_umbral_f1():
    """
    Genera un caso de uso aleatorio (input/output esperado) para la función:

        mejor_umbral_f1(y_true, y_proba, step=0.01)

    Retorna:
        input_data (dict): Diccionario con las claves esperadas por la función solución.
        output_data (dict): Diccionario esperado con best_threshold, best_f1, support_positive.
    """

    rng = np.random.default_rng()  # aleatorio distinto en cada ejecución

    # ------------------------------------------------------------
    # 1) Tamaño del dataset y step
    # ------------------------------------------------------------
    n = int(rng.integers(40, 180))  # 40 a 179 muestras
    possible_steps = np.array([0.01, 0.02, 0.05, 0.1], dtype=float)
    step = float(rng.choice(possible_steps))

    # ------------------------------------------------------------
    # 2) Generar y_true con proporción de positivos aleatoria (evitar extremos triviales)
    # ------------------------------------------------------------
    pos_ratio = float(rng.uniform(0.15, 0.55))  # entre 15% y 55% positivos
    y_true = (rng.random(n) < pos_ratio).astype(int)

    # Asegurar que haya al menos 1 positivo y 1 negativo
    if y_true.sum() == 0:
        y_true[int(rng.integers(0, n))] = 1
    if y_true.sum() == n:
        y_true[int(rng.integers(0, n))] = 0

    # ------------------------------------------------------------
    # 3) Generar y_proba con cierta "separación" (pero con ruido)
    #    - positivos tienden a proba más alta
    #    - negativos tienden a proba más baja
    # ------------------------------------------------------------
    y_proba = np.empty(n, dtype=float)

    # Base para negativos y positivos (en Beta para que sea [0,1])
    neg = rng.beta(2.0, 6.0, size=n)   # sesgada a valores bajos
    pos = rng.beta(6.0, 2.0, size=n)   # sesgada a valores altos

    y_proba[y_true == 0] = neg[y_true == 0]
    y_proba[y_true == 1] = pos[y_true == 1]

    # Ruido pequeño
    y_proba = np.clip(y_proba + rng.normal(0.0, 0.05, size=n), 0.0, 1.0)

    # A veces forzar algunos valores cerca de cortes para generar empates reales
    if rng.random() < 0.35:
        idx = rng.choice(np.arange(n), size=int(rng.integers(3, 8)), replace=False)
        # poner algunos alrededor de 0.5
        y_proba[idx] = np.clip(0.5 + rng.normal(0.0, 0.02, size=len(idx)), 0.0, 1.0)

    # ------------------------------------------------------------
    # 4) Construir INPUT
    # ------------------------------------------------------------
    input_data = {
        "y_true": y_true.copy(),
        "y_proba": y_proba.copy(),
        "step": step,
    }

    # ------------------------------------------------------------
    # 5) Calcular OUTPUT esperado (Ground Truth) según el enunciado:
    #    - probar umbrales 0.0..1.0 inclusive con step
    #    - y_pred = (y_proba >= t).astype(int)
    #    - f1_score(zero_division=0)
    #    - si empate: umbral más pequeño
    # ------------------------------------------------------------
    thresholds = np.arange(0.0, 1.0 + (step / 2), step, dtype=float)  # incluye 1.0 por tolerancia
    best_threshold = float(thresholds[0])
    best_f1 = -1.0

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        score = float(f1_score(y_true, y_pred, zero_division=0))

        if score > best_f1:
            best_f1 = score
            best_threshold = float(t)
        elif score == best_f1 and float(t) < best_threshold:
            best_threshold = float(t)

    output_data = {
        "best_threshold": float(best_threshold),
        "best_f1": float(best_f1),
        "support_positive": int(y_true.sum()),
    }

    return input_data, output_data


if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_mejor_umbral_f1()

    print("=== INPUT ===")
    print("step:", entrada["step"])
    print("n:", len(entrada["y_true"]))
    print("positivos (support_positive):", int(np.sum(entrada["y_true"])))
    print("y_true (primeros 20):", entrada["y_true"][:20])
    print("y_proba (primeros 20):", np.round(entrada["y_proba"][:20], 4))

    print("\n=== OUTPUT ESPERADO ===")
    print(salida_esperada)