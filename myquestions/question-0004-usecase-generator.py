import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def generar_caso_de_uso_mejor_k_kmeans():
    """
    Genera un caso de uso aleatorio (input/output esperado) para la función:

        mejor_k_kmeans(X, k_values, random_state=42)

    Retorna:
        input_data (dict): Diccionario con las claves esperadas por la función solución.
        output_data (dict): Diccionario esperado con best_k, best_score y scores.
    """

    rng = np.random.default_rng()  # aleatorio distinto en cada ejecución

    # ------------------------------------------------------------
    # 1) Elegir parámetros del dataset
    # ------------------------------------------------------------
    n_features = int(rng.integers(2, 6))   # 2 a 5 features
    true_k = int(rng.integers(2, 6))       # clusters "reales" 2 a 5

    # Tamaño por cluster (evitar clusters muy pequeños)
    sizes = rng.integers(30, 90, size=true_k)
    n_samples = int(np.sum(sizes))

    # ------------------------------------------------------------
    # 2) Generar clusters sintéticos (gaussianos) en espacio original
    #    para que silhouette sea calculable y no sea trivial.
    # ------------------------------------------------------------
    centers = rng.uniform(-8.0, 8.0, size=(true_k, n_features))

    # Varianzas distintas por cluster (algo de diversidad)
    stds = rng.uniform(0.4, 1.8, size=true_k)

    X_parts = []
    for i in range(true_k):
        cov = (stds[i] ** 2) * np.eye(n_features)
        Xi = rng.multivariate_normal(mean=centers[i], cov=cov, size=int(sizes[i]))
        X_parts.append(Xi)

    X = np.vstack(X_parts)

    # Barajar filas para quitar estructura
    X = X[rng.permutation(n_samples)]

    # ------------------------------------------------------------
    # 3) Crear k_values (>=2) y asegurar variedad
    # ------------------------------------------------------------
    # Probar entre 3 y 6 valores de k
    num_k = int(rng.integers(3, 7))

    # Rango para k candidatos
    k_min = 2
    k_max = min(10, max(6, true_k + 4))  # hasta 10, pero al menos 6 o true_k+4

    k_candidates = np.arange(k_min, k_max + 1, dtype=int)
    k_values = rng.choice(k_candidates, size=num_k, replace=False)
    k_values = sorted(int(k) for k in k_values)

    random_state = int(rng.integers(0, 10_000))

    # ------------------------------------------------------------
    # 4) Construir INPUT
    # ------------------------------------------------------------
    input_data = {
        "X": X.astype(float),
        "k_values": k_values,
        "random_state": random_state,
    }

    # ------------------------------------------------------------
    # 5) Calcular OUTPUT esperado (Ground Truth) según el enunciado:
    #    - escalar X con StandardScaler (fit_transform en todo X)
    #    - para cada k: KMeans(n_clusters=k, random_state=..., n_init=10)
    #    - silhouette_score sobre X escalado
    #    - escoger mejor (si empate: k más pequeño)
    # ------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scores_dict = {}
    best_k = None
    best_score = None

    for k in k_values:
        model = KMeans(n_clusters=int(k), random_state=random_state, n_init=10)
        labels = model.fit_predict(X_scaled)

        score = float(silhouette_score(X_scaled, labels))
        scores_dict[int(k)] = score

        if best_score is None or score > best_score:
            best_score = score
            best_k = int(k)
        elif score == best_score and int(k) < int(best_k):
            best_k = int(k)

    output_data = {
        "best_k": int(best_k),
        "best_score": float(best_score),
        "scores": {int(k): float(v) for k, v in scores_dict.items()},
    }

    return input_data, output_data


if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_mejor_k_kmeans()

    print("=== INPUT ===")
    print("X shape:", entrada["X"].shape)
    print("k_values:", entrada["k_values"])
    print("random_state:", entrada["random_state"])
    print("X (primeras 3 filas):")
    print(np.round(entrada["X"][:3], 4))

    print("\n=== OUTPUT ESPERADO ===")
    print("best_k:", salida_esperada["best_k"])
    print("best_score:", round(salida_esperada["best_score"], 6))
    print("scores:")
    for k in sorted(salida_esperada["scores"].keys()):
        print(f"  k={k}: {salida_esperada['scores'][k]:.6f}")