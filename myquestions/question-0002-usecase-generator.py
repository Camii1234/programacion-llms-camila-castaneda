import pandas as pd
import numpy as np


def generar_caso_de_uso_matriz_transicion():
    """
    Genera un caso de uso aleatorio (input/output esperado) para la función:

        matriz_transicion(df, user_col, time_col, state_col)

    Retorna:
        input_data (dict): Diccionario con las claves esperadas por la función solución.
        output_data (pd.DataFrame): Matriz de transición esperada (probabilidades).
    """

    rng = np.random.default_rng()  # aleatorio distinto en cada ejecución

    # ------------------------------------------------------------
    # 1) Nombres de columnas (coherentes con el enunciado)
    # ------------------------------------------------------------
    user_col = "user_id"
    time_col = "timestamp"
    state_col = "state"

    # ------------------------------------------------------------
    # 2) Definir estados posibles (pantallas)
    # ------------------------------------------------------------
    possible_states = np.array(
        ["Home", "Search", "Product", "Cart", "Checkout", "Payment", "Profile", "Support"],
        dtype=object
    )

    # Escoger entre 4 y 7 estados para que la matriz no sea trivial
    n_states = int(rng.integers(4, 8))
    states = rng.choice(possible_states, size=n_states, replace=False)
    states = np.array(sorted(states.tolist()), dtype=object)  # para consistencia

    # ------------------------------------------------------------
    # 3) Generar eventos por usuario
    # ------------------------------------------------------------
    n_users = int(rng.integers(4, 10))  # 4 a 9 usuarios
    users = [f"U{str(i).zfill(3)}" for i in range(1, n_users + 1)]

    records = []
    base_time = pd.Timestamp("2026-02-01 08:00:00") + pd.Timedelta(minutes=int(rng.integers(0, 600)))

    for u in users:
        # 5 a 14 eventos por usuario
        n_events = int(rng.integers(5, 15))

        # Creamos un "camino" de navegación con alta probabilidad de avanzar y algo de ruido
        seq = []

        # Estado inicial típico
        seq.append("Home" if "Home" in states else rng.choice(states))

        for _ in range(1, n_events):
            prev = seq[-1]

            # Reglas simples de transición "realistas"
            if prev == "Home":
                candidates = ["Search", "Product", "Profile"]
            elif prev == "Search":
                candidates = ["Product", "Home", "Search"]
            elif prev == "Product":
                candidates = ["Cart", "Search", "Product"]
            elif prev == "Cart":
                candidates = ["Checkout", "Product", "Cart"]
            elif prev == "Checkout":
                candidates = ["Payment", "Cart", "Checkout"]
            elif prev == "Payment":
                candidates = ["Home", "Profile", "Support"]
            elif prev == "Profile":
                candidates = ["Home", "Support", "Profile"]
            else:  # Support u otros
                candidates = ["Home", "Profile", "Support"]

            # Filtrar candidatos a los que realmente existan en `states`
            candidates = [c for c in candidates if c in states]
            if len(candidates) == 0:
                next_state = rng.choice(states)
            else:
                # Sesgo al primer candidato (más probable)
                probs = np.linspace(0.5, 0.1, num=len(candidates))
                probs = probs / probs.sum()
                next_state = rng.choice(np.array(candidates, dtype=object), p=probs)

            seq.append(next_state)

        # Timestamps: incrementos aleatorios, luego desordenamos un poco para forzar que el algoritmo ordene
        increments = rng.integers(1, 30, size=n_events)  # 1 a 29 min
        times = [base_time]
        for inc in increments[1:]:
            times.append(times[-1] + pd.Timedelta(minutes=int(inc)))

        # A veces meter eventos con el mismo timestamp (caso realista)
        if rng.random() < 0.3 and n_events >= 7:
            j = int(rng.integers(1, n_events - 1))
            times[j] = times[j - 1]

        # Guardar registros
        for t, s in zip(times, seq):
            records.append(
                {
                    user_col: u,
                    time_col: t.strftime("%Y-%m-%d %H:%M:%S") if rng.random() < 0.6 else t,
                    state_col: s,
                }
            )

    df = pd.DataFrame(records)

    # Desordenar filas para asegurar que la solución debe ordenar
    df = df.sample(frac=1.0, random_state=int(rng.integers(0, 10_000))).reset_index(drop=True)

    # ------------------------------------------------------------
    # 4) Construir INPUT
    # ------------------------------------------------------------
    input_data = {
        "df": df.copy(),
        "user_col": user_col,
        "time_col": time_col,
        "state_col": state_col,
    }

    # ------------------------------------------------------------
    # 5) Calcular OUTPUT esperado (Ground Truth) replicando el enunciado
    # ------------------------------------------------------------
    expected = df.copy()

    # 1) Convertir time_col a datetime
    expected[time_col] = pd.to_datetime(expected[time_col], errors="coerce")

    # 2) Ordenar por user y time ascendente
    expected = expected.sort_values(by=[user_col, time_col], ascending=[True, True]).reset_index(drop=True)

    # 3) next_state por usuario
    expected["next_state"] = expected.groupby(user_col)[state_col].shift(-1)

    # 4) Contar transiciones state -> next_state, ignorando next_state NaN
    transitions = expected.dropna(subset=["next_state"])

    counts = (
        transitions
        .groupby([state_col, "next_state"])
        .size()
        .rename("count")
        .reset_index()
    )

    # 5) Construir matriz de conteos (filas=estado actual, cols=siguiente estado)
    count_matrix = (
        counts
        .pivot(index=state_col, columns="next_state", values="count")
        .fillna(0.0)
        .astype(float)
    )

    # 6) Asegurar matriz cuadrada con todos los estados vistos (actuales y siguientes)
    all_states = sorted(set(expected[state_col].dropna().unique()).union(set(transitions["next_state"].dropna().unique())))
    count_matrix = count_matrix.reindex(index=all_states, columns=all_states, fill_value=0.0)

    # 7) Convertir a probabilidades: cada fila suma 1.0 (si una fila es todo 0, se queda en 0)
    row_sums = count_matrix.sum(axis=1)
    prob_matrix = count_matrix.div(row_sums.replace(0.0, np.nan), axis=0).fillna(0.0)

    # 8) Orden alfabético en filas y columnas
    prob_matrix = prob_matrix.sort_index(axis=0).sort_index(axis=1)

    output_data = prob_matrix

    return input_data, output_data


if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_matriz_transicion()

    print("=== INPUT ===")
    print("Columnas:", list(entrada["df"].columns))
    print("user_col:", entrada["user_col"])
    print("time_col:", entrada["time_col"])
    print("state_col:", entrada["state_col"])
    print("\nDataFrame (primeras 12 filas):")
    print(entrada["df"].head(12))

    print("\n=== OUTPUT ESPERADO (matriz de transición) ===")
    print("Shape:", salida_esperada.shape)
    print(salida_esperada.head(10))

    # Verificación rápida de suma por fila (aprox 1.0 o 0.0)
    row_sums = salida_esperada.sum(axis=1)
    print("\nSuma por fila (primeros 10):")
    print(row_sums.head(10))