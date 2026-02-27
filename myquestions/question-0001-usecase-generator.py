import pandas as pd
import numpy as np


def generar_caso_de_uso_detectar_solapamientos():
    """
    Genera un caso de uso aleatorio (input/output esperado) para la función:

        detectar_solapamientos(df, paciente_col, fecha_col, hora_inicio_col, duracion_min_col)

    Retorna:
        input_data (dict): Diccionario con las claves esperadas por la función solución.
        output_data (pd.DataFrame): DataFrame esperado tras aplicar la lógica del enunciado.
    """

    rng = np.random.default_rng()  # aleatorio distinto en cada ejecución

    # ------------------------------------------------------------
    # 1) Definir nombres de columnas (pueden ser fijos; lo importante es que sean coherentes)
    # ------------------------------------------------------------
    paciente_col = "paciente_id"
    fecha_col = "fecha"
    hora_inicio_col = "hora_inicio"
    duracion_min_col = "duracion_min"

    # ------------------------------------------------------------
    # 2) Generar datos aleatorios realistas
    # ------------------------------------------------------------
    n_pacientes = int(rng.integers(3, 8))          # 3 a 7 pacientes
    citas_por_paciente = int(rng.integers(4, 10)) # 4 a 9 citas por paciente
    total_citas = n_pacientes * citas_por_paciente

    pacientes = [f"P{str(i).zfill(3)}" for i in range(1, n_pacientes + 1)]

    # Rango de fechas (dentro de 7 días)
    base_date = pd.Timestamp("2026-02-01") + pd.Timedelta(days=int(rng.integers(0, 20)))
    fechas = [base_date + pd.Timedelta(days=int(rng.integers(0, 7))) for _ in range(total_citas)]

    # Horas de inicio en minutos desde 08:00 hasta 17:30 (pasos de 5 min)
    start_min_candidates = np.arange(8 * 60, 17 * 60 + 31, 5)
    start_minutes = rng.choice(start_min_candidates, size=total_citas, replace=True)

    # Duraciones típicas: 10-90 min, múltiplos de 5
    duration_candidates = np.arange(10, 91, 5)
    duraciones = rng.choice(duration_candidates, size=total_citas, replace=True)

    # Asignar paciente a cada cita (balanceado)
    paciente_list = []
    for p in pacientes:
        paciente_list.extend([p] * citas_por_paciente)
    rng.shuffle(paciente_list)

    df = pd.DataFrame(
        {
            paciente_col: paciente_list,
            fecha_col: [d.strftime("%Y-%m-%d") for d in fechas],  # string YYYY-MM-DD
            hora_inicio_col: [f"{m//60:02d}:{m%60:02d}" for m in start_minutes],
            duracion_min_col: duraciones.astype(int),
        }
    )

    # ------------------------------------------------------------
    # 3) Forzar que haya al menos un solapamiento en algunos casos
    #    (para evitar casos triviales donde nunca se solapa nada)
    # ------------------------------------------------------------
    if total_citas >= 6 and rng.random() < 0.85:
        # Elegimos un paciente y dos de sus citas para crear solapamiento
        p = rng.choice(pacientes)
        idxs = df.index[df[paciente_col] == p].to_list()
        if len(idxs) >= 2:
            i1, i2 = rng.choice(idxs, size=2, replace=False)

            # Poner ambas citas el mismo día
            same_day = base_date + pd.Timedelta(days=int(rng.integers(0, 7)))
            df.loc[i1, fecha_col] = same_day.strftime("%Y-%m-%d")
            df.loc[i2, fecha_col] = same_day.strftime("%Y-%m-%d")

            # Definir una hora base y asegurar solapamiento:
            # cita 1: empieza 09:00 dura 60, cita 2: empieza 09:30 dura 30 => solapa
            df.loc[i1, hora_inicio_col] = "09:00"
            df.loc[i1, duracion_min_col] = 60
            df.loc[i2, hora_inicio_col] = "09:30"
            df.loc[i2, duracion_min_col] = 30

    # ------------------------------------------------------------
    # 4) Construir INPUT
    # ------------------------------------------------------------
    input_data = {
        "df": df.copy(),
        "paciente_col": paciente_col,
        "fecha_col": fecha_col,
        "hora_inicio_col": hora_inicio_col,
        "duracion_min_col": duracion_min_col,
    }

    # ------------------------------------------------------------
    # 5) Calcular OUTPUT esperado (Ground Truth)
    #    Replicamos la lógica del enunciado:
    #    - inicio_dt = fecha + hora
    #    - fin_dt = inicio_dt + duración
    #    - ordenar por paciente, inicio_dt
    #    - solapada = inicio_dt < fin_dt_anterior por paciente
    # ------------------------------------------------------------
    expected = df.copy()

    # Convertir fecha a datetime y combinar con hora
    fecha_dt = pd.to_datetime(expected[fecha_col], errors="coerce")
    # Combinar fecha + hora (asumimos formato HH:MM)
    inicio_dt = pd.to_datetime(
        fecha_dt.dt.strftime("%Y-%m-%d") + " " + expected[hora_inicio_col].astype(str),
        errors="coerce",
    )
    expected["inicio_dt"] = inicio_dt

    # fin_dt
    expected["fin_dt"] = expected["inicio_dt"] + pd.to_timedelta(expected[duracion_min_col].astype(int), unit="m")

    # ordenar
    expected = expected.sort_values(by=[paciente_col, "inicio_dt"], ascending=[True, True]).reset_index(drop=True)

    # fin anterior por paciente
    fin_anterior = expected.groupby(paciente_col)["fin_dt"].shift(1)

    # solapamiento
    expected["solapada"] = (expected["inicio_dt"] < fin_anterior).fillna(False).astype(bool)

    output_data = expected

    return input_data, output_data


if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_detectar_solapamientos()

    print("=== INPUT ===")
    print("Columnas:", list(entrada["df"].columns))
    print("paciente_col:", entrada["paciente_col"])
    print("fecha_col:", entrada["fecha_col"])
    print("hora_inicio_col:", entrada["hora_inicio_col"])
    print("duracion_min_col:", entrada["duracion_min_col"])
    print("\nDataFrame (primeras 10 filas):")
    print(entrada["df"].head(10))

    print("\n=== OUTPUT ESPERADO (primeras 10 filas) ===")
    print(salida_esperada.head(10))

    # Conteo de solapadas para ver que no sea trivial
    print("\nSolapadas:", int(salida_esperada["solapada"].sum()))