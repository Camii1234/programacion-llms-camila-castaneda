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