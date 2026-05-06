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