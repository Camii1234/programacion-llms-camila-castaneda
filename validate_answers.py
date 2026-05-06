from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def load_module(file_name: str):
    file_path = ROOT / "myanswers" / file_name
    spec = spec_from_file_location(file_name.replace("-", "_"), file_path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_0166():
    mod = load_module("answer-0166.py")
    y_true = [0, 1, 1, 0, 1, 0]
    y_scores = [0.1, 0.9, 0.8, 0.2, 0.7, 0.3]
    fpr, tpr, auc = mod.calcular_curva_roc(y_true, y_scores)
    print("answer-0166 -> OK")
    print("  fpr:", fpr)
    print("  tpr:", tpr)
    print("  auc:", auc)


def run_0464():
    mod = load_module("answer-0464.py")
    df = pd.DataFrame(
        {
            "grupo": ["A", "A", "B", "B", "C"],
            "valor": [10, 20, 15, 25, 30],
        }
    )
    resumen = mod.resumir_por_grupo_y_normalizar(df, "grupo", "valor")
    print("answer-0464 -> OK")
    print(resumen)


def run_0327():
    mod = load_module("answer-0327.py")
    textos = [
        "No puedo iniciar sesion en la plataforma",
        "Error al procesar el pago",
        "Solicitud de cambio de contrasena",
        "Quiero actualizar mis datos de perfil",
    ]
    etiquetas = np.array(["acceso", "pago", "acceso", "perfil"])
    nuevo_mensaje = "Tengo problemas para entrar a mi cuenta"
    pred = mod.clasificar_tickets_soporte(textos, etiquetas, nuevo_mensaje)
    print("answer-0327 -> OK")
    print("  prediccion:", pred)


def run_0210():
    mod = load_module("answer-0210.py")
    df = pd.DataFrame({"valores": [10, 11, 12, 10, 9, 11, 200, -50]})
    outliers = mod.detectar_outliers(df)
    print("answer-0210 -> OK")
    print(outliers)


def main():
    run_0166()
    print()
    run_0464()
    print()
    run_0327()
    print()
    run_0210()


if __name__ == "__main__":
    main()
