import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


def calcular_curva_roc(y_true, y_scores):
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = float(roc_auc_score(y_true, y_scores))

    return fpr, tpr, auc