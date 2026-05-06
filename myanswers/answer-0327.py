import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def clasificar_tickets_soporte(textos, etiquetas, nuevo_mensaje):
    vectorizador = TfidfVectorizer()
    X = vectorizador.fit_transform(textos)

    modelo = MultinomialNB()
    modelo.fit(X, etiquetas)

    X_nuevo = vectorizador.transform([nuevo_mensaje])
    prediccion = modelo.predict(X_nuevo)

    return prediccion