from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

def run(X_train, y_train, X_test, y_test):
    # Шаг 1: Кодируем текстовые классы в числа
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Шаг 2: Создаём и обучаем модель
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train_enc)

    # Шаг 3: Предсказание для всех строк X_test
    prediction = model.predict(X_test)

    # Шаг 4: Переводим все предсказания обратно в строки
    predicted_labels = le.inverse_transform(prediction)

    # Шаг 5: Accuracy по всему массиву
    acc = accuracy_score(y_test, predicted_labels)

    # Шаг 6: Показываем одно из предсказаний пользователю
    return f"Predicted: {predicted_labels[0]}", acc
