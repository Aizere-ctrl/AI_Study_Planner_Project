from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

def run(X_train, y_train, X_test, y_test):
    # Кодируем текстовые метки в числа
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Обучаем модель
    model = LinearRegression()
    model.fit(X_train, y_train_enc)

    # Предсказание
    prediction = model.predict(X_test)  # Массив из N чисел

    # Округляем до ближайшего класса и ограничиваем в допустимых пределах
    prediction_rounded = np.round(prediction).astype(int)
    prediction_rounded = np.clip(prediction_rounded, 0, len(le.classes_) - 1)

    # Декодируем в текстовые метки
    predicted_labels = le.inverse_transform(prediction_rounded)

    # Accuracy по всей тестовой выборке
    acc = accuracy_score(y_test, predicted_labels)

    # Возвращаем только первое предсказание как пример + точность
    return f"Predicted: {predicted_labels[0]}", acc
