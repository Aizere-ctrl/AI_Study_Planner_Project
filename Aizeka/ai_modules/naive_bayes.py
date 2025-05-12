from sklearn.naive_bayes import GaussianNB  # Наивный Байес для непрерывных данных
from sklearn.metrics import accuracy_score   # Метрика точности
from sklearn.preprocessing import LabelEncoder  # Кодирует классы в числа

def run(X_train, y_train, X_test, y_test):
    # Шаг 1: Кодируем строковые метки в числа
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Шаг 2: Обучаем модель
    model = GaussianNB()
    model.fit(X_train, y_train_enc)

    # Шаг 3: Предсказания для всех строк X_test
    prediction = model.predict(X_test)

    # Шаг 4: Декодируем числовые классы обратно в строки
    predicted_labels = le.inverse_transform(prediction)

    # Шаг 5: Вычисляем accuracy по всем меткам
    acc = accuracy_score(y_test, predicted_labels)

    # Шаг 6: Возвращаем первую строку как пример + точность
    return f"Predicted: {predicted_labels[0]}", acc
