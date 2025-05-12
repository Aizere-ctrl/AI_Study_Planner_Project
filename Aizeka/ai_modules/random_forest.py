from sklearn.ensemble import RandomForestClassifier  # Импорт случайного леса
from sklearn.metrics import accuracy_score           # Метрика точности
from sklearn.preprocessing import LabelEncoder       # Кодирует метки в числа

def run(X_train, y_train, X_test, y_test):
    # Шаг 1: Кодируем строковые метки в числа
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Шаг 2: Создаём и обучаем модель случайного леса
    model = RandomForestClassifier()
    model.fit(X_train, y_train_enc)

    # Шаг 3: Предсказания для всех тестовых точек
    prediction = model.predict(X_test)

    # Шаг 4: Декодируем предсказания в текстовые метки
    predicted_labels = le.inverse_transform(prediction)

    # Шаг 5: Вычисляем точность по всем строкам
    acc = accuracy_score(y_test, predicted_labels)

    # Шаг 6: Возвращаем первое предсказание как пример + общую точность
    return f"Predicted: {predicted_labels[0]}", acc
