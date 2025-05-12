from sklearn.ensemble import GradientBoostingClassifier  # Импорт градиентного бустинга
from sklearn.metrics import accuracy_score               # Метрика accuracy
from sklearn.preprocessing import LabelEncoder           # Кодировка строковых меток

def run(X_train, y_train, X_test, y_test):
    # Шаг 1: Преобразуем текстовые метки в числа
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Шаг 2: Обучаем модель градиентного бустинга
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train_enc)

    # Шаг 3: Предсказания для всех строк X_test
    prediction = model.predict(X_test)

    # Шаг 4: Декодируем предсказания в строковые метки
    predicted_labels = le.inverse_transform(prediction)

    # Шаг 5: Вычисляем точность
    acc = accuracy_score(y_test, predicted_labels)

    # Шаг 6: Возвращаем первую рекомендацию + точность
    return f"Predicted: {predicted_labels[0]}", acc
