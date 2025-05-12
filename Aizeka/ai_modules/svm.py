from sklearn.svm import SVC                    # Импорт SVM-классификатора
from sklearn.metrics import accuracy_score     # Метрика accuracy
from sklearn.preprocessing import LabelEncoder # Преобразование строковых меток в числа

def run(X_train, y_train, X_test, y_test):
    # Шаг 1: Кодируем метки классов
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Шаг 2: Обучаем модель SVM с линейным ядром
    model = SVC(kernel="linear")
    model.fit(X_train, y_train_enc)

    # Шаг 3: Предсказываем классы для всего X_test
    prediction = model.predict(X_test)

    # Шаг 4: Декодируем все предсказания в строки
    predicted_labels = le.inverse_transform(prediction)

    # Шаг 5: Вычисляем точность по всем примерам
    acc = accuracy_score(y_test, predicted_labels)

    # Шаг 6: Показываем только первую рекомендацию
    return f"Predicted: {predicted_labels[0]}", acc
