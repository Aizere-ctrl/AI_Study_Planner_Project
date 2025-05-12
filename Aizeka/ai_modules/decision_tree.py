from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def run(X_train, y_train, X_test, y_test):
    # Шаг 1: Кодируем строковые метки в числа
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Шаг 2: Обучаем дерево решений
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train_enc)

    # Шаг 3: Предсказания на всех X_test
    prediction = model.predict(X_test)

    # Шаг 4: Обратно декодируем все предсказания
    predicted_labels = le.inverse_transform(prediction)

    # Шаг 5: Accuracy — по всему массиву
    acc = accuracy_score(y_test, predicted_labels)

    # Шаг 6: Показываем одно из предсказаний (для таблицы на сайте)
    return f"Predicted: {predicted_labels[0]}", acc
