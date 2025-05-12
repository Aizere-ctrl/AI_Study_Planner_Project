from sklearn.neighbors import KNeighborsClassifier  # Импорт K-ближайших соседей
from sklearn.metrics import accuracy_score           # Метрика точности
from sklearn.preprocessing import LabelEncoder       # Кодирует строковые классы в числа

def run(X_train, y_train, X_test, y_test):
    # Шаг 1: Кодируем классы в числовой формат
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Шаг 2: Создаём модель KNN с 3 соседями
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train_enc)

    # Шаг 3: Предсказания для всех тестовых строк
    prediction = model.predict(X_test)

    # Шаг 4: Декодируем предсказания в строки
    predicted_labels = le.inverse_transform(prediction)

    # Шаг 5: Считаем точность
    acc = accuracy_score(y_test, predicted_labels)

    # Шаг 6: Возвращаем первую рекомендацию + точность
    return f"Predicted: {predicted_labels[0]}", acc
