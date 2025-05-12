from sklearn.cluster import KMeans  # Импорт алгоритма K-средних

def run(X):
    # Шаг 1: Инициализируем модель KMeans с 3 кластерами
    model = KMeans(n_clusters=3, random_state=0)

    # Шаг 2: Обучаем модель на входных данных (без меток!)
    model.fit(X)

    # Шаг 3: Получаем метки кластеров для каждой точки (например: 0, 1, 2)
    labels = model.labels_

    # Центры кластеров (координаты "средних" точек)
    center = model.cluster_centers_

    # Формируем читаемый результат
    result_text = f"Cluster labels: {labels.tolist()}"
    return result_text
