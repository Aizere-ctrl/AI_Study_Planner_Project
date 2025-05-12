from sklearn.decomposition import PCA  # Импорт PCA из sklearn

def run(X):
    # Шаг 1: Создаём объект PCA и задаём количество компонент = 2
    pca = PCA(n_components=2)

    # Шаг 2: Применяем PCA — снижаем размерность до 2D
    components = pca.fit_transform(X)

    # Шаг 3: Сохраняем, какую часть дисперсии объясняют каждая компонента
    explained = pca.explained_variance_ratio_

    # Шаг 4: Формируем читаемый текст
    result_text = f"PCA 2D: Explained variance: {explained[0]:.2f}, {explained[1]:.2f}"

    return result_text
