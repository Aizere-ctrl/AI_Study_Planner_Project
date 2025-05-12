# Импорт Flask и вспомогательных библиотек
from flask import Flask, render_template, request
import numpy as np
import os
from werkzeug.utils import secure_filename

# Импорт всех AI-модулей (все алгоритмы отдельно)
from ai_modules import (
    linear_regression, logistic_regression, decision_tree, random_forest,
    naive_bayes, knn, svm, gradient_boosting,
    kmeans, fp_growth, pca, object_detection
)

# Инициализация Flask-приложения
app = Flask(__name__)

# Папка для загрузки изображений
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# 📘 Генерация расписания на основе введённых часов
def suggest_schedule_day(hours):
    reading = round(hours * 0.4, 2)
    practice = round(hours * 0.4, 2)
    rest = round(hours * 0.2, 2)
    return f"📘 Suggested per day: Reading {reading}h, Practice {practice}h, Rest {rest}h"

# Главный маршрут сайта ("/"), обрабатывает GET и POST
@app.route("/", methods=["GET", "POST"])
def index():
    task_a_results = []        # Сюда записываются результаты 8 supervised моделей
    task_b_results = {}        # Результаты 3 unsupervised моделей
    task_c_result = None       # Результат Computer Vision
    image_path = None          # Путь до изображения
    schedule_message = None    # Выводимая строка с расписанием

    # Если пользователь отправил форму (POST-запрос)
    if request.method == "POST":
        run_mode = request.form.get("run_mode")  # Определяем режим запуска

        # === Режим: только расписание ===
        if run_mode == "schedule_only":
            hours = float(request.form["hours"])
            schedule_message = suggest_schedule_day(hours)

        # === Режим: запустить все AI алгоритмы ===
        elif run_mode == "run_all":
            hours = float(request.form["hours"])
            days = int(request.form["days"])

            # 📚 Обучающая выборка (6 строк, 2 признака) — для обучения моделей
            X_train = np.array([
                [1, 2],
                [2, 1],
                [3, 5],
                [4, 4],
                [2, 3],
                [3, 2]
            ])
            y_train = np.array([
                "Reading",
                "Practice",
                "Rest",
                "Reading",
                "Practice",
                "Rest"
            ])

            # 🎯 Тестовая выборка (3 строки) — для честной оценки accuracy
            X_test = np.array([
                [1, 2],
                [3, 5],
                [2, 1]
            ])
            y_test = np.array([
                "Reading",
                "Rest",
                "Practice"
            ])

            # ✅ Task A: Supervised Learning (8 моделей)
            task_a_results = [
                ("Linear Regression", *linear_regression.run(X_train, y_train, X_test, y_test)),
                ("Logistic Regression", *logistic_regression.run(X_train, y_train, X_test, y_test)),
                ("Decision Tree", *decision_tree.run(X_train, y_train, X_test, y_test)),
                ("Random Forest", *random_forest.run(X_train, y_train, X_test, y_test)),
                ("Naive Bayes", *naive_bayes.run(X_train, y_train, X_test, y_test)),
                ("KNN", *knn.run(X_train, y_train, X_test, y_test)),
                ("SVM", *svm.run(X_train, y_train, X_test, y_test)),
                ("Gradient Boosting", *gradient_boosting.run(X_train, y_train, X_test, y_test))
            ]

            # 🧪 Task B: Unsupervised Learning
            task_b_results["KMeans"] = kmeans.run(X_train)
            task_b_results["FP-Growth"] = fp_growth.run([
                ["Reading", "Practice"],
                ["Rest"],
                ["Reading", "Rest"]
            ])
            task_b_results["PCA"] = pca.run(X_train)

            # 📅 Генерация расписания
            schedule_message = suggest_schedule_day(hours)

        # === Режим: компьютерное зрение ===
        elif run_mode == "run_cv":
            image = request.files.get("image_file")
            if image:
                filename = secure_filename(image.filename)
                saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                image.save(saved_path)
                task_c_result = object_detection.run(saved_path)
                image_path = f"/static/uploads/{filename}"

    # Возврат результата в шаблон index.html
    return render_template("index.html",
                           task_a_results=task_a_results,
                           task_b_results=task_b_results,
                           task_c_result=task_c_result,
                           image_path=image_path,
                           schedule_message=schedule_message)

# Запуск сервера
if __name__ == "__main__":
    app.run(debug=True)
