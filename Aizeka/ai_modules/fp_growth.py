import pandas as pd
from mlxtend.preprocessing import TransactionEncoder     # Для преобразования транзакций в матрицу
from mlxtend.frequent_patterns import fpgrowth, association_rules  # FP-Growth и генерация правил

def run(transactions):
    # Шаг 1: Преобразуем список транзакций в бинарную матрицу (1/0 — присутствие элемента)
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Шаг 2: Применяем алгоритм FP-Growth для поиска частых наборов (поддержка ≥ 50%)
    freq_items = fpgrowth(df, min_support=0.5, use_colnames=True)

    # Шаг 3: Генерируем ассоциативные правила (доверие ≥ 70%)
    rules = association_rules(freq_items, metric="confidence", min_threshold=0.7)

    # Шаг 4: Формируем результат в виде "A → B"
    result = []
    for _, row in rules.iterrows():
        antecedent = ', '.join(list(row['antecedents']))
        consequent = ', '.join(list(row['consequents']))
        result.append(f"{antecedent} → {consequent}")

    # Шаг 5: Если правил нет — возвращаем сообщение
    return result if result else ["No strong rules found"]

