# Импорт необходимых библиотек
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Загрузка данных
data = pd.read_csv('Movie Reviews.csv')

# Преобразуем метки: 'pos' = 1 (положительный отзыв), 'neg' = 0 (отрицательный отзыв)
data['label'] = data['tag'].apply(lambda x: 1 if x == 'pos' else 0)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Преобразование текстов в векторное представление (TF-IDF или CountVectorizer)
# Удаляем стоп-слова и приводим к нижнему регистру
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Обучение модели
model = MultinomialNB(alpha=1.0)  # Параметр alpha можно оптимизировать позже
model.fit(X_train_vectorized, y_train)

# Прогнозирование и оценка
y_pred = model.predict(X_test_vectorized)

# Выводим результаты оценки
print("Точность модели:", accuracy_score(y_test, y_pred))
print("Отчёт классификации:\n", classification_report(y_test, y_pred))
