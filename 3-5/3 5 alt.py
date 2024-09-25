import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Загрузка данных
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# 2. Предобработка данных
# Заполнение пропущенных значений
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
train_data = train_data.drop(columns=['Cabin'])

test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())
test_data = test_data.drop(columns=['Cabin'])

# Преобразование категориальных признаков в числовые (для модели SVM)
label_encoder = LabelEncoder()
train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])
train_data['Embarked'] = label_encoder.fit_transform(train_data['Embarked'])

test_data['Sex'] = label_encoder.fit_transform(test_data['Sex'])
test_data['Embarked'] = label_encoder.fit_transform(test_data['Embarked'])

# 3. Подготовка признаков и целевой переменной
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train_data[features]  # Признаки тренировочной выборки
y = train_data['Survived']  # Целевая переменная

X_test = test_data[features]  # Признаки тестовой выборки (без целевой переменной, так как она неизвестна)

# 4. Разделение на тренировочную и валидационную выборки
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Масштабирование признаков (это важно для SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 6. Построение и обучение модели SVM
svm_model = SVC(kernel='linear', C=1.0, random_state=42)  # Используем линейное ядро
svm_model.fit(X_train_scaled, y_train)

# 7. Оценка модели на кросс-валидации для тренировочной выборки
cross_val_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5)
print(f"Средняя точность модели на кросс-валидации: {cross_val_scores.mean():.2f}")

# 8. Предсказание на валидационной выборке
y_val_pred = svm_model.predict(X_val_scaled)

# 9. Оценка модели на валидационной выборке
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Точность модели на валидационной выборке: {accuracy:.2f}")

# Отчет по классификации
print("Отчет по классификации:\n", classification_report(y_val, y_val_pred))

# Матрица ошибок
print("Матрица ошибок:\n", confusion_matrix(y_val, y_val_pred))

# 10. Предсказания на тестовой выборке (предсказания для пассажиров из test.csv)
y_pred = svm_model.predict(X_test_scaled)

# Добавляем отладочные проверки
print("Предсказания модели для тестовой выборки (первые 10):")
print(y_pred[:10])  # Первые 10 предсказаний

print("Истинные значения целевой переменной для тренировочной выборки (первые 10):")
y_train_array = np.array(y_train)  # Преобразуем в NumPy массив
print(y_train_array[:10])  # Первые 10 значений в виде массива

# 11. Формирование файла для отправки (submission.csv)
# Используем только колонку 'PassengerId' из gender_submission.csv
submission_data = pd.read_csv("gender_submission.csv")[['PassengerId']]
submission_data['Survived'] = y_pred  # Добавляем предсказания модели в колонку 'Survived'

# 12. Сохранение предсказаний в файл
submission_data.to_csv('res/submission_alt.csv', index=False)

print("Предсказания сохранены в файл submission_alt.csv")
