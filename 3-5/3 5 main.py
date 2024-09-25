import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

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
X_train = train_data[features]  # Признаки тренировочной выборки
y_train = train_data['Survived']  # Целевая переменная

X_test = test_data[features]  # Признаки тестовой выборки (без целевой переменной, так как она неизвестна)

# 4. Масштабирование признаков (это важно для SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Построение и обучение модели SVM с rbf-ядром
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)  # Используем rbf-ядро для большей гибкости
svm_model.fit(X_train_scaled, y_train)



# 6. Оценка модели на кросс-валидации для тренировочной выборки
cross_val_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5)
print(f"Средняя точность модели на кросс-валидации: {cross_val_scores.mean():.2f}")

# 7. Предсказания на тестовой выборке (предсказания для пассажиров из test.csv)
y_pred = svm_model.predict(X_test_scaled)

# Добавляем отладочную проверку
print("Предсказания модели для тестовой выборки:")
print(y_pred[:10])  # Проверим первые 10 предсказаний

real_data = pd.read_csv("gender_submission.csv")
y_real = real_data['Survived']
# Преобразуем y_train в массив NumPy и выведем первые 10 значений
y_real_array = np.array(y_real)  # Преобразуем в NumPy массив
print("Истинные значения целевой переменной для реальной выборки (первые 10):")
print(y_real_array[:10])  # Первые 10 значений в виде массива


# 8. Формирование файла для отправки (submission.csv)
# Используем только колонку 'PassengerId' из gender_submission.csv
submission_data = pd.read_csv("gender_submission.csv")[['PassengerId']]
submission_data['Survived'] = y_pred  # Добавляем предсказания модели в колонку 'Survived'

# 9. Сохранение предсказаний в файл
submission_data.to_csv('submission.csv', index=False)

print("Предсказания сохранены в файл submission.csv")
