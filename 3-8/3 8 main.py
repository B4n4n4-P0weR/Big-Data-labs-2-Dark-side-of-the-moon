import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Чтение данных

train_df = pd.read_csv('../titanic/train.csv')
test_df = pd.read_csv('../titanic/test.csv')
gender_submission_df = pd.read_csv('../titanic/gender_submission.csv')

# Убираем ненужные для кластеризации столбцы, оставляем числовые данные
train_data = train_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked', 'Survived'])
test_data = test_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'])

# Обработка пропущенных данных - заполним медианными значениями
train_data.fillna(train_data.median(numeric_only=True), inplace=True)
test_data.fillna(test_data.median(numeric_only=True), inplace=True)

# Преобразуем категориальные данные в числовые (например, пол)
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})

# Нормализуем данные, чтобы все признаки имели одинаковый масштаб
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# Применение KMeans
kmeans_inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(train_scaled)
    kmeans_inertia.append(kmeans.inertia_)

# Построение графика для выбора числа кластеров (метод локтя)
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), kmeans_inertia, marker='o')
plt.title('Метод локтя для K-means')
plt.xlabel('Количество кластеров')
plt.ylabel('Сумма квадратов расстояний (инерция)')
plt.grid(True)
plt.show()

# Применение EM (Gaussian Mixture)
gm_bic = []
for k in range(1, 11):
    gm = GaussianMixture(n_components=k, random_state=42)
    gm.fit(train_scaled)
    gm_bic.append(gm.bic(train_scaled))

# Построение графика для выбора числа компонент смеси (по BIC)
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), gm_bic, marker='o', color='r')
plt.title('Выбор числа компонент смеси для EM (по BIC)')
plt.xlabel('Количество компонент')
plt.ylabel('BIC')
plt.grid(True)
plt.show()

# Определение оптимального количества кластеров и компонент смеси
optimal_k = 3  # Пример значения, выберите оптимальный K по графику
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(train_scaled)

# Выводим центры кластеров
print("Центры кластеров для K-means:")
print(kmeans.cluster_centers_)

# Применение EM с оптимальным количеством компонент
gm = GaussianMixture(n_components=optimal_k, random_state=42)
gm.fit(train_scaled)

# Выводим центры компонент для EM
print("Центры компонент для EM:")
print(gm.means_)
