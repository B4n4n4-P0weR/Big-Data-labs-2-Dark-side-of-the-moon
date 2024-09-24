import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Чтение данных
values_df = pd.read_csv("/Users/vlad/Desktop/Oskin's labs 2 Dark side of the moon/3-1/pokemonDB_dataset.csv", delimiter=',')

# Очистка данных и конвертация в числовой формат
values_df['Height'] = values_df['Height'].str.extract(r'(\d+\.\d+)').astype(float)
values_df['Weight'] = values_df['Weight'].str.extract(r'(\d+\.\d+)').astype(float)
values_df['HP Base'] = values_df['HP Base'].astype(float)

# Сортировка по столбцам 'Height'
values_df.sort_values(by=['Height', 'Weight'], inplace=True)

# Оригинальные данные для сравнения
original_data = values_df.copy()

# Пересчет среднего значения и стандартного отклонения для всего датасета
mean_weight = values_df['Weight'].mean()
std_weight = values_df['Weight'].std()

# Установка порога для определения аномалий на уровне среднее значение + 3 стандартных отклонения
weight_threshold = mean_weight + 5 * std_weight

# Применение этого порога для фильтрации данных
simple_filtered_data = values_df[(values_df['Weight'] <= weight_threshold) & (values_df['Weight'] != '—')]

# Просмотр результатов после применения упрощенного метода фильтрации
print(simple_filtered_data[['Pokemon', 'Height', 'Weight']].head())
print(original_data[['Pokemon', 'Height', 'Weight']].head())

# Визуализация
plt.figure(figsize=(12, 6))

# График без аномалий
# plt.subplot(1, 2, 2)
plt.plot(simple_filtered_data['Height'], simple_filtered_data['Weight'], 'g', label='Без аномалий')
plt.title('График без аномалий')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.legend()
plt.show()

# График с аномалиями
# plt.subplot(1, 2, 2)
plt.plot(original_data['Height'], original_data['Weight'], 'b', label='С аномалиями')
plt.title('График с аномалиями')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.legend()
plt.show()