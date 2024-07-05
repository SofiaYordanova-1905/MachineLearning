import numpy as np
import matplotlib.pyplot as plt
import re

# Завдання 1
# Функція: Y(x) = (1/x) * sin(5x) для x в діапазоні [-5, 5]
x_values = np.linspace(-5, 5, 1000)  # Генерація 1000 точок між -5 і 5

# Замінюємо x = 0 на маленьке значення, щоб уникнути ділення на нуль
small_value = 1e-10
x_values = np.where(x_values == 0, small_value, x_values)

y_values = (1 / x_values) * np.sin(5 * x_values)

plt.plot(x_values, y_values)
plt.title('Графік функції Y(x) = (1/x) * sin(5x)')
plt.xlabel('x')
plt.ylabel('Y(x)')
plt.grid(True)
plt.show()

# Завдання 2
# Зчитування вмісту текстового файлу
with open('text.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Розрахунок частоти кожної букви
frequency_of_letters = {}
for char in text:
    if char.isalpha():
        char_lower = char.lower()
        frequency_of_letters[char_lower] = frequency_of_letters.get(char_lower, 0) + 1

# Побудова гістограми частот літер
letters = list(frequency_of_letters.keys())
frequency = list(frequency_of_letters.values())

plt.bar(letters, frequency, color='skyblue')
plt.xlabel('Літери')
plt.ylabel('Частота')
plt.title('Гістограма частот літер у тексті')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Завдання 3
# Підрахунок різних типів речень у тексті
number_of_dots = text.count('.') - text.count('...')
number_of_questions = text.count('?')
number_of_exclamation_marks = text.count('!')
number_of_3points = len(re.findall(r'\.\.\.(?!\w)', text))

types_of_sentences = ['Розповідні', 'Питальні', 'Окличні', 'Триточкові']
number_types_of_sentences = [number_of_dots, number_of_questions, number_of_exclamation_marks, number_of_3points]

plt.bar(types_of_sentences, number_types_of_sentences, color='salmon')
plt.xlabel('Типи речень')
plt.ylabel('Частота')
plt.title('Гістограма частот типів речень')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
