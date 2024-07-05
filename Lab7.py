import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Завантаження даних
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
columns = ["variance", "skewness", "curtosis", "entropy", "class"]
data = pd.read_csv(url, names=columns, sep=",", header=None)

# Візуалізація даних
sns.pairplot(data, hue='class')
plt.title('Діаграма розсіювання даних аутентифікації банкнот')
plt.show()

# Розділення на ознаки та цільову змінну
X = data.drop(columns='class')
y = data['class']

# Розділення даних на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Створення та навчання моделей SVM з різними ядрами
svm_models = {}
for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
    svm_model = SVC(kernel=kernel)
    y_pred = svm_model.fit(X_train, y_train).predict(X_test)
    svm_models[kernel] = y_pred

# Функція для оцінки моделей
def evaluate_model_performance(y_true, y_pred, title):
    conf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.title(title)
    plt.show()
    print(classification_report(y_true, y_pred))

# Оцінка моделей з різними ядрами
for kernel, y_pred in svm_models.items():
    evaluate_model_performance(y_test, y_pred, f"Оцінка SVM з ядром {kernel.capitalize()}")

# Дослідження впливу параметра C на модель з RBF ядром
C_values = [0.1, 1, 10, 100]
for C in C_values:
    svm_rbf = SVC(kernel='rbf', C=C)
    y_pred = svm_rbf.fit(X_train, y_train).predict(X_test)
    evaluate_model_performance(y_test, y_pred, f"RBF SVM з C={C}")

# Дослідження впливу параметра gamma на модель з RBF ядром
gamma_values = [0.01, 0.1, 1, 10]
for gamma in gamma_values:
    svm_rbf = SVC(kernel='rbf', gamma=gamma)
    y_pred = svm_rbf.fit(X_train, y_train).predict(X_test)
    evaluate_model_performance(y_test, y_pred, f"RBF SVM з gamma={gamma}")
