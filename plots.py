import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

colors = [
    (0.27, 0.09, 0.49, 0.7),
    (0.23, 0.32, 0.55, 0.7),
    (0.13, 0.58, 0.55, 0.7),
    (0.37, 0.79, 0.38, 0.7),
    (0.99, 0.91, 0.14, 0.7)
]

def plot_pie(colors):
    generos = ['Ficção', 'Não-ficção', 'Literatura Clássica', 'Literatura Infantil e Juvenil', 'Humor']
    porcentagens = [77.58, 5.17, 3.50, 10.95, 2.80]

    plt.figure(figsize=(10, 8))
    plt.pie(porcentagens, labels=generos, autopct='%1.1f%%', colors=colors)

    plt.title('Distribuição dos Gêneros')
    plt.savefig('./output/plot/grafico_pizza.png')


def plot_models(folder_path):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    english_level_colors = {
        'A1': (0.99, 0.91, 0.14, 0.7),
        'A2': (0.27, 0.09, 0.49, 0.7),
        'B1': (0.23, 0.32, 0.55, 0.7),
        'B2': (0.13, 0.58, 0.55, 0.7),
        'C1': (0.37, 0.79, 0.38, 0.7)}

    num_rows = -(-len(csv_files) // 2)  # Arredondamento para cima da divisão

    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(18, 6 * num_rows))  # 3 colunas

    for i, csv_file in enumerate(csv_files):
        row = i // 2  # Calcula o índice da linha
        col = i % 2  # Calcula o índice da coluna

        df = pd.read_csv(os.path.join(folder_path, csv_file))

        for level, color in english_level_colors.items():
            subset = df[df['english_level'] == level]
            axes[row, col].hist(subset['difficulty'], bins=10, label=level, color=color)
        axes[row, col].set_xlabel('Dificuldade')
        axes[row, col].set_ylabel('Contagem')
        axes[row, col].set_title(f'{csv_file}')
        axes[row, col].legend()

    # Remova os eixos extras se houver menos subplots do que as células na grade
    for i in range(len(csv_files), num_rows * 2):
        axes.flat[i].axis('off')

    plt.tight_layout()
    plt.savefig('./output/plot/hist_diff.png')

# Exemplo de uso
folder_path = "./output/models"  # Substitua pelo caminho correto para a sua pasta
plot_models(folder_path)

plot_pie(colors)
