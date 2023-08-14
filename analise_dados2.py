import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import plotly.express as px

# Create directories if not exists
subpastas = ["Completo", "Fatais", "Graves"]
for subpasta in subpastas:
    os.makedirs(subpasta, exist_ok=True)

# Seaborn settings
sns.set_theme(style="whitegrid")
palette = sns.color_palette("pastel")

# Load dataset
dados = pd.read_csv('acidentes2015.csv', delimiter=';', decimal=',', encoding='ISO-8859-1')

# Filtering by State
dados_estados = dados[dados['uf'].isin(['SC', 'PR', 'RS'])].dropna().copy()

# Filtering by state and fatal classification
dados_fatais = dados[(dados['uf'].isin(['SC', 'PR', 'RS'])) & (dados['estado_fisico'] == 'Morto       ')].dropna().copy()

# Filtering by state and serious injuries
dados_graves = dados[(dados['uf'].isin(['SC', 'PR', 'RS'])) & (dados['estado_fisico'] == 'Ferido Grave')].dropna().copy()

def criar_grafico(data, coluna, titulo, subpasta):
    plt.figure(figsize=(12, (10 if coluna == 'ano_fabricacao_veiculo' else 6)))
    
    if coluna == 'ano_fabricacao_veiculo':
        y = data[coluna].value_counts().index
        x = data[coluna].value_counts().values
        plt.barh(y, x, align='center')
        plt.gca().invert_yaxis()
        
    elif coluna == 'causa_acidente':
        sizes = data[coluna].value_counts().values
        labels = data[coluna].value_counts().index.tolist()
        
        # Combine os valores menores que 2% em uma categoria "Demais"
        total = sum(sizes)
        threshold = 0.02 * total
        others = sum([size for size in sizes if size < threshold])
        sizes = [size for size in sizes if size >= threshold]
        labels = [label for index, label in enumerate(labels) if data[coluna].value_counts()[label] >= threshold]
        
        # Adicione os valores combinados à lista de tamanhos e rótulos
        if others > 0:
            sizes.append(others)
            labels.append('Demais')
        
        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=45, wedgeprops=dict(width=0.3))
        
        # Define a fonte e o formato dos textos
        plt.setp(autotexts, size=8, weight="bold")
        
        ax.set_title(titulo)
        
    elif coluna == 'tipo_veiculo':
        values = data[coluna].value_counts().values
        categories = data[coluna].value_counts().index
        N = len(categories)
        theta = range(N)
        plt.bar(theta, values)
        plt.xticks(ticks=theta, labels=categories, rotation=45)
        
    else:
        sns.countplot(data=data, x=coluna, order=data[coluna].value_counts().index, palette=palette)
    
    plt.title(titulo)
    plt.tight_layout()
    nome_arquivo = f"{subpasta}/{titulo}.png"
    plt.savefig(nome_arquivo)
    plt.close()

metricas = [
    ('condicao_metereologica', 'Acidentes por Condição Meteorológica'),
    ('tipo_pista', 'Acidentes por Tipo de Pista'),
    ('tipo_veiculo', 'Acidentes por Tipo de Veículo'),
    ('fase_dia', 'Acidentes por Fase do Dia'),
    ('causa_acidente', 'Acidentes por Causa'),
    ('ano_fabricacao_veiculo', 'Acidentes por Ano de Fabricação'),
    ('dia_semana', 'Acidentes por Dia da Semana'),
]

def analisar_dataset(dataset, label):
    salvapasta = {"Todos os Acidentes": "Completo", "Acidentes Fatais": "Fatais", "Acidentes com Feridos Graves": "Graves"}.get(label, "Desconhecido")

    for coluna, titulo in metricas:
        dataset = dataset[~dataset[coluna].isna()]  # Remove null values
        criar_grafico(dataset, coluna, f"{titulo} ({label})", salvapasta)

    # Generating Circle Packing Plot
    generate_circle_packing_plot(dataset, label, salvapasta)

def generate_circle_packing_plot(dataset, label, salvapasta):
    top_cities = dataset['municipio'].value_counts().head(10).index.tolist()
    filtered_data = dataset[dataset['municipio'].isin(top_cities)]
    if label == "Todos os Acidentes":
        fig = px.sunburst(
            filtered_data,
            path=['municipio', 'fase_dia', 'classificacao_acidente'],
            title=f'Circle Packing for {label}'
        )
    elif label == "Acidentes Fatais":
        fig = px.sunburst(
            filtered_data,
            path=['municipio', 'fase_dia', 'condicao_metereologica'],
            title=f'Circle Packing for {label}'
        )
    elif label == "Acidentes com Feridos Graves":
        fig = px.sunburst(
            filtered_data,
            path=['municipio', 'fase_dia', 'condicao_metereologica'],
            title=f'Circle Packing for {label}'
        )
    else:
        print("Wrong label, skipping circle plot")
        return
    filepath = os.path.join(salvapasta, f"circle_packing_{label}.png")
    fig.write_image(filepath)
    fig.show()

analisar_dataset(dados_estados, "Todos os Acidentes")
analisar_dataset(dados_fatais, "Acidentes Fatais")
analisar_dataset(dados_graves, "Acidentes com Feridos Graves")
