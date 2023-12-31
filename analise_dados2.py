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
dados = pd.read_csv('data_estados2015.csv', delimiter=';', decimal=',', encoding='ISO-8859-1')

# Filtering by State
#dados_estados = dados[dados['uf'].isin(['SC', 'PR', 'RS'])].dropna().copy()
#dados_estados.to_csv('data_estados.csv', index=False, sep=';', decimal=',', encoding='ISO-8859-1')  # Setting index=False prevents pandas from writing row numbers.
# Filtering by state and fatal classification
dados_gerais = dados[(dados['estado_fisico'] == 'Ignorado    ')].dropna().copy()
dados_fatais = dados[(dados['estado_fisico'] == 'Morto       ')].dropna().copy()

# Filtering by state and serious injuries
dados_graves = dados[(dados['estado_fisico'] == 'Ferido Grave')].dropna().copy()

def criar_grafico(data, coluna, titulo, subpasta):
    plt.figure(figsize=(12, (10 if coluna == 'ano_fabricacao_veiculo' else 6)))
    
    if coluna == 'ano_fabricacao_veiculo':
        data_temporary = data[(data['ano_fabricacao_veiculo'] != '(null)')]
        y = data_temporary[coluna].value_counts().index
        x = data_temporary[coluna].value_counts().values
        # Create a list of colors based on y labels using color_map.
        # If a label is not in color_map, default to 'grey'.
        colors = [color_map.get(str(label), 'grey') for label in y]
        
        plt.barh(y, x, align='center', color=colors)
        plt.gca().invert_yaxis()
        
    elif coluna == 'causa_acidente':
        sizes = data[coluna].value_counts().values
        labels = data[coluna].value_counts().index.tolist()
        
        # Combine values smaller than 2% into an "Others" category
        total = sum(sizes)
        threshold = 0.02 * total
        others = sum([size for size in sizes if size < threshold])
        sizes = [size for size in sizes if size >= threshold]
        labels = [label for index, label in enumerate(labels) if data[coluna].value_counts()[label] >= threshold]
        
        # Add the combined values to the list of sizes and labels
        if others > 0:
            sizes.append(others)
            labels.append('Demais')
        
        # Create a list of colors based on labels using color_map.
        colors = [color_map.get(str(label), 'grey') for label in labels]
        
        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=45, colors=colors, wedgeprops=dict(width=0.3))
        
        # Set the font and format of the text
        plt.setp(autotexts, size=8, weight="bold")
        
        ax.set_title(titulo)

        
    elif coluna == 'tracado_via':
        values = data[coluna].value_counts().values
        categories = data[coluna].value_counts().index
        N = len(categories)
        theta = range(N)
        colors = [color_map.get(str(category), 'grey') for category in categories]
        
        # Assign colors to bars
        plt.bar(theta, values, color=colors)
        
        # Assign labels to X-axis
        plt.xticks(ticks=theta, labels=categories, rotation=45)
        
        # Manually assign colors to X-axis tick labels
        for i, tick in enumerate(plt.gca().get_xticklabels()):
            tick.set_color(colors[i])

    else:
        unique_labels = data[coluna].value_counts().index.tolist()
        colors = [color_map.get(str(label), 'grey') for label in unique_labels]
        print("Data shape: ", data.shape)
        print("Unique labels: ", unique_labels)

        sns.countplot(data=data, x=coluna, order=unique_labels, palette=colors)

    
    plt.title(titulo)
    plt.tight_layout()
    nome_arquivo = f"{subpasta}/{titulo}.png"
    plt.savefig(nome_arquivo)
    plt.close()

# Custom color map
color_map = {
    'Ceu Claro': 'lightskyblue',
    'Nublado': 'darkgrey',
    'Chuva': 'navy',
    'Sol': 'yellow',
    'Nevoeiro/neblina': 'gainsboro',
    'Ignorada': 'tomato',
    'Vento': 'violet',
    'Granizo': 'orange',
    'Neve': 'khaki',

    'Falta de atenção': 'yellow',
    'Outras': 'peru',
    'Não guardar distância de segurança': 'red',
    'Velocidade incompatível': 'grey',
    'Desobediência à sinalização': 'cyan',
    'Ingestão de álcool': 'lime',
    'Defeito mecânico em veículo' : 'indigo',
    'Ultrapassagem indevida': 'skyblue',
    'Dormindo': 'black',
    'Demais': 'gold',

    'Segunda': 'red',
    'Terça  ': 'orange',
    'Quarta ': 'yellow',
    'Quinta ': 'green',
    'Sexta  ': 'blue',
    'Sábado ': 'indigo',
    'Domingo': 'violet',

    'Pleno dia': 'yellow',
    'Plena noite': 'black',
    'Anoitecer': 'navy',
    'Amanhecer': 'orange',

    'Simples ': 'red',
    'Dupla   ': 'blue',
    'Múltipla': 'green',

    'Reta      ': 'red',
    'Curva     ': 'blue',
    'Cruzamento': 'green'
}

metricas = [
    ('condicao_metereologica', 'Acidentes por Condição Meteorológica'),
    ('tipo_pista', 'Acidentes por Tipo de Pista'),
    ('tracado_via', 'Acidentes por Traçado da Via'),
    ('fase_dia', 'Acidentes por Fase do Dia'),
    ('causa_acidente', 'Acidentes por Causa'),
    ('ano_fabricacao_veiculo', 'Acidentes por Ano de Fabricação'),
    ('dia_semana', 'Acidentes por Dia da Semana'),
]

metricas_filtro = [
    'condicao_metereologica',
    'tipo_pista',
    'tracado_via',
    'fase_dia',
    'causa_acidente',
    'ano_fabricacao_veiculo',
    'dia_semana']

def analisar_dataset(dataset, label):
    salvapasta = {"Todos os Acidentes": "Completo", "Acidentes Fatais": "Fatais", "Acidentes com Feridos Graves": "Graves"}.get(label, "Desconhecido")

    for coluna, titulo in metricas:
        dataset = dataset[~dataset[coluna].isna()]  # Remove null values
        criar_grafico(dataset, coluna, f"{titulo} ({label})", salvapasta)

    # Generating Circle Packing Plot
    # generate_circle_packing_plot(dataset, label, salvapasta)

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


dados_graves_semnull = dados_graves[dados_graves['ano_fabricacao_veiculo']!='(null)']
dados_graves_semvazio = dados_graves_semnull.fillna('vazia', inplace=True)
dados_graves_semvazio2 = dados_graves_semnull[dados_graves_semnull['ano_fabricacao_veiculo']!='    ']


dados_fatais_semnull = dados_fatais[dados_fatais['ano_fabricacao_veiculo']!='(null)']
dados_fatais_semvazio = dados_fatais_semnull.fillna('vazia', inplace=True)
dados_fatais_semvazio2 = dados_fatais_semnull[dados_fatais_semnull['ano_fabricacao_veiculo']!='    ']

dados_semnull = dados[dados['ano_fabricacao_veiculo']!='(null)']
dados_semvazio = dados_semnull.fillna('vazia', inplace=True)
dados_semvazio2 = dados_semnull[dados_semnull['ano_fabricacao_veiculo']!='    ']

analisar_dataset(dados_semvazio2, "Todos os Acidentes")
analisar_dataset(dados_fatais_semvazio2, "Acidentes Fatais")
analisar_dataset(dados_graves_semvazio2, "Acidentes com Feridos Graves")
