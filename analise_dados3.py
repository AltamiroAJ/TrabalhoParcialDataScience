import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from unidecode import unidecode


def generate_correlation_heatmap(dataset,subpasta):
    # Copy the dataset to keep the original intact
    encoded_data = dataset.copy()

    # Remove specific columns
    columns_to_remove = ['id', 'pesid', 'id_veiculo', 'ilesos',	'feridos_leves', 'feridos_graves', 'mortos', 'latitude', 'longitude', 'regional', 'delegacia', 'uop', 'naturalidade', 'nacionalidade']
    encoded_data = encoded_data.drop(columns=columns_to_remove, errors='ignore')  # 'errors='ignore'' ensures the code doesn't break if the columns don't exist

    # Initialize label encoder
    labelencoder = LabelEncoder()

    # Apply LabelEncoder to each column
    for col in encoded_data.columns:
        encoded_data[col] = labelencoder.fit_transform(encoded_data[col].astype(str))

    # Compute the correlation matrix for all columns
    correlation_matrix_encoded = encoded_data.corr()

    # Generate a heatmap for all columns
    plt.figure(figsize=(20, 20))  # You may want to adjust the figure size based on the number of columns
    sns.heatmap(correlation_matrix_encoded, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap for All Columns (Encoded)")
    plt.tight_layout()

    # Save the heatmap
    nome_arquivo_encoded = f"{subpasta}\Correlation_Heatmap_All_Encoded_Columns.png"
    plt.savefig(nome_arquivo_encoded)
    plt.close()

def encode_qualitative_to_numeric(dataset, metricas_filtro):
    encoded_dataset = dataset.astype(str).copy()  # Deep copy of the original dataset
    le = LabelEncoder()  # Initialize label encoder
    
    for col in metricas_filtro:
        # Ensure the column does not contain NaN values
        if encoded_dataset[col].isna().any():
            encoded_dataset[col].fillna("Unknown", inplace=True)  # Filling NaN values with "Unknown"
        
        encoded_dataset[col] = le.fit_transform(encoded_dataset[col])  # Encode qualitative column to numerical
    
    return encoded_dataset

# Create directories if not exists
subpastas = ["Completo", "Fatais", "Graves"]
for subpasta in subpastas:
    os.makedirs(subpasta, exist_ok=True)

# Seaborn settings
sns.set_theme(style="whitegrid")
palette = sns.color_palette("pastel")

# Load dataset
#dados = pd.read_csv('acidentes2017.csv', delimiter=';', decimal=',', encoding='ISO-8859-1')
filenames = [
    'acidentes2007.csv',
    'acidentes2008.csv',
    'acidentes2009.csv',
    'acidentes2010.csv',
    'acidentes2011.csv',
    'acidentes2012.csv',
    'acidentes2013.csv',
    'acidentes2014.csv',
    'acidentes2015.csv',
    'acidentes2016.csv',
    'acidentes2017.csv',
    'acidentes2018.csv',
    'acidentes2019.csv',
    'acidentes2020.csv',
    'acidentes2021.csv',
    'acidentes2022.csv'
]

dfs = [pd.read_csv(filename, delimiter=';', decimal=',', encoding='ISO-8859-1', low_memory=False) for filename in filenames]

dados = pd.concat(dfs, ignore_index=True)
def clean_text(val):
    if isinstance(val, str):
        val = val.lower()
        val = unidecode(val)
        val = val.strip()
    return val

dados = dados.applymap(clean_text)
dados['dia_semana'] = dados['dia_semana'].replace('segunda-feira', 'segunda')
dados['dia_semana'] = dados['dia_semana'].replace('terca-feira', 'terca')
dados['dia_semana'] = dados['dia_semana'].replace('quarta-feira', 'quarta')
dados['dia_semana'] = dados['dia_semana'].replace('quinta-feira', 'quinta')
dados['dia_semana'] = dados['dia_semana'].replace('sexta-feira', 'sexta')

# Filtering by State
#dados_estados = dados[dados['uf'].isin(['SC', 'PR', 'RS'])].dropna().copy()
#dados_estados.to_csv('data_estados.csv', index=False, sep=';', decimal=',', encoding='ISO-8859-1')  # Setting index=False prevents pandas from writing row numbers.
# Filtering by state and fatal classification
#dados_gerais = dados[dados['uf'].isin(['SC', 'PR', 'RS'])].copy()
#dados_fatais = dados[dados['uf'].isin(['SC', 'PR', 'RS']) & dados['estado_fisico'].isin(['Morto       ', 'Óbito'])].copy()
dados_gerais = dados[dados['uf'].isin(['sc', 'pr', 'rs'])].copy()
dados_fatais = dados[dados['uf'].isin(['sc', 'pr', 'rs']) & dados['estado_fisico'].isin(['morto', 'obito'])].copy()
dados_graves = dados[dados['uf'].isin(['sc', 'pr', 'rs']) & dados['estado_fisico'].isin(['ferido grave','lesoes graves'])].copy()

# Filtering by state and serious injuries
#dados_graves = dados[dados['uf'].isin(['SC', 'PR', 'RS']) & dados['estado_fisico'].isin(['Ferido Grave','Lesões Graves'])].copy()

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
            labels.append('demais')
        
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

    # Encode the dataset
    encoded_dataset = encode_qualitative_to_numeric(data, metricas_filtro)

    # Calculate the correlation matrix
    correlation_matrix = encoded_dataset[metricas_filtro].corr()

    # Generate a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(f"Correlation Heatmap ({titulo})")
    plt.tight_layout()

    # Save the heatmap
    nome_arquivo = f"{subpasta}/Correlation_Heatmap_{subpasta}.png"
    plt.savefig(nome_arquivo)
    plt.close()

# Custom color map
color_map = {
    'ceu claro': 'lightskyblue',
    'nublado': 'darkgrey',
    'chuva': 'navy',
    'sol': 'yellow',
    'nevoeiro/neblina': 'gainsboro',
    'ignorada': 'tomato',
    'vento': 'violet',
    'granizo': 'orange',
    'neve': 'khaki',

    'falta de atencao': 'yellow',
    'outras': 'peru',
    'nao guardar distancia de segurança': 'red',
    'velocidade incompativel': 'grey',
    'desobediencia a sinalizacao': 'cyan',
    'ingestao de alcool': 'lime',
    'defeito mecanico em veiculo' : 'indigo',
    'ultrapassagem indevida': 'skyblue',
    'dormindo': 'black',
    'demais': 'gold',

    'segunda': 'red',
    'terca': 'orange',
    'quarta': 'yellow',
    'quinta': 'green',
    'sexta': 'blue',
    'sabado': 'indigo',
    'domingo': 'violet',

    'pleno dia': 'yellow',
    'plena noite': 'black',
    'anoitecer': 'navy',
    'amanhecer': 'orange',

    'simples': 'red',
    'dupla': 'blue',
    'múltipla': 'green',

    'reta': 'tomato',
    'curva': 'cyan',
    'cruzamento': 'lime'
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


dados_graves_semnull = dados_graves[dados_graves['ano_fabricacao_veiculo']!='(null)'].astype(str).copy()
dados_graves_semvazio = dados_graves_semnull.fillna('Vazia', inplace=True)
dados_graves_semvazio2 = dados_graves_semnull[dados_graves_semnull['ano_fabricacao_veiculo']!='    ']


dados_fatais_semnull = dados_fatais[dados_fatais['ano_fabricacao_veiculo']!='(null)'].astype(str).copy()
dados_fatais_semvazio = dados_fatais_semnull.fillna('vazia', inplace=True)
dados_fatais_semvazio2 = dados_fatais_semnull[dados_fatais_semnull['ano_fabricacao_veiculo']!='    ']

dados_semnull = dados_gerais[dados['ano_fabricacao_veiculo']!='(null)'].astype(str).copy()
dados_semvazio = dados_semnull.fillna('vazia', inplace=True)
dados_semvazio2 = dados_semnull[dados_semnull['ano_fabricacao_veiculo']!='    ']


# # This could be added just before calling `analisar_dataset` for "Acidentes Fatais"
# dados_fatais_encoded = encode_qualitative_to_numeric(dados_fatais_semvazio2, metricas_filtro)


analisar_dataset(dados_semvazio2, "Todos os Acidentes")
analisar_dataset(dados_graves_semvazio2, "Acidentes com Feridos Graves")
analisar_dataset(dados_fatais_semvazio2, "Acidentes Fatais")

generate_correlation_heatmap(dados_semvazio2, "Completo")
generate_correlation_heatmap(dados_graves_semvazio2, "Graves")
generate_correlation_heatmap(dados_fatais_semvazio2, "Fatais")