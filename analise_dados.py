import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import folium
import os
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import geopandas as gpd

subpastas = ["Completo", "Fatais", "Graves"]  # Nomes de suas subpastas
for subpasta in subpastas:
    os.makedirs(subpasta, exist_ok=True)

###
### Arquivos geojson disponiveis em https://github.com/tbrugz/geodata-br
### Arquivos shapefile disponiveis em https://forest-gis.com/download-de-shapefiles/ FOREST-GIS Avulsos

# Configurações para Seaborn
sns.set_theme(style="whitegrid")
palette = sns.color_palette("pastel")

# Carregar o dataset
dados = pd.read_csv('acidentes2015.csv', delimiter=';', decimal=',', encoding='ISO-8859-1')

# Corrigindo dados ausentes na coluna 'marca'
#dados_corrigidos = dados['marca'].fillna('Não Informado', inplace=True)

# Filtre os dados para os estados mencionados e classificação de acidentes graves e fatais
estados = ['SC', 'PR', 'RS']

#Filtrando por Estado
dados_estados = dados[dados['uf'].isin(['SC', 'PR', 'RS'])].copy()

#Filtrando por estado e classificacao fatal
dados_fatais = dados[(dados['uf'].isin(['SC', 'PR', 'RS'])) & (dados['estado_fisico'] == 'Morto       ')].copy()
# Agrupando por ID
#dados_fatais_agrupados = dados_fatais.groupby('id').first().reset_index()

#filtrando por estado e feridos graves
dados_graves = dados[(dados['uf'].isin(['SC', 'PR', 'RS'])) & (dados['estado_fisico'] == 'Ferido Grave')].copy()
# Agrupando por ID
dados_graves_agrupados = dados_graves.groupby('id').first().reset_index()


# Função para criar gráficos de barras

def criar_grafico(data, coluna, titulo, subpasta):
    if data.empty:
        print(f"No data available for {coluna}. Skipping plot.")
        return
    if data[coluna].nunique() == 0:
        print(f"No unique values in column {coluna}. Skipping plot.")
        return
    plt.figure(figsize=(12, 6))
    
    if coluna == 'ano_fabricacao_veiculo':
        y = data[coluna].value_counts().index
        x = data[coluna].value_counts().values
        plt.barh(y, x, align='center')
        plt.gca().invert_yaxis()  # Invert y-axis to have the oldest cars on top
        
    elif coluna == 'causa_acidente':
        sizes = data[coluna].value_counts().values
        labels = data[coluna].value_counts().index
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.3))
        
    elif coluna == 'tipo_veiculo':
        # Assuming you would like a radial column (bar in polar coordinates)
        values = data[coluna].value_counts().values
        categories = data[coluna].value_counts().index
        N = len(categories)
        theta = range(N)
        plt.bar(theta, values)
        plt.xticks(ticks=theta, labels=categories, rotation=45)
        
    elif coluna == 'condicao_metereologica' and 'Desconhecido' in data[coluna].values:
        data = data[data[coluna] != 'Desconhecido']
        sns.countplot(data=data, x=coluna, order=data[coluna].value_counts().index, palette=palette)
        
    else:
        sns.countplot(data=data, x=coluna, order=data[coluna].value_counts().index, palette=palette)
    
    plt.title(titulo)
    plt.tight_layout()
    nome_arquivo = f"{subpasta}/{titulo}.png"
    plt.savefig(nome_arquivo)
    plt.close()

# Realize a análise exploratória:
#    ('marca', 'Acidentes por Marca de Veículo'),
#    ('horario', 'Acidentes por Horário')
metricas = [
    ('condicao_metereologica', 'Acidentes por Condição Meteorológica'),
    ('tipo_pista', 'Acidentes por Tipo de Pista'),
    ('tipo_veiculo', 'Acidentes por Tipo de Veículo'),
    ('fase_dia', 'Acidentes por Fase do Dia'),
    ('causa_acidente', 'Acidentes por Causa'),
    ('ano_fabricacao_veiculo', 'Acidentes por Ano de Fabricação'),
    ('dia_semana', 'Acidentes por Dia da Semana'),
]


# Para o mapa da região sul:
# Carregar o ShapeFile das rodovias
# Inicializar o mapa centrado na região sul
m = folium.Map([-27.5953, -53.4958], zoom_start=6)

# Adicionar rodovias ao mapa
def get_municipality_centroids(shapefile_path):
    # Read the shapefile with geopandas
    gdf = gpd.read_file(shapefile_path)

    # Calculate the centroid for each municipality
    gdf['centroid'] = gdf['geometry'].centroid

    # Extract the latitude and longitude
    gdf['latitude'] = gdf['centroid'].apply(lambda p: p.y)
    gdf['longitude'] = gdf['centroid'].apply(lambda p: p.x)

    return gdf[['name', 'latitude', 'longitude']]
# Como não temos coordenadas exatas para acidentes, não podemos adicionar pontos exatos. 
# No entanto, uma abordagem possível seria usar os GeoJSONs dos municípios para destacar aqueles com mais acidentes.

def analisar_dataset(dataset, label, municipality_data):
    if label == "Todos os Acidentes":
        salvapasta = "Completo"
    elif label == "Acidentes Fatais":
        salvapasta = "Fatais"
    elif label == "Acidentes com Feridos Graves":
        salvapasta = "Graves"
    else:
        salvapasta = input("Digite o nome da subpasta para salvar os gráficos: ")
        
    # Verifique se a subpasta existe, caso contrário, crie-a
    if not os.path.exists(salvapasta):
        os.makedirs(salvapasta) 

    for coluna, titulo in metricas:
        # Teste para identificar o erro no plot
        dataset[coluna].fillna('Desconhecido', inplace=True)
        dataset.dropna(subset=[coluna], inplace=True)
        criar_grafico(dataset, coluna, f"{titulo} ({label})", salvapasta)
    
    # Criando o mapa com os dados do conjunto específico

    m = folium.Map([-27.5953, -53.4958], zoom_start=6)
#    folium.GeoJson(rodovias).add_to(m)
    for estado in estados:
        geojson_municipios = gpd.read_file(f'geojson/geojs-{estado}-mun.json')
        municipios_com_acidentes = dataset[dataset['uf'] == estado]['municipio'].value_counts()
        # Normalizar os dados para ter um valor entre 0 e 1
        maximo = municipios_com_acidentes.max()
        municipios_com_acidentes = municipios_com_acidentes / maximo
        folium.Choropleth(
            geo_data=geojson_municipios,
            name=f'Acidentes em {estado} ({label})',
            data=municipios_com_acidentes,
            columns=['municipio', 'count'],
            key_on='feature.properties.name',
            fill_color='YlOrRd',
            fill_opacity=0.5,
            line_opacity=0.1,
        ).add_to(m)
    m.save(f'{salvapasta}/mapa_acidentes_{label}.html')

    # Merge dataset with municipality_data on municipality name
    merged_data = dataset.merge(municipality_data, left_on='municipio', right_on='name', how='left')

    n = folium.Map([-27.5953, -53.4958], zoom_start=6)

    for index, row in merged_data.iterrows():
        # Check if latitude and longitude are not NaN before plotting
        if not pd.isnull(row['latitude']) and not pd.isnull(row['longitude']):
            folium.Circle(
                location=[row['latitude'], row['longitude']],
                popup=row['municipio'],
                radius=float(row['numero_acidentes']) * 10,  # O multiplicador 10 pode ser ajustado conforme necessário
                color='blue',
                fill=True,
                fill_color='blue'
            ).add_to(n)

    n.save(f'{salvapasta}/bubble_map_{label}.html')


# Análise para cada conjunto de dados
municipality_data = get_municipality_centroids('Shapefiles/ST_DNIT_Rodovias_SNV2015_03.shp')

analisar_dataset(dados_estados, "Todos os Acidentes", municipality_data)
analisar_dataset(dados_fatais, "Acidentes Fatais", municipality_data)
analisar_dataset(dados_graves, "Acidentes com Feridos Graves", municipality_data)

