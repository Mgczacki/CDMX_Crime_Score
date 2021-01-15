import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import requests
import urllib.parse
import json
import numpy as np
import pickle
#PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
#Annoy
from annoy import AnnoyIndex
index = AnnoyIndex(3,'euclidean')
index.load('t_pred.tree')
#Parametros
neighbors = 10
days = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Red Neuronal
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(14, 26)
        self.fc2 = nn.Linear(26, 28)
        self.fc3 = nn.Linear(28, 18)
        self.fc4 = nn.Linear(18, 15)
        self.fc5 = nn.Linear(15, 15)
        self.fc6 = nn.Linear(15, 12)
        self.fc7 = nn.Linear(12, 12)
        self.fc8 = nn.Linear(12, 10)
        self.fc9 = nn.Linear(10, 6)
        self.fc10 = nn.Linear(6, 6)
        self.fc11 = nn.Linear(6, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.relu(self.fc10(x))
        x = torch.sigmoid(self.fc11(x))
        return x
net = Net()
net = torch.load("best-model.pt").to(device)
#Función de transformación de coordenadas
def transformacion_coordenadas(row):
    lon = row["longitud"] * np.pi/180
    lat = row["latitud"] * np.pi/180
    r = 6371
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.array([x, y, z])

f = open("diccionario_fechas", "rb")
diccionario_fechas = pickle.load(f)
f.close()

def generate_score(coords):
    datos = []
    vecinos_cercanos = neighbors
    l_v, l_d = index.get_nns_by_vector(coords, vecinos_cercanos, include_distances=True)
    for i in range(vecinos_cercanos):
        datos.append(l_d[i])
        datos.append((diccionario_fechas[-1] - diccionario_fechas[l_v[i]]).days/days)
    print(datos)
    datos = torch.tensor(np.array(datos).reshape(1, -1)).double().to(device)
    net.eval()
    with torch.no_grad():
        return net(datos).item()
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

token_map = open(".mapbox_token").read()
token_google = open(".google_token").read()

px.set_mapbox_access_token(token_map)

#Cargando el diccionario de fechas anexo al indice predictivo

#Columnas del dataframe
columnas = ['lat', 'lon', 'score', 'size', 'query']
#Datos del dataframe
lista_direcciones = [[0.0, 0.0, 0.0, 0.0, ""], [0.0, 0.0, 1.0, 0.0, ""]]
#Dataframe inicial
df = pd.DataFrame(lista_direcciones, columns = columnas)
def generate_map(df):
    return px.scatter_mapbox(df,
                            lat="lat", 
                            lon="lon", 
                            size='size',
                            color='score',
                            text = 'query',
                            center=dict(lon=-99.1332,lat = 19.4326),
                            color_continuous_scale=px.colors.sequential.Jet, 
                            size_max=15,
                            zoom=10)

fig = generate_map(df)


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

app.layout = html.Div(children=[
    html.P(id='placeholder'),
    html.H1(children='Mexico City Crime Score'),
    html.Div(children='''
        A simple model for predicting a location's crime score using the 10 nearest crimes in the month.
    '''),
    html.Div(dcc.Input(id='input-on-submit', type='text', placeholder="Input a location.")),
    html.Button('Submit', id='submit-val', n_clicks=0),
    html.Div(id='container-button-basic',
             children='Enter a position and press submit.'),
    dcc.Graph(
        id='example-graph',
        figure=fig
    ),
    html.A(html.Button('Clean Map'),href='/')
])

@app.callback(
    dash.dependencies.Output('placeholder', 'hidden'),
    [dash.dependencies.Input('input-on-submit', 'value')]
    )
def update_query_text(inp):
    global input_query
    input_query = inp
    return True
@app.callback(
    dash.dependencies.Output('example-graph', 'figure'),
    [dash.dependencies.Input('submit-val', 'n_clicks')]
    )
def update_output(n_clicks):
    new_df = df
    try:
        parsed = urllib.parse.quote(input_query)
        direc = 'https://maps.googleapis.com/maps/api/geocode/json?address='+parsed+'&key='+token_google
        response = requests.get(direc)
        localizacion = json.loads(response.text)
        q = localizacion["results"][0]['geometry']['location']
        lat = q['lat']
        lon = q['lng']
        coords = transformacion_coordenadas({"longitud":lon, "latitud":lat})
        score = generate_score(coords)
        print(score)
        lista_direcciones.append([lat, lon, score, 15.0, input_query])
        new_df = pd.DataFrame(lista_direcciones, columns = columnas)
    except:
        pass
    return generate_map(new_df)


if __name__ == '__main__':
    app.run_server(debug=True)
