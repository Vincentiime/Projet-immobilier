import streamlit as st
import pandas as pd
import numpy as np
import folium
import pickle
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from streamlit_folium import st_folium
from streamlit_folium import folium_static
import xgboost as xgb
import os, json


APP_TITLE = "Prix immobilier et coûts de rénovation énergétique"
APP_SUBTITLE = 'Sources Ademe & Data.gouv'


@st.cache(allow_output_mutation=True)
def load_data(data):
    df = pd.read_csv(data)
    return df

def load_model(filepath):
        with open(filepath,'rb') as f:
            return json.load(f, decode('utf-8', errors='replace'))

#LOAD DATA
df1 = load_data("data3.csv")
df1_sampled = df1.sample(20000)

model = xgb.XGBRegressor()
model.load_model('model_xgb_save.json')
@st.cache(allow_output_mutation=True)


def convert_address(location, street, city):
    global latitude, longitude
    geolocator = Nominatim(user_agent="immo_app")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    location = geolocator.geocode(street+", "+city)
    if location is None:
        latitude is None , longitude is None
    else:
        latitude = location.latitude
        longitude = location.longitude
    return latitude, longitude 


def predictionprix(surface_reelle_bati, prix_surface, longitude,latitude, Appartement, Maison):
       
    data = {'surface_reelle_bati' : surface_reelle_bati,
            'prix_surface': prix_surface,
            'longitude': longitude,
            'latitude': latitude,
            'Appartement': Appartement,
            'Maison': Maison
            }
    prediction = model.predict(pd.DataFrame([[surface_reelle_bati, prix_surface, longitude, latitude, Appartement, Maison]], columns=['surface_reelle_bati', 'prix_surface', 'longitude', 'latitude', 'Appartement','Maison']))
    return prediction

def plotDot(point, this_map):
    folium.CircleMarker(location=[point.latitude, point.longitude],
                        radius=10,
                        weight=1,
                        popup=(point.valeur_fonciere, point.surface_reelle_bati, point.prix_surface),
                        fill_color='#000011').add_to(this_map)

def generate_map(df1_sampled, coordinates_gps):
    this_map = folium.Map(prefer_canvas=True, location = coordinates_gps, zoom_start =14, tiles ='cartodb positron')
    df1_sampled.apply(plotDot, axis = 1, args=[this_map])
    return this_map

valeur_fonciere = 0
surface_reelle_bati = 0
def prix_m2(valeur_fonciere, surface_reelle_bati):
    prix_surface = 0
    try:
        prix_surface = int(float(valeur_fonciere)) // int(float(surface_reelle_bati))
    except ValueError:
        pass


# Energetic price  coefficient of renovation :
lst1= [ 300, 350,400, 500]
lst2= [ 200, 250, 300, 400]
lst3= [ 100, 150, 200, 300]
coef_renovation =pd.DataFrame(list(zip(lst1, lst2, lst3)),
               columns =['G', 'F', 'E'], index =['D', 'C', 'B', 'A'])

#STREAMLIT APP

st.title(APP_TITLE)
st.text(APP_SUBTITLE)
st.write("Vous avez trouver un bien intéressant sur site de vente immobilière. Vous souhaitez connaitre la marge de négociation possible en obtenant son prix réel lié à sa décôte de classe énergétique.")

st.image('https://static.latribune.fr/full_width/427344/paris-immobilier-toits.jpg')

st.header("Les caractéristiques du bien convoité :")


valeur_fonciere =st.text_input('Prix du bien (€)', key = int)
    
surface_reelle_bati =st.slider('Surface habitable (m²)', 9, 1000, 9)
    
prix_surface = prix_m2(valeur_fonciere, surface_reelle_bati)
if surface_reelle_bati == 'None':
    pass
st.info("Entrer l'adresse française du bien souhaité:")
street = st.text_input("Numéro et nom de rue", " 75 rue de Clichy")
if street == None:
    pass
city = st.text_input("Ville", "Paris")
if city  == None:
    pass
    
choix = st.radio("Type de bien :", ["Maison", "Appartement"])
Appartement = 0
Maison = 0
if choix == "Appartement":
    Appartement == 1 , Maison == 0
else:
    choix == "Maison"
    Appartement == 0 , Maison == 1
    

code_departement =st.select_slider('Code département:',['1','2','2A','2B','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','21','22','23','24','25','26','27','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','58','59','60','61','62','63','64','65','66','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95'])

classe_energetique = st.radio('Classe énergétique',('A','B','C','D','E', 'F', 'G'))
if classe_energetique == 'G' or classe_energetique =='F'or classe_energetique =='E':
    classe_energie_souhait = st.radio("Classe énergétique souhaitée",('A','B','C','D'))
    if classe_energetique == 'G':
        if classe_energie_souhait == 'A':
            cost = coef_renovation.iloc[3][0]*surface_reelle_bati
            st.write(f' Les coûts de travaux de rénovation estimés sont de : {cost} €') #TODO : suppress this line and duplicates to add at the bottom
        elif classe_energie_souhait == 'B':
            cost = coef_renovation.iloc[2][0]*surface_reelle_bati
            st.write(f' Les coûts de travaux de rénovation estimés sont de : {cost} €')
        elif classe_energie_souhait == 'C':
            cost = coef_renovation.iloc[1][0]*surface_reelle_bati
            st.write(f' Les coûts de travaux de rénovation estimés sont de : {cost} €')
        elif classe_energie_souhait == 'D':
            cost = coef_renovation.iloc[0][0]*surface_reelle_bati
            st.write(f' Les coûts de travaux de rénovation estimés sont de : {cost} €')
    if classe_energetique == 'F':
        if classe_energie_souhait == 'A':
            cost = coef_renovation.iloc[3][1]*surface_reelle_bati
            st.write(f' Les coûts de travaux de rénovation estimés sont de : {cost} €')
        elif classe_energie_souhait == 'B':
            cost = coef_renovation.iloc[2][1]*surface_reelle_bati
            st.write(f' Les coûts de travaux de rénovation estimés sont de : {cost} €')
        elif classe_energie_souhait == 'C':
            cost = coef_renovation.iloc[1][1]*surface_reelle_bati
            st.write(f' Les coûts de travaux de rénovation estimés sont de : {cost} €')
        elif classe_energie_souhait == 'D':
            cost = coef_renovation.iloc[0][1]*surface_reelle_bati
            st.write(f' Les coûts de travaux de rénovation estimés sont de : {cost} €')
    if classe_energetique == 'E':
        if classe_energie_souhait == 'A':
            cost = coef_renovation.iloc[3][2]*surface_reelle_bati
            st.write(f' Les coûts de travaux de rénovation estimés sont de : {cost} €')
        elif classe_energie_souhait == 'B':
            cost = coef_renovation.iloc[2][2]*surface_reelle_bati
            st.write(f' Les coûts de travaux de rénovation estimés sont de : {cost} €')
        elif classe_energie_souhait == 'C':
            cost = coef_renovation.iloc[1][2]*surface_reelle_bati
            st.write(f' Les coûts de travaux de rénovation estimés sont de : {cost} €')
        elif classe_energie_souhait == 'D':
            cost = coef_renovation.iloc[0][2]*surface_reelle_bati
            st.write(f' Les coûts de travaux de rénovation estimés sont de : {cost} €')
else:
    st.write("Les travaux de rénovation ne sont pas obligatoires")




location = street + ' '+ city
coordinates_gps = convert_address(location, street, city)
if location == None:
    st.write("Aucune adresse encore renseignée")
else:
    st.write(f"Les coordonnées gps de l'adresse renseignée sont : {coordinates_gps}")
    
#Price estimator
if st.button('Estimer le prix du bien'):
    
    prix_estime = int(predictionprix(surface_reelle_bati, prix_surface, coordinates_gps[1],coordinates_gps[0], Appartement, Maison))/10
    
    dif_prixmarche = int(abs(prix_estime - int(valeur_fonciere)))
    

    st.write(f'Le prix estimé du bien selon son voisinage est de {prix_estime} €.')
    if int(prix_estime) > int(valeur_fonciere):
        st.write(f'Soit une différence de {dif_prixmarche} € par rapport au prix demandé. La négociation semble difficile.')
    if int(prix_estime) < int(valeur_fonciere):
        st.write(f'Soit une différence de {dif_prixmarche} € par rapport au prix demandé. Une négociation est à envisager.')

    this_map = folium.Map(prefer_canvas=True, location = coordinates_gps, zoom_start =14, tiles ='cartodb positron')
    
#MORE INFO OF LAST YEAR REAL ESTATE SALES:    
st.header("Ventes immobilières de l'année 2021")
st.write("Légende indiquant respectivement le prix de la vente, la surface habitable, le prix au m²")
    
m = generate_map(df1_sampled, coordinates_gps)
folium_static(m, width=700)


       



    