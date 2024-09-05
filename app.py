import streamlit as st
import numpy as np
#import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
from datetime import datetime
with open('Flight_price_pred.pkl','rb') as file:
    model = pickle.load(file)

with open('ohe_encoder_airline.pkl','rb') as file:
    ohe_Airline  = pickle.load(file)

with open('ohe_encoder_destination.pkl','rb') as file:
    ohe_Destination = pickle.load(file)

with open('ohe_encoder_source.pkl','rb') as file:
    ohe_Source = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

st.title('Flight Price Prediction')


# User input

Airline = st.selectbox('Airline', ohe_Airline.categories_[0])
Source =  st.selectbox('Source', ohe_Source.categories_[0])
Destination = st.selectbox('Destination', ohe_Destination.categories_[0])
Total_Stops = st.slider('Total Stops', 0, 4)
journey_date = st.date_input("Select Journey Date")
journey_time = st.time_input("Select Journey Time")
journey_datetime = datetime.combine(journey_date, journey_time)
Journey_day = journey_date.day
Journey_month = journey_date.month
Dep_hour = journey_datetime.hour
Dep_min = journey_datetime.minute
arrival_time = st.time_input("Select Arrival Time")
Arrival_hour = arrival_time.hour
Arrival_min = arrival_time.minute

Duration_hours = abs(Arrival_hour - Dep_hour)
Duration_mins = abs(Arrival_min - Dep_min)

input_data = pd.DataFrame({
    'Total_Stops': [Total_Stops],
    'Journey_day': [Journey_day],
    'Journey_month': [Journey_month],
    'Dep_hour': [Dep_hour],
    'Dep_min': [Dep_min],
    'Arrival_hour': [Arrival_hour],
    'Arrival_min': [Arrival_min],
    'Duration_hours': [Duration_hours],
    'Duration_mins': [Duration_mins]
})

encoded_airline = ohe_Airline.transform(input_data[['Airline']])
encoded_airline_df = pd.DataFrame(encoded_airline.toarray(), columns=ohe_Airline.get_feature_names_out(['Airline']))

input_data.reset_index(drop=True, inplace=True)
encoded_airline_df.reset_index(drop=True, inplace=True)
input_data = pd.concat([input_data, encoded_airline_df], axis=1)

#source
encoded_source  = ohe_Source.transform(input_data[['Source']])
encoded_source_df = pd.DataFrame(encoded_source.toarray(), columns=ohe_Source.get_feature_names_out(['Source']))

input_data.reset_index(drop=True, inplace=True)
encoded_source_df.reset_index(drop=True, inplace=True)
input_data = pd.concat([input_data.drop('Source', axis=1), encoded_source_df], axis=1)

#destination
encoded_destination  = ohe_Destination.transform(input_data[['Destination']])
encoded_destination_df = pd.DataFrame(encoded_destination.toarray(), columns=ohe_Destination.get_feature_names_out(['Destination']))

input_data.reset_index(drop=True, inplace=True)
encoded_destination_df.reset_index(drop=True, inplace=True)
input_data = pd.concat([input_data.drop('Destination',axis=1), encoded_destination_df], axis=1)

input_data_scaled = scaler.transform(input_data)
prediction = model.predict(input_data_scaled)

st.write(f'Flight Price is Rs: {prediction}')