import streamlit as st 
import pandas as pd 
import numpy as np 
import joblib


df =  pd.read_csv(r"car_ads_details_kaggle.csv")

st.set_page_config("AutoDrive AI" , layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(r"car_price.pkl")




@st.cache_resource
def load_meta():
    return joblib.load(r"meta.pkl")


model =  load_model()
data  = load_meta()  

st.title("Egyptian used car  for Price intelligence")

st.markdown('''

predict the fair value for car based on its features
** predict churn **
''')

model  = load_model()


st.sidebar.header("Car Info")


brands  =  df.Brand.unique().tolist()

brand  =  st.sidebar.selectbox("brand" ,brands,  key=15677)
modell  =  st.sidebar.text_input("model" , key = 1545)
kilometers =  st.sidebar.number_input("kilometers " , min_value= 0.  , value=10.,max_value= 10000000., key  =  455747)
year =  st.sidebar.number_input("year " , 1960. ,2026. , step = 1.,key  =  4322457)
fuel_type  =  st.sidebar.selectbox("fuel_type" ,df['Fuel Type'].unique().tolist(),  key=176577)
Transmission_Type  =  st.sidebar.selectbox("Transmission_Type" ,df["Transmission Type"].unique().tolist(),  key=156+77)
eng_cap =  st.sidebar.number_input("eng_cap " , float(df['Engine Capacity (CC)'].min()) ,float(df['Engine Capacity (CC)'].max()), step = 100.,key  =  4556+7)
body_type  =  st.sidebar.selectbox("body_type" ,df["Body Type"].unique().tolist(),  key=17547)

car_age = 2026 -  year
kilo_per = kilometers /  np.clip(car_age , 1 , 30000000)

asking_p =  st.sidebar.number_input("Asking Price" , min_value=0. , max_value= 2000000000. , step = 5000.,key = 100241)

# year =
# 

pred_button =  st.sidebar.button("analyze car")

if pred_button :
  df = pd.DataFrame({"brand" : [brand] , "model" : modell , 
                "kilo" : [kilometers] , "year": [year] , "fuel" : [fuel_type] , "trans": [Transmission_Type] 
                , "en": [eng_cap ], "bod" : [body_type] , "car": [car_age]  , "kil":[kilo_per]})
  df.columns = ['Brand',
 'Model',
 'Kilometers',
 'Year',
 'Fuel Type',
 'Transmission Type',
 'Engine Capacity (CC)',
 'Body Type',
 
 'CarAge',
 'KeloPerYear'] 
#   print(data["all_feature_names"].)
  print(df)
  preds = model.predict(df)[0]
  st.header("Valueation Result")
  col1  , col2  , col3 =  st.columns(3)
  with col1 :
     st.metric("Estimated value" , f"{preds} EGP")

  with col2 :
     dif =  asking_p -  preds
     st.metric("price Diferrence " , f"{dif} EGP")
  with col3 :
     prem =  (dif/preds) *100
     st.metric("price premuim" , prem)
     if prem > 10 :
        st.error("overprice")

     if prem < -10 :
        st.success("undervalue")