import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pickle import load

st.title("Welcome to Diamond 💎 Price Prediction Website")
st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://wallpaperaccess.com/full/2464096.jpg")
    }
   .sidebar .sidebar-content {
        background: url("url_goes_here")
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.image("https://www.thermofisher.com/blog/mining/wp-content/uploads/sites/3/2018/04/diamond.jpg",width=700)


df = pd.read_csv("data/diamonds.csv")

# Dataset Details

st.title('Diamond Dataset')

dataset = st.radio('show dataframe', ('Top 5', 'columns'))

if dataset == 'Top 5':
    st.write(df.head(5))
else:
    st.write(set(df.columns))


# Date Description

st.title('Statistical Analysis')

describe = st.checkbox(label='Data Description')

if describe:
    st.write('Data Description')
    st.write(df.describe())

# Loading pretrained models from pickle file

or_enc = load(open('models/ordinal_encoder.pkl', 'rb'))
scaler = load(open('models/standard_scaler.pkl', 'rb'))
rf_model = load(open('models/rf_model.pkl', 'rb'))

st.title('💎 Diamond Price Prediction 💎')
carat = st.number_input("Enter the carat of the diamond: ", min_value=0.2, max_value=5.01)
cut = st.number_input("Enter the cut details of the diamond:" , min_value= 1, max_value=5)
color = st.number_input("Enter the color details of the diamond:" , min_value= 1, max_value= 7)
clarity = st.number_input("Enter the clarity details of the diamond:" , min_value= 1, max_value= 8)

depth = st.number_input("enter the depth of the diamond: ", min_value= 43, max_value= 79)

table = st.number_input("Enter the table dimension of the diamond: ", min_value= 43, max_value= 95)

x = st.number_input("Enter the x dimension of the diamond in mm: ", min_value= 0.0, max_value= 10.74)

y = st.number_input("Enter the y dimensions of the diamond in mm: ", min_value= 0.0, max_value= 58.9)

z = st.number_input("Enter the z dimensions of the diamond in mm: ", min_value= 0.0, max_value= 31.8)
click = st.button("Predict")

if click:
    if carat and cut and color and clarity and depth and table and x and y and z:
        num = pd.DataFrame({'carat': [carat], 'depth': [depth], 'table': [table], 'x': [x], 'y': [y], 'z': [z]})
        catg = pd.DataFrame({'cut': [cut], 'color': [color], 'clarity': [clarity]})

        rescaled_cat = or_enc.transform(catg)
        rescaled_num = scaler.transform(num)

        query_point = pd.concat([pd.DataFrame(rescaled_cat), pd.DataFrame(rescaled_num)], axis=1)
        price = rf_model.predict(query_point)

        st.success(f"The price of Selected Diamond is $ {round(price[0], 2)}")
        st.balloons()
    else:
        st.error('Enter the correct the values to get the predictions')
        st.snow()

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://wallpaperaccess.com/full/2464096.jpg")
    }
   .sidebar .sidebar-content {
        background: url("url_goes_here")
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.image("https://www.thermofisher.com/blog/mining/wp-content/uploads/sites/3/2018/04/diamond.jpg",width=700)
