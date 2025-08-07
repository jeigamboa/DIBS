import streamlit as st
import pandas as pd
import numpy as np

test_ = "testing!"

st.write(test_)

option = st.selectbox(
    'Choose your BPI Branch:',
    ['Apple', 'Banana', 'Mango', 'Strawberry']
)

loc_dict= {'lat': [ 14.6401984, 14.6157534,14.6445752,14.6390686, 14.6236143,14.6366666, 14.6034851], 
           'lon': [ 121.0742878,  121.070107, 121.048706, 121.074222, 121.0539746, 121.0281416,  121.078829]}

chart_df = pd.DataFrame.from_dict(loc_dict)

st.map(chart_df,size=20, color="#0044ff")