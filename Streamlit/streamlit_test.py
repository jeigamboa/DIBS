import streamlit as st
import pandas as pd
import numpy as np

test_ = "testing!"

st.write(test_)



loc_dict= {'lat': [ 14.6401984, 14.6157534,14.6445752,14.6390686, 14.6236143,14.6366666, 14.6034851], 
           'lon': [ 121.0742878,  121.070107, 121.048706, 121.074222, 121.0539746, 121.0281416,  121.078829]}

chart_df = pd.DataFrame.from_dict(loc_dict)

x = np.linspace(0, 10, 100)
Apple = np.sin(x)
Banana = np.cos(x)
Mango = np.arctan(x)
Strawberry= np.tanh(x)

plt_dat = pd.DataFrame.from_dict({'x':x, 'Apple':Apple, 'Banana': Banana, 'Mango': Mango, 'Strawberry': Strawberry})
col1, col2 = st.columns([6, 4])

with col2:
    option = st.selectbox(
    'Choose your BPI Branch:',
    ['Apple', 'Banana', 'Mango', 'Strawberry', 'Trial1', 'Trial2']
    )
    st.subheader("Line Graph")
    if option in plt_dat.columns:
        st.line_chart(plt_dat, x='x', y=option, x_label='x', y_label='y')
    else:
        st.warning(f"No data available for '{option}'. Please select a valid branch.")

    

with col1:
    st.subheader("Map")
    st.map(chart_df)