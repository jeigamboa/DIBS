import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

test_ = "testing!"

st.write(test_)


loc_dict= {'Branch Name': ['Apple', 'Banana', 'Mango', 'Strawberry', 'Trial1', 'Trial2', 'Trial3'],
           'lat': [ 14.6401984, 14.6157534,14.6445752,14.6390686, 14.6236143,14.6366666, 14.6034851], 
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
    ['Apple', 'Banana', 'Mango', 'Strawberry', 'Trial1', 'Trial2', 'Trial3']
    )
    st.subheader("Line Graph")
    if option in plt_dat.columns:
        st.line_chart(plt_dat, x='x', y=option, x_label='x', y_label='y')
    else:
        st.warning(f"No data available for '{option}'. Please select a valid branch.")

    

with col1:
    st.subheader("Map")

    # Add color column: red for selected, blue for others
    chart_df['color'] = chart_df['Branch Name'].apply(
        lambda name: [255, 0, 0] if name == option else [0, 100, 255]
    )

    # Define pydeck layer
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=chart_df,
        get_position='[lon, lat]',
        get_color='color',
        get_radius=100,
        pickable=True,
    )

    # Set the map view centered on the selected branch
    selected_row = chart_df[chart_df['Branch Name'] == option].iloc[0]
    view_state = pdk.ViewState(
        latitude=selected_row['lat'],
        longitude=selected_row['lon'],
        zoom=14,
        pitch=0
    )

    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))