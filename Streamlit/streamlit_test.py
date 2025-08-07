import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

#edit later with directory to lat long data for branches
branch_data = pd.read_csv('data/Compiled_Data.csv')

test_ = "testing!"

st.write(test_)


loc_dict= {'Branch Name': branch_data['BPI Branch'],
           'lat': branch_data['Latitude'], 
           'lon': branch_data['Longitude']}

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
    branch_data['BPI Branch']
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


