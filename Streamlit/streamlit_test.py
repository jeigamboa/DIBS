import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

# Set page config for dark mode and wide layout
st.set_page_config(page_title="streamlit test", layout="wide", initial_sidebar_state="expanded")

# Load CSS file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("styles/style.css")  # Adjust path as needed

# Import simulation model
import sys
sys.path.append('banksim')
from banksim import model

# Sidebar navigation
st.sidebar.title("streamlit test")
st.sidebar.markdown(" ")
st.sidebar.markdown("map")
st.sidebar.markdown("layout")
st.sidebar.markdown("model test")
st.sidebar.markdown("app")

# Branch data
branch_data = pd.read_csv('data/data_with_brgy_population_iat_area.csv')

loc_dict= {'Branch Name': branch_data['Branch'],
           'lat': branch_data['Latitude'], 
           'lon': branch_data['Longitude'],
           'base_mean_iat': branch_data['Mean_iat'],
           'branch area': branch_data['Area_sqm']}

chart_df = pd.DataFrame.from_dict(loc_dict)

x = np.linspace(0, 10, 100)
Apple = np.sin(x)
Banana = np.cos(x)
Mango = np.arctan(x)
Strawberry= np.tanh(x)

plt_dat = pd.DataFrame.from_dict({'x':x, 'Apple':Apple, 'Banana': Banana, 'Mango': Mango, 'Strawberry': Strawberry})

# Main layout
# Replace the previous markdown header with a local-image header
header_img_path = "header.png"  # <-- change filename if needed

# reduce gap and image width so text is closer to the image
header_col1, header_col2 = st.columns([0.8, 6], gap="small")
with header_col1:
    # display local image (logo / header image)
    st.image(header_img_path, width=80)  # reduced from 94 to 80
with header_col2:
    st.markdown(
        """
        <div class="app-header">
            <div style="font-size:1.8em;font-weight:700;color:#222;margin-bottom:4px;">Building a better Philippines — one family, one community at a time</div>
            <div style="font-size:1em;color:#666;margin-top:0;">Inspired by BPI's vision
        </div>
        """, unsafe_allow_html=True
    )

col1, col2 = st.columns([6, 4], gap="large")


with col1:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)
    st.subheader("Map")
    # Add color column: red for selected, blue for others
    # Default selection
    option = st.session_state.get("selected_branch", branch_data['Branch'].iloc[0])
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
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="light"))
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="right-panel">', unsafe_allow_html=True)
    option = st.selectbox(
        'BPI Katipunan Loyola Heights...',
        branch_data['Branch'],
        index=branch_data['Branch'].tolist().index(option) if option in branch_data['Branch'].tolist() else 0,
        key="selected_branch"
    )
    st.subheader("Line Graph")
    if option in plt_dat.columns:
        st.line_chart(plt_dat, x='x', y=option, x_label='x', y_label='y')
    else:
        st.warning(f"No data available for '{option}'. Please select a valid branch.")
    # --- Simulation Integration ---
    branch_row = branch_data[branch_data['Branch'] == option].iloc[0]

    customer_cap = int(np.ceil(branch_row['Area_sqm']*0.075 / 1)) #calculate customer capacity from branch data.
    #By default, 7.5% of the area is allocated for waiting area, and one customer occupies one square meter.
    exp = model.Experiment(base_mean_iat=branch_row['Mean_iat'],
                           customer_capacity=customer_cap)
    sim_results = model.single_run(exp)
    st.subheader("Simulation Results")
    st.markdown(
        f"""
        <div class="sim-results">
        <b>Mean Wait Time:</b> {sim_results['01_mean_wait_time']:.2f} mins<br>
        <b>Teller Utilization:</b> {sim_results['02_teller_util']:.2f}%<br>
        <b>Mean Outside Wait Time:</b> {sim_results['03_mean_outside_wait_time']:.2f} mins<br>
        <b>Long Teller Utilization:</b> {sim_results['04_long_teller_util']:.2f}%
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
