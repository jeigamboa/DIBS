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

# Branch data
branch_data = pd.read_csv('data/data_with_brgy_population_iat_area.csv')

loc_dict= {'Branch Name': branch_data['Branch'],
           'lat': branch_data['Latitude'], 
           'lon': branch_data['Longitude'],
           'base_mean_iat': branch_data['Mean_iat'],
           'branch area': branch_data['Area_sqm']}

chart_df = pd.DataFrame.from_dict(loc_dict)

# Main layout
# Replace the previous markdown header with a local-image headerx
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

st.divider()

col1, col2 = st.columns([6, 4], gap="large")

with col1:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)
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

    ###Selection box
    st.subheader('Branch')
    
    st.markdown('<div class="right-panel">', unsafe_allow_html=True)
    
    option = st.selectbox(

        'Select a branch:',
        branch_data['Branch'],
        index=branch_data['Branch'].tolist().index(option) if option in branch_data['Branch'].tolist() else 0,
        key="selected_branch"
    )
    st.divider()
    

    ###Run simulation of default
    branch_row = branch_data[branch_data['Branch'] == option].iloc[0]


    customer_cap = int(np.ceil(branch_row['Area_sqm']*0.075 / 1)) #calculate customer capacity from branch data.
    #By default, 7.5% of the area is allocated for waiting area, and one customer occupies one square meter.
    exp = model.Experiment(base_mean_iat=branch_row['Mean_iat'],
                           customer_capacity=customer_cap)
    sim_results = model.single_run(exp)
    
    
    st.subheader("Single-day Simulation Summary")

    st.markdown(
        f"""
        <div class="sim-results">
        <b>Mean Waiting Time:  </b> <br>
        <font size=12>{sim_results['01_mean_wait_time']:.2f} mins</font><br>
        </div>
        """, unsafe_allow_html=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

col3, col4, col5 = st.columns([3, 3, 4], gap="large")
with col3:
    st.markdown(f"""<b> Short Transaction Teller Utilization: </b> <br>
                <font size=12> {sim_results['02_teller_util']:.2f}%</font>
                """, unsafe_allow_html=True)
    
with col4:
    st.markdown(f"""<b> Long Transaction Teller Utilization: </b> <br>
                <font size=12> {sim_results['04_long_teller_util']:.2f}%</font>
                """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""<b> Mean Outside Waiting Time: </b> <br>
                <font size=12> {sim_results['03_mean_outside_wait_time']:.2f} mins</font>
                """, unsafe_allow_html=True)    

st.divider()
###plot line chart

s_toa = pd.to_timedelta(sim_results['06_short_transact_time_of_arrival'], unit="m") + pd.Timestamp("09:00:00")
short_toa = s_toa.strftime("%H:%M:%S")

plt_dat_s = pd.DataFrame.from_dict({
    'Arrival time of customer': short_toa,
    'Waiting time (mins)': sim_results['05_short_transact_wait_times']
    })

l_toa = pd.to_timedelta(sim_results['10_long_transact_wait_times_time_of_arrival'], unit="m") + pd.Timestamp("09:00:00")
long_toa = l_toa.strftime("%H:%M:%S")

plt_dat_l = pd.DataFrame.from_dict({
    'Arrival time of customer':l_toa,
    'Waiting time (mins)': sim_results['09_long_transact_wait_times']
    })

st.subheader("Waiting times of customers performing short transactions")
st.line_chart(plt_dat_s, x='Arrival time of customer', 
                y='Waiting time (mins)')

st.subheader('Waiting times of customers performing long transactions')
st.line_chart(plt_dat_l, x='Arrival time of customer', 
                y='Waiting time (mins)')
    
