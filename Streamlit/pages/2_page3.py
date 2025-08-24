import streamlit as st
import pandas as pd
import numpy as np
from banksim.model import Experiment, month_run, daily_rand

if "seed_gen" not in st.session_state:
    st.session_state.seed_gen = 0  # default


# Set page config for dark mode and wide layout
st.set_page_config(page_title="streamlit test", layout="wide", initial_sidebar_state="expanded")

# Load CSS file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("styles/style.css")  # Adjust path as needed

st.title('Single-Month Dynamic Branch Simulator')

branch_data = pd.read_csv('data/data_with_brgy_population_iat_area.csv')
loc_dict= {'Branch Name': branch_data['Branch'],
           'lat': branch_data['Latitude'], 
           'lon': branch_data['Longitude'],
           'base_mean_iat': branch_data['Mean_iat'],
           'branch area': branch_data['Area_sqm']}

chart_df = pd.DataFrame.from_dict(loc_dict)

option = st.session_state.get("selected_branch", branch_data['Branch'].iloc[0])
chart_df['color'] = chart_df['Branch Name'].apply(
    lambda name: [255, 0, 0] if name == option else [0, 100, 255]
)
branch_row = branch_data[branch_data['Branch'] == option].iloc[0]

st.subheader('Branch')

option = st.selectbox(

        'Select a branch:',
        branch_data['Branch'],
        index=branch_data['Branch'].tolist().index(option) if option in branch_data['Branch'].tolist() else 0,
        key="selected_branch"
    )

st.divider()



col1, col2 = st.columns([3, 9], gap="large")
with col1:

    st.subheader('Simulation variables')
    ###OPERATION VARIABLES
    st.markdown('Operational Variables')
    n_operators = st.slider('Short Transaction Tellers', 1, 15, 2, step=1)
    n_long_operators = st.slider('Long Transaction Tellers', 1, 10, 2, step=1)
    ratio = st.slider('Waiting Area to Floor Area Ratio', 0.005, 1.0, 0.05, step=0.005)

    ###Customer preferences and behavior

    st.write('Customer Variables')
    customer_preferred_area = st.slider('Area per customer (sq. m)', 0.5, 2.5, 1.0, step=0.05)

    customer_cap = int(np.ceil(branch_row['Area_sqm']*ratio / customer_preferred_area)) #calculate customer capacity from branch data.

    st.write('Seasonality Variables')
    last_day_of_month = st.slider('Last day of Month', 28, 31, 30, step=1)
    force_max = st.slider('Max. Daily Mean IAT (mins)', 1.0, 30.0, 10.0, step = 0.1,
                          help='Highest daily mean inter-arrival time (IAT) of customers that wish to do short transactions. Corresponds to IAT on a "slow" day.')
    force_min = st.slider('Min. Daily Mean IAT (mins)', 1.0, 15.0, 3.0, step = 0.1,
                          help='Lowest daily mean inter-arrival time (IAT) of customers that want to do short transactions. Corresponds to IAT on a peak day.')
    rate = st.slider('Short Transaction IAT Decay Rate', 5.0, 30.0, 20.0, step = 1.0,
                     help = 'Rate of decrease of daily mean IAT, starting from peak days. Higher rate means that people are more likely to do short transactions on the peak days (15th, or 30th) than after peak days.')
    run_days = [k for k in range(last_day_of_month)]

    user_experiment = Experiment(n_operators=n_operators,
                                n_long_operators=n_long_operators,
                                customer_capacity=customer_cap)
    
    st.write('Randomness Control')

    RANDOM_ = st.button('Random Run')
    if RANDOM_: 
        st.session_state.seed_gen = np.random.randint(0, 101)

    st.session_state.seed_gen = st.number_input("Simulation seed:", 
                        min_value =0, 
                        max_value =100, 
                        value=st.session_state.seed_gen,
                        key="seed_gen_input", 
                        help='An integer from 0 to 100. If you wish to recreate a specific run, enter its seed number here.')

with col2:
    st.write("""Simulate branch operations for a month.
             Peak days for short transactions are usually on the 15th and 30th calendar day, so the IAT is smaller on these days.""")

    if st.button("Run simulation for one month") or RANDOM_:

        #  add a spinner and then display success box

        with st.spinner("Simulating ..."):
            # run multiple replications of experment
            results = month_run(user_experiment,
                                last_day_of_month=last_day_of_month,
                                force_max=force_max,
                                force_min=force_min,
                                rate=rate,
                                seed=st.session_state.seed_gen)
            #st.write(f'Simulation seed: {st.session_state.seed_gen}') #was used for debugging
            
        plt_days = np.array([k+1 for k in run_days])
        plt_dat= pd.DataFrame.from_dict({
            'Calendar day': plt_days,
            'Short Transactions (mins)': results['01_1month_mean_wait_time'],
            'Outside Waiting Time (mins)': results['03_1month_mean_outside_wait_time'],
            'Short Transaction Teller Utilization (%)': results['02_1month_teller_util'],
            'Long Transaction Teller Utilization (%)': results['04_1month_long_teller_util']
            })
        
        st.subheader('Daily mean waiting times over a month')
        st.line_chart(plt_dat, x='Calendar day', 
                y=['Short Transactions (mins)',
                   'Outside Waiting Time (mins)'])
        
        #st.subheader('Mean outside waiting times over a month')
        #st.line_chart(plt_dat, x='Calendar day', 
        #        y='Daily Mean Outside Waiting Time (mins)')
        
        st.subheader('Teller Utilization')
        st.line_chart(plt_dat, x='Calendar day', 
                y=['Short Transaction Teller Utilization (%)',
                   'Long Transaction Teller Utilization (%)'])
