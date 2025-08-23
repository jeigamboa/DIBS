import streamlit as st
import pandas as pd
import numpy as np
from banksim.model import Experiment, single_run

if "seed_gen" not in st.session_state:
    st.session_state.seed_gen = 0  # default


# Set page config for dark mode and wide layout
st.set_page_config(page_title="streamlit test", layout="wide", initial_sidebar_state="expanded")

# Load CSS file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("styles/style.css")  # Adjust path as needed

st.title('Single-Day Dynamic Branch Simulator')

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
    st.write('Operational Variables')
    n_operators = st.slider('Short Transaction Tellers', 1, 15, 2, step=1)
    n_long_operators = st.slider('Long Transaction Tellers', 1, 10, 2, step=1)
    ratio = st.slider('Waiting Area to Floor Area Ratio', 0.005, 0.5, 0.05, step=0.005)

    ###Customer preferences and behavior

    st.write('Customer Variables')
    customer_preferred_area = st.slider('Area per customer (sq. m)', 0.5, 2.5, 1.0, step=0.05)

    if branch_row['Mean_iat'] < 7.0:
        mean_iat = st.slider('Mean Inter-Arrival Time (mins)', 0.1, 7.0, branch_row['Mean_iat'], step=0.05, 
                            help='This adjusts the daily mean inter-arrival time of customers seeking to do short transactions.')
    else:
        mean_iat = st.slider('Mean Inter-Arrival Time (mins)', 0.1, branch_row['Mean_iat'] +5.0, branch_row['Mean_iat'], step=0.05, 
                            help='This adjusts the daily mean inter-arrival time of customers seeking to do short transactions.')
    #long_transact_mean_iat = st.slider('Long Transaction Mean Inter-Arrival Time (mins)', 45, 180, 75 , step=1,
    #                                   help='Set the probability that an incoming customer will do a long transaction.')
    customer_cap = int(np.ceil(branch_row['Area_sqm']*ratio / customer_preferred_area)) #calculate customer capacity from branch data.

    user_experiment = Experiment(n_operators=n_operators,
                                n_long_operators=n_long_operators,
                                customer_capacity=customer_cap, base_mean_iat = mean_iat)
    
    
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
    if st.button("Run simulation for one day") or RANDOM_:

        #  add a spinner and then display success box

        with st.spinner("Simulating ..."):
            # run multiple replications of experment
            
            results = single_run(user_experiment, rep=st.session_state.seed_gen)
            #st.write(f'Simulation seed: {st.session_state.seed_gen}') #was used for debugging
            

        s_toa = pd.to_timedelta(results['06_short_transact_time_of_arrival'], unit="m") + pd.Timestamp("09:00:00")
        short_toa = s_toa.strftime("%H:%M:%S")

        l_toa = pd.to_timedelta(results['10_long_transact_wait_times_time_of_arrival'], unit="m") + pd.Timestamp("09:00:00")
        long_toa = l_toa.strftime("%H:%M:%S")

        o_toa = pd.to_timedelta(results['08_outside_wait_times_time_of_arrival'], unit="m") + pd.Timestamp("09:00:00")
        out_toa = o_toa.strftime("%H:%M:%S")

        plt_dat_s = pd.DataFrame.from_dict({
            'Arrival time of customer': short_toa,
            'Waiting time (mins)': results['05_short_transact_wait_times']
            })


        plt_dat_l = pd.DataFrame.from_dict({
            'Arrival time of customer': long_toa,
            'Waiting time (mins)': results['09_long_transact_wait_times']
            })
        
        plt_dat_out = pd.DataFrame.from_dict({
            'Arrival time of customer': out_toa,
            'Waiting time outside branch (mins)': results['07_outside_wait_times']
        })

        st.subheader("Waiting times of customers performing short transactions")
        st.line_chart(plt_dat_s, x='Arrival time of customer', 
                        y='Waiting time (mins)')

        st.subheader('Waiting times of customers performing long transactions')
        st.line_chart(plt_dat_l, x='Arrival time of customer', 
                        y='Waiting time (mins)')
        
        find_gr_0 = np.where(np.array(results['07_outside_wait_times']) > 0.01)[0]
        if len(find_gr_0) > 0:
            st.write('The branch exceeded its physical capacity.')

            st.line_chart(plt_dat_out, x='Arrival time of customer', 
                        y='Waiting time outside branch (mins)')

        
        

    
