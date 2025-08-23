import streamlit as st
import pandas as pd
import numpy as np
from banksim.model import Experiment, single_run, single_run_n_days

st.title('Test Bank Simulator')

with st.sidebar:
    ###OPERATION VARIABLES
    n_operators = st.slider('Short Transaction Tellers', 1, 15, 2, step=1)
    n_long_operators = st.slider('Long Transaction Tellers', 1, 10, 2, step=1)
    WAITING_AREA = 10 #in sq.m . this is a test pa lang we should pribably get waiting_area allocated from bank area & ratio of floor area dedicated to waiting area
    
    ###Customer preferences and behavior
    customer_preferred_area = st.slider('Area per customer (sq. m)', 0.5, 2.5, 1.0, step=0.05)
    #mean_iat = st.slider('Mean Inter-Arrival Time (mins)', 1.0, 7.0, 2.0, step=0.05) #fix later
    #long_transact_mean_iat = st.slider('Long Transaction Mean Inter-Arrival Time (mins)', 45, 180, 75 , step=1,
    #                                   help='Set the probability that an incoming customer will do a long transaction.')

    customer_capacity = int(np.ceil(WAITING_AREA/customer_preferred_area))

user_experiment = Experiment(n_operators=n_operators,
                             n_long_operators=n_long_operators,
                             customer_capacity=customer_capacity)

if st.button("Run simulation for one day"):

    #  add a spinner and then display success box
    with st.spinner("Simulating ..."):
        # run multiple replications of experment
        results = single_run(user_experiment)

    st.success("Done!")

    st.write(results)



    
