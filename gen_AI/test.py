# Preliminaries
import numpy as np
import pandas as pd

# Importing openai library and dotenv to store secret key
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Initializing client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Ingesting files, change filepath to correct location before running code
filepath = 'C:/David/000 Work Prep and Independent Proj/DIBS/Streamlit/data'

branch_loc_data = pd.read_csv(filepath + '/Compiled_Data.csv') 
brgy_pop_data = pd.read_csv(filepath + '/barangay_population.csv')

prompt = input()

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    store=True,
    max_tokens=100,
    messages=[
        {"role": "system", "content": '''You are an assistant for managerial-level up to executive level bankers. 
         Your job is to deliver accurate and concise insights to business queries. If a query is irrelevant to
         the database, abort generation and politely indicate the irrelevance.'''},
        {"role": "user", "content": '''Irrelevant query such as: [What is the capital of X country?], 
         [Any questions relating to the user personally.]'''},
        {"role": "assistant", "content": "Our apologies, your query does not seem to be relevant."},
        {"role": "user", "content": f"Refer to {branch_loc_data}, {brgy_pop_data}. Finish the answer within 100 tokens. {prompt}"}
    ],
    temperature=0
)

print(completion.choices[0].message.content)