# Importing preliminaries
import chromadb
import os
import pandas as pd

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initializing OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

data_path = r"data"
chroma_path = r"chroma_db"

chroma_client = chromadb.PersistentClient(path=chroma_path)

collection = chroma_client.get_or_create_collection(name="bpi_info")

user_query = input("Hello Branch Manager, how can I help you? \n")

results = collection.query(
    query_texts=[user_query],
    n_results=4
)

print(results['documents'])

branch_loc_data = pd.read_csv('../Streamlit/data/Compiled_Data.csv') 
brgy_pop_data = pd.read_csv('../Streamlit/data/barangay_population.csv')

system_prompt = '''You are an assistant for managerial-level up to executive level bankers. 
         Your job is to deliver accurate and concise insights to business queries. If a query is irrelevant to
         the database, abort generation and politely indicate the irrelevance.
         Refer to the data: '''+ str(results["documents"]) +''' '''

irrelevant_prompt = '''Irrelevant query such as: [What is the capital of X country?], 
         [Any questions relating to the user personally.]'''    

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    store=True,
    max_tokens=100,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": irrelevant_prompt},
        {"role": "assistant", "content": "Our apologies, your query does not seem to be relevant."},
        {"role": "user", "content": f"Refer to {branch_loc_data}, {brgy_pop_data}. Finish the answer within 100 tokens. {user_query}"}
    ],
    temperature=0
)

print(completion.choices[0].message.content)