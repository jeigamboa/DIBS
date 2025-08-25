## DIBS: Dynamic and Informed Branch Simulator (DIBS): Improving Branch Location and Layout through Simulation and AI-powered Insights

Project submitted to the BPI DATA Wave 2025 Hackathon, powered by Eskwelabs, by (team) PISIKA, consisting of the ff. authors:
- David Daffon (SanD-CMPRG),
- Jei Gamboa (IPL-Team One),
- Mitch Kong (IPL-Team One),
- Paul Remo (Theory Group-QuantMathPhys), 

all of whom were undergraduate researchers at the National Institute of Physics, UP Diliman, at the time of the Hackathon.

### Contents

* `RAG Ingest` - contains `csvs` and `documents` for input of pertinent documents for RAG usability. Also contains `ingest.py` which is the script for ingesting the documents or csvs into the supabase database.

* `Streamlit` - contains the prototype. `banksim` contains the simulation model as `model.py`, `data` contains both simulated and scraped data, and `pages` contain the streamlit pages. `5_Ask_DIBS` contains the agentic RAG chatbot.

* `.env.sample` - template `.env` for containing API keys for both OpenAI and supabase's APIs, etc. Primarily for our safety so that we can remove our API keys.

### How to run

Python version 3.12 is recommended. Libraries and packages are listed in `gen_requirements.txt` and `RAG_requirements.txt`. For ease, kindly run:

``` pip install -r gen_requirements.txt``` and
```pip install -r RAG_requirements.txt```

Upon installation, since we have removed our API keys for safety, please use an your own API keys for supabase and OpenAI and save them in `.env.sample` as `.env`.

Please don't forget to update ``file_path`` in `ingest.py` and `5_Ask_DIBS.py`.

### Acknowledgments

* Various free online tutorials: 
* BPI/Eskwelabs mentors for their domain expertise and advice.


