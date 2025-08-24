## Team PISIKA's DIBS: Dynamic and Informed Branch Simulator (DIBS): Improving Branch Location and Layout through Simulation and AI-powered Insights

Project submitted to the BPI DATA Wave 2025 Hackathon, powered by Eskwelabs, by the ff. authors:
- David Daffon,
- Jei Gamboa,
- Mitch Kong, and
- Paul Remo,
all of whom are affiliated with the National Institute of Physics at the time of the Hackathon.

Contents:
- gen_AI = directory for all gen_AI APIs and capabilities
- Streamlit = directory for project prototyping
- Hello world.ipynb = initializing

How to run .py file:

set up anaconda navigator -> set up virtual environment -> run using terminal

=== Setting up Anaconda Navigator ===
1. Download and install anaconda navigator from https://www.anaconda.com/products/navigator


=== Setting up virtual environment ===
1. Open Anaconda Navigator
2. Click Environments
3. Click Create > Pick a name for new virtual environment (env_name) > Check Python 
4. Open Anaconda Prompt (Can search for this in the start bar)
5. Enter "conda activate env_name" (w/o quotes)
6. Enter "conda install conda-forge::package_names". specifically, install:
streamlit
7. Wait and type "y" if prompted with [Y/N]

=== Run using terminal ===
1. Open Anaconda prompt
2. Activate env with dash-leaflet by typing "conda activate env_name"
3. Navigate to folder where the .py file is located by typing "cd file/path/files"
4. When in the folder where the .py file is, type "python Script_name.py"
5. Type "streamlit run Script_name.py"