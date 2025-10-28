# SARMAD Demo

A biological age prediction demo using explainable AI

## Installation

Install required packages:
```bash
pip install -r requirements.txt
```

## Run Demo

Run the enhanced demo app:
```bash
streamlit run demo_app_enhanced.py
```

Notice: if the models folder `./SARMAD_MODEL` is missing, the demo will download it automatically. 
Model's size is 16GB!
Altenatively, extract `SARMAD_MODEL` manually in the same folder as `demo_app_enhanced.py`

The app will open in your browser at `http://localhost:8501`

## Required Packages

- streamlit
- pandas
- numpy
- gdown
- plotly

- xgboost
