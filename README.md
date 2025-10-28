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

Download the models from: https://drive.google.com/file/d/1bxxh_YOIo9vW6cfVhfUBrKsIHGhZSqSZ/view?usp=drive_link

Alternatively, extract the folder `./SARMAD_MODEL` inside `SARMAD_MODEL.zip` manually to the same folder as `demo_app_enhanced.py`

The app will open in your browser at `http://localhost:8501`

## Required Packages

- streamlit
- pandas
- numpy
- gdown
- plotly

- xgboost



