import streamlit as st
import pandas as pd
from src.model_io import load_model
from src.config_loader import load_config
from src.data_preprocessing import load_data
import os
from typing import Any,  Tuple, List

@st.cache_resource
def setup_app() -> Tuple[Any, pd.DataFrame, List[str]]:
    """
    Loads configuration, trained model and dataframe.
    Returns model, dataframe and target labels.
    """
    # Loading configuraton, default parameters: config.yaml
    config = load_config()

    # Loading the trained model, stored at the specified path
    model = load_model(config['saved_model_path'])
    
    # Loading data from our pandas DataFrame stored at the path specified in the YAML file
    data_path = config['raw_data_path']
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: run the pipeline before")
    
    wine_df = pd.read_csv(data_path)

    # Defining target groups names
    target_labels = ['Class_0', 'Class_1', 'Class_2']

    return model, wine_df, target_labels
    
# Initialize the setup
try:
    model, wine_df, target_labels = setup_app()
except  Exception as e:
    st.error(f"Error during setup: {e}")
    st.stop()

# Initializing the web page:
# Title and icon of the page, Layout wide means it uses the whole page and not only a central column
st.set_page_config(page_title="Wine Classifier AI", page_icon="🍷", layout="wide")

# BULIDNG USER INTERFACE
st.title("AI WINE CLASSIFIER")
st.markdown("""
"This application uses a **Machine Learning (Random Forest Classifier)** model to determine the class of a particular wine
based on its chemical analyisis.
""")

# Initializing the sidebar, header and explanation 
st.sidebar.header("Enter chemical analysis")
st.sidebar.info("Adjust the values using the sliders")

input_data = {}

# Removing target column from the DataFrame in order to show only the input data
feature_names = wine_df.drop(columns=['target']).columns

for feature in feature_names:
    # Calculating limits for each element to present a precise range of movement in the slider
    min_value = float(wine_df[feature].min())
    max_value = float(wine_df[feature].max())
    mean_value = float(wine_df[feature].mean())

    # Creating the slider for the current element
    input_data[feature] = st.sidebar.slider(
        label=feature.capitalize().replace("_", " "), # Capitalize() makes the first letter upper()
        min_value=min_value,
        max_value=max_value,
        value=mean_value
    )

# Converting input_data into pandas DataFrame so that our model can make the prediction
input_df = pd.DataFrame([input_data])

st.subheader("Wine data to Analyze:")
st.dataframe(input_df, hide_index=True)

# Initializing button to start analysis
if st.button("Start Analysis", type="primary", use_container_width=True):
    with st.spinner("Analyzing..."): # Loading spinner

        # Making the prediction: predict() returns a list even if we only have one element
        prediction_index = model.predict(input_df)[0] 

        # Calculating the feature name of the index found
        prediction_label = target_labels[prediction_index]

        st.success(f"### The model is confident this wine belongs to: **{prediction_label.upper()}**!")

        # Balloons animation
        st.balloons()

    

