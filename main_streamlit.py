import requests
import pandas
import pickle
import gzip
import os
import pandas as pd
from openai.embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    tsne_components_from_embeddings,
    chart_from_components,
    indices_of_nearest_neighbors_from_distances,
)
import openai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()

from config import *

import streamlit as st

# constants
EMBEDDING_MODEL = "text-embedding-ada-002"

openai.api_key = os.environ.get('OPENAI_API_KEY')

def get_key_words(element_content, prompt):
    """
        Returns keywords extracted from element content
    """

    text = prompt + " " + element_content
    max_tokens = 4097

    if len(text)> 4097*2.5:
        text = text[:int(max_tokens*2)]
        print("TRUNCATED: ", len(text))
    
    try:
        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text}
                ]
            )

        key_words = response["choices"][0]["message"]["content"]
    except Exception as e:
        print("ERROR: ", e)
        key_words = ""
    return key_words

def embedding_from_string(
    string: str,
    model: str = EMBEDDING_MODEL,
) -> list:
    """
        Returns embedding from string
    """
    embedding = get_embedding(string, model)
    
    return embedding

def get_embeddings_data(file: str):
    with gzip.open(file, 'rb') as f:
        # Loads data from .gzip file
        data = pickle.load(f)
        
    return data

def get_source_elements(file: str):
    try:
        with open(file, 'rb') as handle:
            source_elements = pickle.load(handle)
        
    except:
        print("No source elements file found")
    
    return source_elements

def get_element_info(element):
    return "{'name': "+element['name']+", 'description': "+str(element['content'])+"}" 

def get_recommendations(input_embedding, source_data, max_distance: float = 2.0, max_courses: int = 10, print_results: bool = False):
    source_embeddings = []
    recommended_elements = []
    for element in source_data:
        source_embeddings.append(element['embeddings'])
      
    # get distances between the source embedding and other embeddings (function from embeddings_utils.py)
    distances = distances_from_embeddings(input_embedding, source_embeddings, distance_metric="cosine")
    
    # get indices of nearest neighbors (function from embeddings_utils.py)
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)
    
    # print k nearest neighbors:
    k_counter = 0
    for i in indices_of_nearest_neighbors:
        # stop after printing out k articles
        if k_counter >= max_courses:
            break
            
        if distances[i]>max_distance:
            break
        k_counter += 1

        if print_results == True:
            # print out the similar strings and their distances
            print(
                f"""
            --- Recommendation #{k_counter} (nearest neighbor {k_counter}) ---
            Element: {source_data[i]['name']}
            Distance: {distances[i]:0.3f}"""
            )
        
        recommended_elements.append({"course":source_data[i]['name'], "distance": str(round(distances[i],3))})

    return recommended_elements

# Load source elements data
embeddings_data = get_embeddings_data(embeddings_data_file)
source_elements = get_source_elements(source_elements_file)

####################### DEFINE INPUT ELEMENT MENU ##########################
empty_element = {'id': "", 'name': "None", 'content': "" }
source_elements = [empty_element] + source_elements

input_element = {'id': "100", 'name': "", 'content': "" }
st.header("Text recommendation using OpenAI")

st.write("First, define the input element, or choose from one of the source elements ")

st.subheader("Select input element from source elements")
st.write("The following elements represent online courses:")
selected_element = st.selectbox("Select an element", options=[element['name'] for element in source_elements])
    
if selected_element:
    source_selected_element = next((element for element in source_elements if element['name'] == selected_element), None)
    input_element['name'] = source_selected_element['name']
    input_element['content'] = source_selected_element['content']


st.subheader("Or create your own")
with st.expander("Create your own element"):
    element_name = st.text_input("Type the name of the element", value=input_element['name'])
    element_content = st.text_input("Type the content of the element", value=input_element['content'])

    input_element['name'] = element_name
    input_element['content'] = element_content


st.write("Input element: ", input_element)

##############################################################################

############################ PROMPT MENU ###########################


st.subheader("Define ChatGPT prompt")
st.write("Here you must define the prompt that ChatGPT will use to generate the keywords for your input element. \
You can use the default prompt only if the input element represents an online course.")

default_promtp = "This is the information about an online course. You must generate 5 words that describe as precisely as possible the main topic of the course. You must write the 5 words in one single line."

prompt = st.text_input("Type the prompt", value=default_promtp)

##############################################################################


############################ RECOMMENDATIONS MENU ###########################

st.subheader("Generate recommendations")
st.write("Get recommended source elements based on the input element and prompt")

max_distance = st.slider("Max Distance", min_value=0.0, max_value=0.5, step=0.01, value=0.16)
st.write("Selected Max Distance: ", max_distance)


if 'generate_button_pressed' not in st.session_state:
    st.session_state.generate_button_pressed = False


if st.button("Generate recommendations"):
    st.session_state.generate_button_pressed = True


    with st.spinner("Generating keywords..."):
        # Get input element info formated
        input_element_info = get_element_info(input_element)

        # Get keywords from input element
        input_element_keywords = get_key_words(input_element_info, prompt)
        st.session_state.input_element_keywords = input_element_keywords
    st.write("Keywords from input element: ", input_element_keywords)

    with st.spinner("Generating embeddings..."):
        # Get embeddings from input element:
        input_element_embeddings = embedding_from_string(input_element_keywords)
    st.write("Embeddings generated!")

    with st.spinner("Generating recommendations..."):
        # Get recommended source elements from input element:
        st.session_state.recommended_elements = get_recommendations(input_element_embeddings, embeddings_data, max_distance=max_distance, print_results=False)


    

st.subheader("Results")
st.write("Input Element name: ", input_element['name'] )
if 'input_element_keywords' in st.session_state:
    st.write("Keywords: ", st.session_state.input_element_keywords)
st.write("Recommendations:")

if 'recommended_elements' in st.session_state:
    recommended_elements_adjusted = []
    for element in st.session_state.recommended_elements:
        if float(element['distance'])< max_distance:
            recommended_elements_adjusted.append(element)
    df = pd.DataFrame(recommended_elements_adjusted)
    st.table(df)