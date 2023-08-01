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
    print("TEXT", text)
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

def get_element_info(element):
    return "{'name': "+element['name']+", 'description': "+str(element['content'])+"}" 


##################### PAGE ###########################
st.header("Keywords generation using OpenAI")

st.write("First, define the input element, or choose from one of the source elements ")

tab1, tab2 = st.tabs(["Input Element", "Target element"])
with tab1:
    input_element = {'name': "", 'content': ""}
    st.subheader("Generate an input element")
    with st.expander("Create your own input element"):
        element_name = st.text_input("Type the name of the element", value=input_element['name'])
        element_content = st.text_input("Type the content of the element", value=input_element['content'])

        input_element['name'] = element_name
        input_element['content'] = element_content
    
    # selected_element_type = st.selectbox("Select the element type", options=[element['name'] for element in source_elements])
    element_type = st.text_input("What does the element represent? Example: online course, scientific article, book...")

    prompt = f"""The folowwing element in JSON format represents a {element_type}.\
        You must generate 5 words that describe as precisely as possible the main topic of the {element_type}.\
        You must write the 5 words in one single line."""
    
    if st.button("Generate keywords"):
        st.session_state.generate_button_pressed = True

        with st.spinner("Generating keywords..."):
             # Get input element info formated
            input_element_info = get_element_info(input_element)

            # Get keywords from input element
            input_element_keywords = get_key_words(input_element_info, prompt)
            st.session_state.input_element_keywords = input_element_keywords
        st.write("Keywords from input element: ", input_element_keywords)

    
