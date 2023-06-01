# NLP-OpenAI

Natural Language Processing project using OpenAI technology.

## Brief description

This repository contains all the necessary code to run an IA model that aims to generate associations between different text elements based on the topic that they treat.

The model receives an element as input (input text) and returns those texts from the list of source elements (source texts or target texts) that show the highest similarity in their main topic.

The process to achieve this is as follows:

- ONLY FIRST TIME: All source elements are sent to the ChatGPT API with a prompt. This returns the characterization of all texts in the form of 5 keywords. These keywords are transformed into embeddings and stored in a .gz file named *embeddings_data.gz*.

1. The input text is sent to the ChatGPT API with a prompt, and it returns a characterization of the input text in the form of 5 keywords.

2. The characterization of the input text are transformed into embeddings (numeric vectors) using OpenAI.

3. The cosine distance between the input text embedding and the embeddings of all the target texts is calculated.

4. Those target texts that are under the defined maximum distance are returned. They are the final output of the system. 


Below is the diagram of the model:

![image](https://github.com/NCSanto01/NLP-OpenAI/assets/78079809/50954898-e707-4f93-a8ea-693098771533)


The file *UnderstandingEmbeddings.ipynb* contains a simple demo that shows how OpenAI embeddings work.


## Instructions

***IMPORTANT:*** You need an OpenAI API key to run the model. This key must be stored in a local text file named *.env* at the root of the project under the name of an environment variable "OPENAI_API_KEY". Below is an image that shows the *.env* file:

![image](https://github.com/NCSanto01/NLP-OpenAI/assets/78079809/da523d38-040f-4a77-94bb-4f1f704fb9ba)


### Creating source elements (target texts)

The first step is to feed the source elements to the model. This can be done in the CreateElements file, where you must add the code that creates the elements, which then are stored in a file named *source.pickle* as a list of elements.

All elements must be in the form of a dictionary with the following structure:
```python
{
    'id': " ",
    'name': " ",
    'content': " "
}
```

A few source elements that represent online courses are given as an example.


### Generating data of source elements

Once all source elements are stored in the *source.pickle* file, their corresponding characterizations and embeddings must be generated.

This can be done with the file *GenerateData.ipynb* by following the instructions described in it.

At the end, you will have a file named *embeddings_data.gz* which contains all the source elements data that the model needs.


### Running the model

The model can be run in two ways.

#### Notebook: *main.ipynb*

In this file you can run the model just like the other files, inside a notebook.

#### Streamlit page: *main_streamlit.py* (RECOMMENDED)

This simple streamlit page allows you to run the model in a faster, easier, more visual and more intuitive way.

Having streamlit installed (``` pip install streamlit ```), run the following command to open the page:
```console
streamlit run main_streamlit.py
```

### System configuration

In the *config.py* file, you can change all the file names as you wish.


