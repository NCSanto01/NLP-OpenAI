# NLP-OpenAI

Natural Language Processing project using OpenAI technology.

## Brief description

This repository contains all the necessary code to run an IA model that aims to generate associations between different text elements based on the topic that they treat.

The model receives an element as input (input text) and returns those texts from the list of source elements (source texts or target texts) that show the highest similarity in their main topic.

The process to achieve this is as follows:

- ONLY FIRST TIME: All source elements are sent to the ChatGPT API with a prompt. This returns the characterization of all texts in the form of 5 keywords. These keywords are transformed into embeddings and stored in a .gz file called "embeddings_data.gz".

1. The input text is sent to the ChatGPT API with a prompt, and it returns a characterization of the input text in the form of 5 keywords.

2. The characterization of the input text are transformed into embeddings (numeric vectors) using OpenAI.

3. The cosine distance between the input text embedding and the embeddings of all the target texts is calculated.

4. Those target texts that are under the defined maximum distance are returned. They are the final output of the system. 


Below is the diagram of the model:

![image](https://github.com/NCSanto01/NLP-OpenAI/assets/78079809/50954898-e707-4f93-a8ea-693098771533)


## Instructions

***IMPORTANT:*** You need an OpenAI API key to run the model. This key must be stored in a local text file named ".env" at the root of the project under the name of an environment variable "OPENAI_API_KEY". Below is an image that shows the .env file:

