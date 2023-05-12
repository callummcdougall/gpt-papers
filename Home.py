import os
import streamlit as st
import pickle
import sys
import platform
from pathlib import Path
from typing import List

st.set_page_config(layout="wide", page_icon="🔬")

is_local = (platform.processor() != "")

from chatbot import answer_question, Embedding, EmbeddingGroup, get_pages_from_url, get_pages_from_upload

MAX_CHAR_LEN = 600

if "history" not in st.session_state:
    st.session_state["history"] = []

# %%

st.markdown(r"""
## 🤖 Chatbot

This is a simple chatbot that can answer questions about an online paper (or any other kind of pdf, although it's been optimized for papers).

Here's how to use it:

1. Create embeddings. You can do this in one of two ways:
    a. Type in the url to the box below, and hit "create embedding", then download the embedding file.
    b. Upload a pdf, then hit "create embedding".

2. Upload the embedding file you get from step 1 to the "upload embedding" box below.

3. Ask your question!
""")


st.markdown("---")

cols = st.columns(2)

with cols[0]:

    upload_pdf = st.file_uploader(
        label = "Upload pdf",
    )
    upload_pdf_button = st.button("Create embedding from pdf")
    if upload_pdf_button:
        if upload_pdf:
            st.write(upload_pdf.readlines())
            st.write(dir(upload_pdf))
            pages = get_pages_from_upload(upload_pdf)
            embeddings = EmbeddingGroup()
            my_bar = st.progress(0.0)
            total_char_len = sum([len(page) for page in pages])
            char_len = 0
            for page in pages:
                while len(page) > 0:
                    prefix = page[:MAX_CHAR_LEN]
                    char_len += len(prefix)
                    page = page[MAX_CHAR_LEN:]
                    embeddings.add_embedding("", prefix)
                    my_bar.progress(100 * char_len / total_char_len)
            download_embeddings = st.download_button(
                label = "Download embedding",
                data = pickle.dumps(embeddings),
                file_name = "my_embeddings.pkl",
            )
        else:
            st.error("You must upload a file.")
        st.write(pages)

with cols[1]:

    input_url = st.text_input(
        label = "URL",
        value = "",
    )
    input_url_button = st.button("Create embedding from url")
    if input_url_button:
        if input_url:
            with st.spinner("Extracting pages of text..."):
                pages = get_pages_from_url(input_url)
            embeddings = EmbeddingGroup()
            my_bar = st.progress(0.0)
            total_char_len = sum([len(page) for page in pages])
            char_len = 0
            st.write(char_len, total_char_len)
            for page in pages:
                while len(page) > 0:
                    prefix = page[:MAX_CHAR_LEN]
                    char_len += len(prefix)
                    page = page[MAX_CHAR_LEN:]
                    embeddings.add_embedding("", prefix)
                    my_bar.progress(char_len / total_char_len)
            download_embeddings = st.download_button(
                label = "Download embedding",
                data = pickle.dumps(embeddings),
                file_name = "my_embeddings.pkl",
            )
        else:
            st.error("You must input a url.")


st.markdown("")
st.markdown("---")
st.markdown("")

upload = st.file_uploader(
    label = "Upload embedding",
    type = "pkl",
)
if upload:
    st.session_state["my_embeddings"] = pickle.loads(upload.read())

st.markdown("")
st.markdown("---")
st.markdown("")

question = st.text_area(
    label = "Prompt:", 
    value = "", 
    key = "input",
    placeholder="Type your prompt here, then press Ctrl+Enter.\nThe prompt will be prepended with most of the page content (so you can ask questions about the material)."
)

with st.sidebar:

    model = st.radio(
        "Model",
        options = ["gpt-4", "gpt-3.5-turbo", "text-davinci-003"],
        index = 1
    )

    temp = st.slider(
        "Temperature",
        min_value = 0.0,
        max_value = 2.0,
        value = 0.5,
    )

    st.markdown("---")

    clear_output_button = st.button("Clear output")
    if clear_output_button:
        st.session_state["history"] = []
        st.session_state["suppress_output"] = True
    else:
        st.session_state["suppress_output"] = False

    st.markdown("")
    st.markdown("*Note - chat history is not yet supported, so you should limit your prompts to single questions.*")


st.markdown("## Response:")
response_global_container = st.container()

# import streamlit_chat as sc

# %%

if question and (not st.session_state["suppress_output"]):
    with response_global_container:
        st.info(question)
        response_container = st.empty()
        for i, hist in enumerate(st.session_state["history"]):
            if i % 2 == 0:
                st.info(hist)
            else:
                st.markdown(hist)
        st.session_state["history"].append(question)
        # Get all the embeddings, by reading from file
        my_embeddings: EmbeddingGroup = st.session_state["my_embeddings"]
        # If we're not including solutions, then filter them out
        # if not include_solns:
        #     my_embeddings=my_embeddings.filter(title_filter = lambda x: "(solution)" not in x)
        # if exercises:
        #     my_embeddings=my_embeddings.filter(title_filter = lambda x: any([ex.replace("_", " ") in x for ex in exercises]))
        if len(my_embeddings) == 0:
            st.error("Warning - your filters are excluding all content from the chatbot's context window.")
            # st.stop()
        response = answer_question(
            my_embeddings=st.session_state["my_embeddings"], 
            question=question, 
            prompt_template="SIMPLE", # "SOURCES", "COMPLEX"
            model=model,
            debug=False,
            temperature=temp,
            container=response_container,
            max_len=2000, # max content length
            max_tokens=2000,
        )
else:
    st.session_state["suppress_output"] = False



# %%
