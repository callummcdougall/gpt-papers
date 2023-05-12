# %%

import streamlit as st
import openai
from openai.embeddings_utils import distances_from_embeddings
from tensorboard import notebook
import torch as t
import numpy as np
import pandas as pd
import tiktoken
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import os
from transformer_lens import utils
from tqdm.notebook import tqdm
import torch as t
from typing import List, Optional, Dict, Union, Callable
import pickle
from dataclasses import dataclass
import time

# Get to chapter0_fundamentals directory (or whatever the chapter dir is)
import os, sys

from chatbot import Embedding, EmbeddingGroup, tokenizer, SEPARATOR

openai.api_key = "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"

MAIN = __name__ == '__main__'

# %%

class StreamlitPage:
    '''This is a class which is designed to read a Streamlit page, and convert it into a series of chunks,
    then save those chunks as a text file.
    It's useful to have the chunks in this form, because then I can manually inspect them.
    '''
    def __init__(
        self, 
        path: Path, 
        save_chunks: bool, 
        max_chunk_length: int = 700,
        version: Optional[int] = None,
    ):
        self.path = path
        self.title = path.name.split("]_")[-1].replace(".py", "")
        self.max_chunk_length = max_chunk_length
        text_raw = path.read_text()        
        self.text = self.format_streamlit_page(text_raw)
        self.chunks = self.chunk_streamlit_page(self.text, filename=path.name)
        if save_chunks:
            self.save(version)

    def format_streamlit_page(self, text_raw: str):
        '''
        Just gets the st.markdown-wrapped markdown from the instructions page, stripping away all the 
        bits I don't need (e.g. imports outside the markdown, or contents pages).
        '''
        lines = text_raw.splitlines()
        lines_to_return = []
        adding_lines = False
        prev_line_not_empty = False
        for idx, line in enumerate(lines):
            this_line_not_empty = (line.strip() != "")
            if line.strip() == 'st.markdown(r"""':
                adding_lines = True
            elif line.strip() == r'""")' or line.strip() == r'""", unsafe_allow_html=True)':
                adding_lines = False
            elif adding_lines and (prev_line_not_empty or this_line_not_empty):
                lines_to_return.append(line)
            prev_line_not_empty = (line.strip() != "")
        return "\n".join(lines_to_return)

    def chunk_streamlit_page(self, s: str, filename: str = "") -> List[str]:
        '''
        Splits the text from a streamlit page into chunks.

        Basic idea: if you're on an empty line and you're not in a block (e.g. a python function), add 
        this index as a chunk split point. If the line isn't empty, then check its contents to see if 
        you've just started or ended a new block. At the end, split at all the chunk split points.
        '''
        lines = s.splitlines() + ["", ""]
        chunk_split_indices = [0]
        debug = False
        in_block = {"python": False, "title": False, "learning_obj": False, "details": False, "ul": False, "ol": False, "colon": False}
        for (i, line), line1, line2 in zip(enumerate(lines), lines[1:], lines[2:]):

            # Add chunk if you're not in any of the "wait until the end" bits
            # Also, if this is a double line break, we don't want to split twice!
            if (line.strip() == ""):
                if all([not v for v in in_block.values()]) and (line1.strip() != ""):
                    chunk_split_indices.append(i)

            else:
                # If in a python function, don't add a chunk until you're done
                if line.strip() == "```python":
                    in_block["python"] = True
                    in_block["title"] = False
                if line.strip() == "```":
                    in_block["python"] = False
                if not in_block["python"]:
                    in_block["title"] = line.strip().startswith("#")
                in_block["colon"] = line.strip().endswith(":")
                if "<details>" in line:
                    in_block["details"] = True
                if "</details>" in line:
                    in_block["details"] = False
                # If in bullet points, complete the bullet points!
                # Note, we can be in a bullet point and end a bullet point at the same time
                if self.is_ul(line):
                    in_block["ul"] = True
                if any([(L.strip() and (not self.is_ul(L))) for L in [line1, line2]]) and in_block["ul"]:
                    in_block["ul"] = False
                # If in numbered points, complete the numbered points!
                if self.is_ol(line):
                    in_block["ol"] = True
                if any([(L.strip() and (not self.is_ol(L))) for L in [line1, line2]]) and in_block["ol"]:
                    in_block["ol"] = False

        
        chunks = []
        for split_start, split_end in zip(chunk_split_indices, chunk_split_indices[1:]):
            # We don't want to include the chunk split line itself, since it's empty
            chunk = "\n".join(lines[split_start+1:split_end]).strip()
            if self.filter_chunks(chunk):
                chunks.extend(self.process_chunk(chunk))

        return chunks

    def filter_chunks(self, chunk: str):
        '''
        Checks if a chunk should be included in the embeddings.
        '''
        # Don't include empty chunks
        if chunk.strip() == "": return False

        # Don't include markdown line dividers
        if chunk.strip() == "---": return False

        # Don't include images
        if len(chunk.splitlines()) == 1 and "<img" in chunk: return False

        return True

    def process_chunk(self, chunk: str):
        '''
        Turns some of the messy bits of chunks into nice text for inclusion in the chatbot context.

        Also splits it up if it's too long.
        '''
        chunk = (
            chunk
            .replace("<details>", "")
            .replace("</details>", "")
            .replace("<summary>Solution</summary>", "Solution:\n")
            .replace("\n\n```", "\n```")
            .replace("```python\n\n", "```python\n")
        ).strip()
        if len(tokenizer.encode(chunk)) > self.max_chunk_length:
            # Split chunk up into an estimate for the appropriate number of chunks
            num_chunks = int(len(tokenizer.encode(chunk)) / self.max_chunk_length) + 1
            split_points = [int(i * len(chunk) / num_chunks) for i in range(num_chunks)] + [len(chunk)]
            chunks = [chunk[i: j] for i, j in zip(split_points[:-1], split_points[1:])]
            return chunks
        else:
            return [chunk]
                
    def save(self, version: Optional[int]) -> None:
        filename = f"chunk_{self.title}.txt"
        if version is not None:
            filename = f"chunk_{self.title}_v{version}.txt"
        path: Path = self.path.parent / filename
        with path.open("w", encoding="utf-8") as f:
            f.write(SEPARATOR.join(self.chunks))
        print(f"Saved {len(self.chunks)} chunks at {path.name!r}")

    def is_ul(self, line: str):
        return line.strip().startswith("* ") or line.strip().startswith("- ")

    def is_ol(self, line: str):
        return any([line.startswith(f"{d}. ") for d in range(1, 10)])



# %%

# Code to create chunks, and save them

if MAIN:
    my_embeddings = EmbeddingGroup()
    chunks = []

    paths = Path('instructions/pages').glob('*.py')
    for path in paths:
        if "🤖" not in path.name:
            # if "02" in path.name:
            #     one = StreamlitPage(path, save_chunks=True, version=None).text
            chunks.extend(StreamlitPage(path, save_chunks=True, version=None).chunks)

# %%

# Code to generate embeddings from chunks

if MAIN:
    chunk_paths = Path(r"C:\Users\calsm\Documents\AI Alignment\ARENA_2\chapter0_fundamentals\streamlit\pages").glob("chunk_*.txt")
    my_embeddings = EmbeddingGroup.from_chunked_files(chunk_paths)
    my_embeddings.save("my_embeddings.pkl")

# %%

