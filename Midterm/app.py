from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os

os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
# optionally set your wandb settings or configs
os.environ["WANDB_PROJECT"] = "trying_it_out"

pdf_link = "https://d18rn0p25nwr6d.cloudfront.net/CIK-0001045810/1cbe8fe7-e08a-46e3-8dcc-b429fc06c1a4.pdf"
loader = PyMuPDFLoader(
    pdf_link,
)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


documents = loader.load()
vector_store = FAISS.from_documents(documents, embedding_model)
retriever = vector_store.as_retriever()

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.prompts import ChatPromptTemplate

template = """Answer the question based only on the following context. If you cannot answer the question with the context, please respond with 'I don't know':

Context:
{context}

Question:
{input}
"""

prompt = ChatPromptTemplate.from_template(template)

document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)

import sys
from pathlib import Path
from chainlit.playground.providers.openai import ChatOpenAI
import chainlit as cl

import logging

@cl.on_message
async def main(query: cl.Message):
    logging.error(query.content)
    response = await retrieval_chain.ainvoke({"input": query.content})
    response = response['answer']
    if response:
        await cl.Message(content=response).send()
