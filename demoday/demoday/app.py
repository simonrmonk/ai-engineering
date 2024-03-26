from chainlit.input_widget import Select
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os

# os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
# os.environ["WANDB_PROJECT"] = "trying_it_out"

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

import sys
from pathlib import Path
from chainlit.playground.providers.openai import ChatOpenAI
import chainlit as cl
from populate_db import app, config
from website_data import website_retrieval_chain

import logging

logger = logging.getLogger(__name__)

@cl.set_chat_profiles
async def persona_profile():
    return [
        cl.ChatProfile(
            name="Chat with FPL Data",
            markdown_description="Get the latest stats",
            icon="https://picsum.photos/200",
        ),
        cl.ChatProfile(
            name="FPL Scout Expert Q&A",
            markdown_description="Ask the FPL Scout",
            icon="https://picsum.photos/250",
        ),
    ]
    
@cl.on_chat_start
async def on_chat_start():
    user = cl.user_session.get("user")
    chat_profile = cl.user_session.get("chat_profile")
    logger.info(f"User: {user}, Chat Profile: {chat_profile}")
    
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="OpenAI - Model",
                values=["gpt-4-turbo", "gpt-3.5-turbo"],
                initial_index=0,
            ),
        ]
    ).send()

async def run_scout_qa(query):
    
    msg = cl.Message(content='')
    new_line = '\n'
    
    async for chunk in website_retrieval_chain.astream(
        {"input": query.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        answer = chunk.get('answer')
        if answer:
            await msg.stream_token(answer)
        context = chunk.get('context')
        if context:
            sources = [x.metadata['link'] for x in context]
            sources = f'{new_line}{new_line} Sources:{new_line}' + f"{new_line}".join(list(set(sources)))
            
    await cl.Message(content=sources).send()
    
    return msg.content


@cl.on_message
async def main(query: cl.Message):
    chat_profile = cl.user_session.get("chat_profile")
    logger.info(f"User Input Question: {query.content}")
    if chat_profile == "Chat with FPL Data":
        response = await app.ainvoke({"input": query.content, "chat_history": []}, config)
        response = response['agent_outcome'].return_values['output']
        if response:
            await cl.Message(content=response).send()
    elif chat_profile == "FPL Scout Expert Q&A":
        response = await run_scout_qa(query)
