from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_models import AzureChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import HTMLHeaderTextSplitter
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

docs = list()
from langchain_core.documents import Document

filter_links_list = [
]

chunk_size = 500
chunk_overlap = 30
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap
)

links = ['https://www.fantasyfootballscout.co.uk/2024/03/18/fpl-gameweek-30-early-scout-picks-chelsea-double-up']

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

for link in links:
    if link in filter_links_list:
        continue
    print(link)

    html_header_splits = html_splitter.split_text_from_url(link)
    toss = html_header_splits.pop(0)
    html_header_splits = html_header_splits[:5]
    metadata = dict()
    metadata["link"] = link

    splits = text_splitter.split_documents(html_header_splits)

    for chunk in splits:
        text = list()
        if len(chunk.page_content) < 80: # and len(chunk.metadata) == 0:
            continue
        text.append(chunk.page_content)

        for _, value in chunk.metadata.items():
            text.append(value)

        # endpoint = link.split("/")[-1]
        # endpoint = endpoint.replace("-", " ")
        # endpoint = f"The endpoint name that this text is found at is here: {endpoint}"

        text = " ".join(text) # + [endpoint])

        chunk = Document(page_content=text, metadata=metadata)
        docs.append(chunk)


vectorstore = Chroma.from_documents(
    docs, embeddings_model #, persist_directory="./chroma_db"
)
vectorstore.similarity_search("Who has the best match ups this week?")

retriever = vectorstore.as_retriever()

template = """You are an expert in Fantasy Premiere League Football.
You will be given questions about the current game week along with some context. Make sure you're answers are grounded in the context provided.
Today's Date is: March 25, 2024

Make sure you supplement your answers with clear terminology. For example, if you reference a "Double Gameweek", make sure you explain what you mean by this.

Answer the question based only on the following context.
If you cannot answer the question with the context, please respond with 'I don't know'.

Context:
{context}

Question:
{input}
"""
prompt = ChatPromptTemplate.from_template(template)

document_chain = create_stuff_documents_chain(llm, prompt)
website_retrieval_chain = create_retrieval_chain(retriever, document_chain)

# retrieval_chain.invoke({'input':'Who has the best match ups this week?'})