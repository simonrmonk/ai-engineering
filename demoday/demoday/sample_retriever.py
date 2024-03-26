from langchain_core.example_selectors import SemanticSimilarityExampleSelector
import json
from pathlib import Path
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

def load_json_file(file_path, encoding="utf8"):
    with open(file_path, encoding=encoding) as file:
        return json.load(file)

HERE = Path(__file__).parent
examples = load_json_file(HERE.joinpath("query_samples.json"))

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embeddings_model,
    FAISS,
    k=3,
    input_keys=["input"],
)

system_prefix = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 10 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.

Here is a mapping of positions to position codes in the DB:
    GK: goalkeeper
    DEF: defender
    MID: midfielder
    FWD: forward

If you are going to be filtering on the categorical column "full_name", you absolutely must call the noun_lookup_tool.

Here are some examples of user inputs and their corresponding SQL queries:"""

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate.from_template(
        "User input: {input}\nSQL query: {query}"
    ),
    input_variables=["input"],
    prefix=system_prefix,
    suffix="",
)

full_sql_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# # Example formatted prompt
# prompt_val = full_prompt.invoke(
#     {
#         "input": "How many arists are there",
#         "top_k": 5,
#         "dialect": "SQLite",
#         "agent_scratchpad": [],
#     }
# )