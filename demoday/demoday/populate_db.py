from langchain_community.tools import HumanInputRun
import chainlit as cl
import ast
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from column_name_getter_chain import chain_filter_confirmer
from langchain_core.agents import AgentActionMessageLog
from langchain_core.messages import AIMessage
import json
from langchain.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from website_data import website_retrieval_chain
from langchain.agents import (
    AgentExecutor,
    create_openai_functions_agent,
    create_sql_agent,
)
import operator
from typing import Annotated, Sequence, TypedDict, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda

from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor
import logging
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from typing import Literal
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from sqlalchemy import create_engine, text
import pandas as pd
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import os



embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
llm4 = ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=0)
llm3 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


engine = create_engine('sqlite:///fpl.db', echo=True)

HERE = Path(__file__).parents[1]

data_path = HERE.joinpath('FPL_csvs')

# Get a list of all CSV files in the directory
csv_files = [str(f) for f in data_path.glob('*.csv')]

csv_files = [csv_files[0]]

table_names = list()
for csv_file in csv_files:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    table_name = Path(csv_file).stem
    if table_name == 'cleaned_players':
        df['full_name'] = df['first_name'] + ' ' + df['second_name']
        df = df.drop(columns=['first_name', 'second_name'])
    
    # Write the DataFrame to the SQLite database
    # Use the filename (without the .csv extension) as the table name
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    table_names.append(table_name)

# Define your SQL query
query = text("SELECT * FROM cleaned_players")

with engine.connect() as connection:
    # Execute the query
    result = connection.execute(query)

    # Fetch all rows from the result
    rows = result.fetchall()
    df = pd.DataFrame(rows)
    categorical_columns = df.select_dtypes(include=['object']).columns

HERE = Path(__file__).parent
vector_store_path = str(HERE.joinpath("chroma_db_nouns"))

# outputs = {}
# categorical_columns = ['full_name']
# for category in categorical_columns:
#     df_temp = df[[category]].drop_duplicates()
#     items = df_temp[category].tolist()
#     docs = list()
#     for item in items:
#         item = Document(page_content=item.strip())
#         print(item.page_content)
#         docs.append(item)
#     Chroma.from_documents(docs, embeddings_model, persist_directory=vector_store_path, collection_name=category)

retrievers = dict()
for column in categorical_columns:

    vectorstore = Chroma(
        persist_directory=vector_store_path, embedding_function=embeddings_model, collection_name=column
    )
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.4})
    retrievers[column] = retriever
# retrievers[column].get_relevant_documents("Silva")

new_line_character = '\n'
examples_noun_id = [
    {
        # 'question_to_user': ,
        'question_to_user': f"There are multiple player names that match that query, please choose one:{new_line_character}{new_line_character.join(['Raheem Sterling', 'Sterling Silver', 'Harry Kane', 'Mohamed Salah'])}",
        'user_response': 'Harry',
        'output': 'Harry Kane'
    },
    {
        'question_to_user': f"There are multiple player names that match that query, please choose one:{new_line_character}{new_line_character.join(['Raheem Sterling', 'Sterling Silver', 'Harry Kane', 'Mohamed Salah'])}",
        'user_response': 'the last one',
        'output': 'Mohamed Salah'
    },
    {
        'question_to_user': f"There are multiple player names that match that query, please choose one:{new_line_character}{new_line_character.join(['Raheem Sterling', 'Sterling Silver', 'Harry Kane', 'Mohamed Salah'])}",
        'user_response': 'none of those',
        'output': 'NA'
    },
    {
        'question_to_user': 'Please confirm the player you are interest in is Raheem Sterling. Y/N?',
        'user_response': 'yes',
        'output': 'Raheem Sterling'
    },
    {
        'question_to_user': 'Please confirm the player you are interest in is Raheem Sterling. Y/N?',
        'user_response': 'no thanks',
        'output': 'NA'
    }
]

example_prompt = PromptTemplate(
    input_variables=["question_to_user", "user_response", "output"], template="Question to User: {question_to_user}\n'User Response': {user_response}\n{output}"
)

system = '''You will be provided a list of options a human has to choose from.
Your an expect at identifying, given a user input, which of the options they are refering to.
You must only respond with the exact spelling of the option the user is refering to.
The only exception is when you're not sure which option the user is refering to or the user indicates
that the option they want is something else, in which case you should respond with 'NA'.
'''

few_shot_prompt = FewShotPromptTemplate(
    examples=examples_noun_id,
    example_prompt=example_prompt,
    suffix=system + "Question to User: {question_to_user}\n'User Response': {user_response}",
    input_variables=["question_to_user", "user_response"],
)


full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("human", "Question to User: {question_to_user}\n'User Response': {user_response}"),
    ]
)

chain_get_noun = full_prompt | llm4

def ask_humanv2(column, options):

    if len(options) > 1:
        numbered_items = ''.join([f"{new_line_character}{i+1}. {item}" for i, item in enumerate(options)])
        question = f'There are multiple {column}s that match that query, please choose one:{new_line_character}{numbered_items}'
    elif len(options) == 1:
        question = f'Please confirm the {column} you want to filter on is {options[0]}'
    elif len(options) == 0:
        return 'NA'
    human_response = input(question)
    # try:
    #     human_response  = await cl.AskUserMessage(content=question, timeout=20).send()
    #     human_response = human_response['output']
    #     noun = chain_get_noun.invoke({'question_to_user':  options, 'user_response': human_response})
    # except:
    #     noun = 'NA'
    noun = chain_get_noun.invoke({'question_to_user':  options, 'user_response': human_response})
    
    if noun:
        # return data # change this with new state
        return noun

class NounLookupInput(BaseModel):
    column: Literal['full_name'] = Field(description="This is the column that the user wants to filter on. For example 'full_name'")
    input: str = Field(description="This is the string that the user passed in to use for the column filter. For example 'Raheem'")

@tool(args_schema=NounLookupInput)
def noun_lookup_tool(column, input) -> str:
    '''
    This tool must be called when we want to filter a column.
    The input is the original spelling that the user put in.
    The tool will ask the human for clarification on which exact noun they want to filter on.
    The output of this tool is an exact match spelling of the noun.
    If the response is 'NA', it means the user didn't want any of the options OR there were no nouns that were scemantically similar enough.
    
    Example input: {'column': 'full_name', 'input': 'Raheem'}
    Example output: 'Raheem Sterling'
    
    Make sure that when you use this tool, you provide the exact noun name you ended up using in your final output for transparency.
    
    '''
    
    retriever = retrievers[column]
    options = retriever.get_relevant_documents(input)
    options = [option.page_content for option in options]
    noun = ask_humanv2(column, options)
    
    return noun

# def general_human_feedback(question_for_user) -> str:
    
#     try:
#         human_response  = await cl.AskUserMessage(content=question, timeout=20).send()
#         human_response = human_response['output']
#         noun = chain_get_noun.invoke({'question_to_user':  options, 'user_response': human_response})
#     except:
#         noun = 'NA'
    
#     return noun


def _get_agent_state(input_schema=None):
    if input_schema is None:

        class AgentState(TypedDict):
            # The input string
            input: str
            # The list of previous messages in the conversation
            chat_history: Sequence[BaseMessage]
            # The outcome of a given call to the agent
            # Needs `None` as a valid type, since this is what this will start as
            agent_outcome: Union[AgentAction, AgentFinish, None]
            # List of actions and corresponding observations
            # Here we annotate this with `operator.add` to indicate that operations to
            # this state should be ADDED to the existing values (not overwrite it)
            intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

    else:

        class AgentState(input_schema):
            # The outcome of a given call to the agent
            # Needs `None` as a valid type, since this is what this will start as
            agent_outcome: Union[AgentAction, AgentFinish, None]
            # List of actions and corresponding observations
            # Here we annotate this with `operator.add` to indicate that operations to
            # this state should be ADDED to the existing values (not overwrite it)
            intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

    return AgentState

# @tool
# def first_model(data):
#     human_input = data["input"]
    
#     outputs = chain_filter_confirmer.invoke({'user_query': human_input, 'table_schema': db.get_table_info(['cleaned_players'])})
    
#     actual_list_of_tuples = ast.literal_eval(outputs.content)
    
#     agent_actions = list()
#     for output in actual_list_of_tuples:
#         column, noun = output
#         message_log = AIMessage(
#             content= '',
#             additional_kwarg= {'function_call': {'column': column, 'input': noun, 'name': 'noun_lookup_tool'}},
#             type='ai'
            
#         )
#         log = "\nInvoking: `noun_lookup_tool` with `{'column': '{column}', 'input': '{noun}'}`\n\n\n"
#         agent_action = AgentActionMessageLog(
#             tool="noun_lookup_tool", tool_input={'column': column, 'input': noun}, log=log , message_log=[message_log]
#         )
#         agent_actions.append(agent_action)

#     return {"agent_outcome": agent_actions}

def create_agent_executor(agent_runnable, agent_runnable_backup, tools, input_schema=None):
    if isinstance(tools, ToolExecutor):
        tool_executor = tools
    else:
        tool_executor = ToolExecutor(tools)

    state = _get_agent_state(input_schema)

    # Define logic that will be used to determine which conditional edge to go down

    def should_continue(data):
        # If the agent outcome is an AgentFinish, then we return `exit` string
        # This will be used when setting up the graph to define the flow
        if isinstance(data["agent_outcome"], AgentFinish):
            return "end"
        # Otherwise, an AgentAction is returned
        # Here we return `continue` string
        # This will be used when setting up the graph to define the flow
        else:
            return "continue"

    def run_agent(data):
        agent_outcome = agent_runnable.invoke(data)
        return {"agent_outcome": agent_outcome}

    async def arun_agent(data):
        agent_outcome = await agent_runnable.ainvoke(data)
        return {"agent_outcome": agent_outcome}

    # Define the function to execute tools
    def execute_tools(data):
        # Get the most recent agent_outcome - this is the key added in the `agent` above
        agent_action = data["agent_outcome"]
        if isinstance(agent_action, list):
            output = tool_executor.batch(agent_action, return_exceptions=True)
            return {
                "intermediate_steps": [
                    (action, str(out)) for action, out in zip(agent_action, output)
                ]
            }
        output = tool_executor.invoke(agent_action)
        return {"intermediate_steps": [(agent_action, str(output))]}

    async def aexecute_tools(data):
        # Get the most recent agent_outcome - this is the key added in the `agent` above
        agent_action = data["agent_outcome"]
        if isinstance(agent_action, list):
            output = list()
            for action in agent_action:
                out = await tool_executor.ainvoke(action)
                output.append(out)
            return {
                "intermediate_steps": [
                    (action, str(out)) for action, out in zip(agent_action, output)
                ]
            }

        output = await tool_executor.ainvoke(agent_action)
        return {"intermediate_steps": [(agent_action, str(output))]}

    # def first_model(data):
    #     human_input = data["input"]
        
    #     outputs = chain_filter_confirmer.invoke({'user_query': human_input, 'table_schema': db.get_table_info(['cleaned_players'])})
        
    #     actual_list_of_tuples = ast.literal_eval(outputs.content)
        
    #     agent_actions = list()
    #     for output in actual_list_of_tuples:
    #         column, noun = output
    #         message_log = AIMessage(
    #             content= '',
    #             additional_kwarg= {'function_call': {'column': column, 'input': noun, 'name': 'noun_lookup_tool'}},
    #             type='ai'
                
    #         )
    #         log = "\nInvoking: `noun_lookup_tool` with `{'column': '{column}', 'input': '{noun}'}`\n\n\n"
    #         agent_action = AgentActionMessageLog(
    #             tool="noun_lookup_tool", tool_input={'column': column, 'input': noun}, log=log , message_log=[message_log]
    #         )
    #         agent_actions.append(agent_action)

    #     return {"agent_outcome": agent_actions}

    # Define a new graph
    workflow = StateGraph(state)

    # Define the two nodes we will cycle between
    # workflow.add_node("first_agent", first_model)
    workflow.add_node("agent", RunnableLambda(run_agent, arun_agent))
    workflow.add_node("action", RunnableLambda(execute_tools, aexecute_tools))
    # workflow.add_node("human", RunnableLambda(ask_human))

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point("agent")
    # workflow.set_entry_point("first_agent")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
        {
            # If `tools`, then we call the tool node.
            "continue": "action",
            # Otherwise we finish.
            "end": END,
        },
    )
    
    workflow.add_edge("action", "agent")
    # After we call the first agent, we know we want to go to action
    # workflow.add_edge("first_agent", "action")

    return workflow.compile()

db = SQLDatabase(engine, include_tables=["cleaned_players"])


toolkit = SQLDatabaseToolkit(db=db, llm=llm4)

tools = toolkit.get_tools()

# from langchain.tools import StructuredTool, tool
# website_info = StructuredTool.from_function(
#     func=website_retrieval_chain.invoke,
#     name="website_retrieval_chain",
#     description="This tool is used to retrieve expert opinion about planning for the current game week. It will return valuable insights about the current game week.",
#     return_direct=True,
# )

tool_human = HumanInputRun()

tools = tools + [tool_human, noun_lookup_tool] #, website_info]

from langchain import hub
prompt = hub.pull("hwchase17/openai-functions-agent")
from sample_retriever import full_sql_prompt
agent_runnable4 = create_openai_functions_agent(llm4, tools, full_sql_prompt) # prompt)
agent_runnable3 = create_openai_functions_agent(llm3, tools, full_sql_prompt) # prompt)
app = create_agent_executor(agent_runnable4, agent_runnable3, tools)
from langchain_core.runnables import RunnableConfig
config = RunnableConfig(recursion_limit=100)

output = app.invoke({"input": "Tell me how many goals Mohamed has?", "chat_history": []}, config)

print('test')


