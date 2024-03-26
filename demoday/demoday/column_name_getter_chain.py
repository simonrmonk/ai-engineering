from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

new_line_character = '\n'
examples_noun_id = [
    {
        'user_query': 'How many goals has Silva scored?',
        'output': [('full_name', 'Silva')]
    },
    {
        'user_query': 'Who has scored more goals, Silva or Salah?',
        'output': [('full_name', 'Silva'), ('full_name', 'Salah')]
    },
]

example_prompt = PromptTemplate(
    input_variables=["user_query", "output"], template="Input by user:\n{user_query}\n\nAI Output:\n{output}"
)

system = '''You will be provided with a user query about a database and an SQL table schema.
From these inputs, you will need to determine the columns that the user is referring to in the query.
Only return a list of tuples that contain the column name and noun for filter on, exactly like this: [(column_name1, noun1), (column_name2, 'noun2'), ('column_name1', 'noun3')].

Here is the table schema:
{table_schema}


'''

few_shot_prompt = FewShotPromptTemplate(
    examples=examples_noun_id,
    example_prompt=example_prompt,
    prefix=system,
    suffix='',
    input_variables=["table_schema"],
)


full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("human", "Input by user:\n{user_query}\n\nAI Output:\n"),
    ]
)

# .format(table_schema=db.get_table_info(['cleaned_players'])))

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) #, model_kwargs={"response_format": {"type": "json_object"}})

chain_filter_confirmer = full_prompt | llm

# chain_filter_confirmer.invoke({'user_query':  'How many goals has Raheem scored vs. Salah?', 'table_schema': db.get_table_info(['cleaned_players'])})