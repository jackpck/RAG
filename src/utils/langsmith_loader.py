from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate
from langsmith.schemas import Dataset
from langsmith.utils import LangSmithError
import os
import json

os.environ["LANGSMITH_API_KEY"] = os.environ["LANGSMITH_API_KEY"].rstrip()
os.environ["LANGSMITH_WORKSPACE_ID"] = os.environ["LANGSMITH_WORKSPACE_ID"].rstrip()
os.environ["LANGSMITH_ENDPOINT"] = os.environ["LANGSMITH_ENDPOINT"].rstrip()
os.environ["LANGSMITH_PROJECT"] = os.environ["LANGSMITH_PROJECT"].rstrip()
os.environ["LANGSMITH_TRACING"] = os.environ["LANGSMITH_TRACING"].rstrip()
os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"].rstrip()

def load_prompt(prompt_name: str,
                prompt_version: str = "latest") -> ChatPromptTemplate:
    client = Client()
    prompt = client.pull_prompt(f"{prompt_name}:{prompt_version}")

    return prompt

def load_data(data_name: str,
              examples: dict = None) -> Dataset:
    """
    load dataset from langsmith
    :param data_name: data_name of the dataset saved on langsmith. Always need to specify this
    :param examples: if creating dataset the first time, specify example. This will create and push
                     the examples to langsmith
    :return:
    """
    client = Client()
    if examples:
        if not client.has_dataset(dataset_name=data_name):
            dataset = client.create_dataset(dataset_name=data_name)
        else:
            dataset = client.read_dataset(dataset_name=data_name)
        client.create_examples(
            dataset_id=dataset.id,
            examples=examples
        )
        dataset = client.read_dataset(dataset_name=data_name)
    else:
        try:
            dataset = client.read_dataset(dataset_name=data_name)
        except LangSmithError:
            print("need to push examples to langsmith first. Please specify the examples argument")
            exit
    return dataset

if __name__ == "__main__":

    EVAL_DATA_PATH = "./data/evaluation/eval_examples.json"
    data_name = "rag_test"

    with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
        data = f.read()
    examples = json.loads(data)["examples"]

    dataset = load_data(data_name=data_name,
                        examples=examples)

    print(dataset)

