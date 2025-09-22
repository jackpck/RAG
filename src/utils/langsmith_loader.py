from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate

def load_prompt(client: Client) -> dict[ChatPromptTemplate]:
    prompt_list = client.list_prompts(is_public=False)
    prompt_dict = {}
    for p in prompt_list.repos:
        prompt_name = f"{p.repo_handle}:{p.last_commit_hash[:8]}"
        prompt_dict[p.description] = client.pull_prompt(prompt_name)

    return prompt_dict

def load_data(client: Client,
              data_name: str,
              examples: dict = None):
    try:
        dataset = client.read_data(dataset_name=data_name)
    except LangSmithError:
        dataset = client.create_dataset(dataset_name=data_name)
        client.create_examples(
            dataset_id=dataset.id,
            examples=examples
        )
        dataset = client.read_data(dataset_name=data_name)

    return dataset


