from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
import os
from langchain.chat_models import init_chat_model
import argparse
import json

from src.prompts import prompts

os.environ["LANGSMITH_API_KEY"] = os.environ["LANGSMITH_API_KEY"].rstrip()
os.environ["LANGSMITH_WORKSPACE_ID"] = os.environ["LANGSMITH_WORKSPACE_ID"].rstrip()
os.environ["LANGSMITH_ENDPOINT"] = os.environ["LANGSMITH_ENDPOINT"].rstrip()
os.environ["LANGSMITH_PROJECT"] = os.environ["LANGSMITH_PROJECT"].rstrip()
os.environ["LANGSMITH_TRACING"] = os.environ["LANGSMITH_TRACING"].rstrip()
os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"].rstrip()

parser = argparse.ArgumentParser()
parser.add_argument("--action", type=str, required=True)
args = parser.parse_args()

action = args.action

PROMPT_METADATA_PATH = "./src/prompts/prompt_metadata.json"
with open(PROMPT_METADATA_PATH, "r", encoding="utf-8") as f:
    data = f.read()
prompt_metadata_dict = json.loads(data)

client = Client()

if action == "push":
    for prompt_name, prompt_metadata in prompt_metadata_dict.items():
        prompt_identifier = prompt_metadata["prompt_identifier"]
        prompt_type = prompt_identifier.split("-")[0]
        if prompt_type == "system":
            message = SystemMessage(content=getattr(prompts, prompt_name))
        elif prompt_type == "user":
            message = HumanMessage(content=getattr(prompts, prompt_name))
        else:
            raise ValueError("prompt_type must be 'system' or 'user'")
        prompt = ChatPromptTemplate.from_messages([message])
        try:
            client.push_prompt(prompt_identifier=prompt_identifier,
                               object=prompt,
                               description=prompt_metadata["description"],
                               tags=prompt_metadata["tag_list"])
        except:
            continue
elif action == "pull":
    # below is a demonstration of passing the pulled prompt to an LLM

    prompt_list = client.list_prompts(is_public=False)
    prompt_dict = {}
    for p in prompt_list.repos:
        prompt_name = f"{p.repo_handle}:{p.last_commit_hash[:8]}"
        prompt_dict[p.description] = client.pull_prompt(prompt_name)

    print({k: v.format_messages()[0].content for k,v in prompt_dict.items()})
else:
    print("action must be either 'push' or 'pull'")
