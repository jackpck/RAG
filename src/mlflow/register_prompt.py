import mlflow
import json
import argparse

from src.prompts import prompts

parser = argparse.ArgumentParser()
parser.add_argument("--commit_msg", type=str, required=True)
args = parser.parse_args()

commit_message = args.commit_msg
version_metadata = {
    "author": "jackypck93@gmail.com",
}

PROMPT_METADATA_PATH = "./src/prompts/prompt_metadata.json"
with open(PROMPT_METADATA_PATH, "r", encoding="utf-8") as f:
    data = f.read()
prompt_metadata_dict = json.loads(data)

#mlflow.set_tracking_uri("sqlite:///mlflow.db")
for prompt_name, prompt_metadata in prompt_metadata_dict.items():
    prompt = mlflow.genai.register_prompt(
        name=prompt_name,
        template=getattr(prompts, prompt_name),
        commit_message=commit_message,
        #tags=prompt_metadata["tag_list"],
    )
