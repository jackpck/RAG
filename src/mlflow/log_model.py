import mlflow
import json
import subprocess

subprocess.run(["python", "-m", "src.mlflow.set_model"])

chain_path = "./src/set_model.py"
USER_QUERY_PATH = "./src/user_query/user_query.txt"
PROMPT_METADATA_PATH = "./src/prompts/prompt_metadata.json"
CONFIG_PATH = "./configs/inference_pipeline_config.yaml"  # use this for inference
model_name = "RAG"
code_paths = ["src/components", "src/utils"]
prompt_version = 1

with open(USER_QUERY_PATH, "r", encoding="utf-8") as f:
    user_query = f.read()

with open(PROMPT_METADATA_PATH, "r", encoding="utf-8") as f:
    data = f.read()
prompt_metadata_dict = json.loads(data)

prompt_list = [f"prompts:/{prompt_name}/{prompt_version}"
               for prompt_name, _ in prompt_metadata_dict.items()]

with mlflow.start_run():
    info = mlflow.langchain.log_model(
        lc_model=chain_path,
        name=model_name,
        input_example=user_query,
        code_paths=code_paths,
        prompts=prompt_list,
    )
    mlflow.log_artifact(CONFIG_PATH)
