import os
import mlflow
import subprocess
import json
import sys
from mlflow.genai import scorer

from src.utils.mlflow_loader import load_data, load_prompt
from src.components.runner import ChainRunner
from src.utils.validation import LLMJudge

os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"].rstrip()

CONFIG_PATH = "./configs/inference_pipeline_config.yaml"  # use this for inference
EVAL_DATA_PATH = "./data/evaluation/eval_examples_test.json"
USER_QUERY_PATH = "./src/user_query/user_query.txt"
PROMPT_METADATA_PATH = "./src/prompts/prompt_metadata.json"

chain_path = "./src/mlflow/set_model.py"
model_name = "RAG"
code_paths = ["src/components", "src/utils"]

commit_message = "initial commit"
version_metadata = {
    "author": "jackypck93@gmail.com",
}

data_name = "eval_examples_test"
experiment_name = "first-experiment"
experiment_description = "first-experiment"
experiment_tags = {"mlflow.note.content": experiment_description}
#run_name = "first-run"

judge_params = {
    "judge_prompt_name": "LLMJUDGE_PROMPT",
    "judge_prompt_version": "latest",
    "judge_model": "gemini-2.5-flash",
    "judge_model_provider": "google_genai",
    "judge_temperature": 0,
    "judge_top_k": 10,
    "judge_top_p": 0.9
}

with open(USER_QUERY_PATH, "r", encoding="utf-8") as f:
    user_query = f.read()

with open(PROMPT_METADATA_PATH, "r", encoding="utf-8") as f:
    data = f.read()
prompt_metadata_dict = json.loads(data)

# check experiment
if (experiment := mlflow.get_experiment_by_name(name=experiment_name)):
    experiment_id = experiment.experiment_id
    print(f"experiment exists. get experiment_id")
else:
    # create experiment
    experiment_id = mlflow.create_experiment(
        name=experiment_name,
        tags=experiment_tags,
    )
    print(f"experiment doesn't exist. new experiment created")

# Load prompts outside of run to property load from mlflow
# If need to load within run, do: mlflow.set_tracking_uri("sqlite:///mlflow.db")
judge_prompt = load_prompt(prompt_name=judge_params["judge_prompt_name"],
                           prompt_version=judge_params["judge_prompt_version"])
judge_prompt_str = judge_prompt.template
print(f"load prompt")

with mlflow.start_run(experiment_id=experiment_id) as run:
    print(f"Experiment ID: {run.info.experiment_id}\nRun ID: {run.info.run_id}")

    mlflow.log_artifact(local_path=CONFIG_PATH)
    print(f"log artifact")

    # log LLM judge params
    mlflow.log_params(params=judge_params)
    print(f"log LLM judge params")

    # Load evaluation examples
    with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
        data = f.read()
    examples = json.loads(data)["examples"]

    mlflow.log_artifact(local_path=EVAL_DATA_PATH)
    print(f"load data")

    # Run evaluation
    RAG_chain = ChainRunner(config_path=CONFIG_PATH)
    def run_inference(text: str) -> dict:
        response = RAG_chain.run(text)["result"]
        return {"response": response}
    print(f"instantiate RAG_chain")

    LLM_judge = LLMJudge(model=judge_params["judge_model"],
                         model_provider=judge_params["judge_model_provider"],
                         temperature=judge_params["judge_temperature"],
                         top_k=judge_params["judge_top_k"],
                         top_p=judge_params["judge_top_p"],
                         judge_prompt=judge_prompt_str)
    print(f"load LLM judge")

    @scorer
    def accuracy_metric(outputs: dict,
                        expectations: dict):
        return LLM_judge.accuracy_metric(outputs=outputs,
                                         expectations=expectations,
                                         llm_judge=LLM_judge.llm_judge)

    results = mlflow.genai.evaluate(
        data=examples,
        predict_fn=run_inference,
        scorers=[accuracy_metric],
    )
    print(f"run evaluation")

    subprocess.run([sys.executable, "-m", "src.mlflow.set_model"])
    print(f"set model")

    prompt_list = [f"prompts:/{judge_params['judge_prompt_name']}@{judge_params['judge_prompt_version']}"
                   for prompt_name, _ in prompt_metadata_dict.items()]

    info = mlflow.langchain.log_model(
        lc_model=chain_path,
        name=model_name,
        input_example=user_query,
        code_paths=code_paths,
        prompts=prompt_list,
    )
    print(f"log model")
