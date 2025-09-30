import os
import mlflow
import subprocess
import json
import sys
from mlflow.genai import scorer
from mlflow.genai.scorers import RetrievalGroundedness, RelevanceToQuery

from src.utils.mlflow_loader import load_data, load_prompt
from src.components.runner import ChainRunner
from src.utils.validation import LLMJudge
import configs.experiment_config as config

os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"].rstrip()

# mlflow.langchain.autolog() # to enable tracing. Seems to only work with 0.0.354 <= langchain <= 0.3.25

with open(config.USER_QUERY_PATH, "r", encoding="utf-8") as f:
    user_query = f.read()

with open(config.PROMPT_METADATA_PATH, "r", encoding="utf-8") as f:
    data = f.read()
prompt_metadata_dict = json.loads(data)

# check experiment
print(f"SET UP EXPERIMENT")
if (experiment := mlflow.get_experiment_by_name(name=config.experiment_name)):
    experiment_id = experiment.experiment_id
    print(f"experiment exists. get experiment_id")
else:
    # create experiment
    experiment_id = mlflow.create_experiment(
        name=config.experiment_name,
        tags=config.experiment_tags,
    )
    print(f"experiment doesn't exist. new experiment created")

# Load prompts outside of run to property load from mlflow
# If need to load within run, do: mlflow.set_tracking_uri("sqlite:///mlflow.db")
print(f"LOAD PROMPT")
judge_prompt = load_prompt(prompt_name=config.judge_params["judge_prompt_name"],
                           prompt_version=config.judge_params["judge_prompt_version"])
judge_prompt_str = judge_prompt.template

print(f"START RUN")
with mlflow.start_run(experiment_id=experiment_id) as run:
    print(f"Experiment ID: {run.info.experiment_id}\nRun ID: {run.info.run_id}")

    print(f"LOG ARTIFACT")
    mlflow.log_artifact(local_path=config.CONFIG_PATH)

    # log LLM judge params
    print(f"LOG LLM JUDGE PARAMS")
    mlflow.log_params(params=config.judge_params)

    # Load evaluation examples
    # mlflow.genai.create_dataset only work with sql db.
    # set_tracking_uri will mess up the experiment run. To be investigate
    print(f"LOAD AND LOG EVAL DATA")
    with open(config.EVAL_DATA_PATH, "r", encoding="utf-8") as f:
        data = f.read()
    examples = json.loads(data)["examples"]
    mlflow.log_artifact(local_path=config.EVAL_DATA_PATH)

    # Run evaluation
    print(f"INSTANTIATE RAG_chain")
    RAG_chain = ChainRunner(config_path=config.CONFIG_PATH)
    def run_inference(text: str) -> dict:
        response = RAG_chain.run(text)["result"]
        return {"response": response}

    print(f"LOAD LLM JUDGE")
    LLM_judge = LLMJudge(model=config.judge_params["judge_model"],
                         model_provider=config.judge_params["judge_model_provider"],
                         temperature=config.judge_params["judge_temperature"],
                         top_k=config.judge_params["judge_top_k"],
                         top_p=config.judge_params["judge_top_p"],
                         judge_prompt=judge_prompt_str)

    print(f"DEFINE METRICS")
    @scorer
    def accuracy_metric(outputs: dict,
                        expectations: dict):
        return LLM_judge.accuracy_metric(outputs=outputs,
                                         expectations=expectations,
                                         llm_judge=LLM_judge.llm_judge)

    #print(f"GET TRACE")
    #traces = mlflow.search_traces(experiment_ids=[experiment_id],
    #                              run_id=run.info.run_id)
    #print(f"traces:\n{traces}")

    print(f"RUN EVAL")
    #results = mlflow.genai.evaluate(
    #    data=traces,
    #    scorers=[RetrievalGroundedness()],
    #)

    ## uncomment below if only using accuracy_metric
    results = mlflow.genai.evaluate(
        data=examples,
        predict_fn=run_inference,
        scorers=[accuracy_metric],
    )

    print(f"evaluation results:\n{results}")

    print(f"SET MODEL")
    # save model as code as LLM is not serializable
    subprocess.run([sys.executable, "-m", "src.mlflow.set_model"])

    print(f"LOG MODEL")
    prompt_list = [f"prompts:/{config.judge_params['judge_prompt_name']}@{config.judge_params['judge_prompt_version']}"
                   for prompt_name, _ in prompt_metadata_dict.items()]

    info = mlflow.langchain.log_model(
        lc_model=config.chain_path,
        name=config.model_name,
        input_example=user_query,
        code_paths=config.code_paths,
        prompts=prompt_list,
    )
