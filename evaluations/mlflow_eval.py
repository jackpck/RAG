import os
import mlflow
import json
import sys
from mlflow.genai import scorer

from src.utils.langsmith_loader import load_prompt
from src.components.runner import ChainRunner
from src.utils.validation import LLMJudge
import configs.experiment_config as config

os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"].rstrip()

with open(config.USER_QUERY_PATH, "r", encoding="utf-8") as f:
    user_query = f.read()

with open(config.PROMPT_METADATA_PATH, "r", encoding="utf-8") as f:
    data = f.read()
prompt_metadata_dict = json.loads(data)

print(f"LOAD PROMPT")
judge_prompt = load_prompt(prompt_name="system-llmjudge-prompt",
                           prompt_version="latest")
judge_prompt_str = judge_prompt.format_messages()[0].content

# Load evaluation examples
print(f"LOAD AND LOG EVAL DATA")
with open(config.EVAL_DATA_PATH, "r", encoding="utf-8") as f:
    data = f.read()
examples = json.loads(data)["examples"]

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

print(f"RUN EVAL")
results = mlflow.genai.evaluate(
    data=examples,
    predict_fn=run_inference,
    scorers=[accuracy_metric],
)

