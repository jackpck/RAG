import mlflow
from mlflow.genai import datasets
from mlflow.data.evaluation_dataset import EvaluationDataset

def load_prompt(prompt_name: str,
                prompt_version: str | int):
    if prompt_version == "latest":
        return mlflow.genai.load_prompt(f"prompts:/{prompt_name}@{prompt_version}")
    elif isinstance(prompt_version, int):
        return mlflow.genai.load_prompt(f"prompts:/{prompt_name}/{prompt_version}")


def load_data(name: str,
              experiment_id: list[str],
              tags: dict[str],
              test_cases: dict) -> EvaluationDataset:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    dataset = datasets.create_dataset(name=name,
                             experiment_id=experiment_id,
                             tags=tags
                             )
    return dataset.merge_records(test_cases)

if __name__ == "__main__":
    import json

    #prompt = load_prompt("LLMJUDGE_PROMPT","latest")
    #print(prompt.template)

    EVAL_DATA_PATH = "./data/evaluation/eval_examples_test.json"
    data_name = "eval_examples_test"
    with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
        data = f.read()
    examples = json.loads(data)["examples"]

    experiment_id = 0
    experiment_tags = {}
    dataset = load_data(name=data_name,
                        experiment_id=experiment_id,
                        tags=experiment_tags,
                        test_cases=examples)
    print(dataset.records)
