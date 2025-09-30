from langsmith import Client
import os
import json

from src.components.runner import ChainRunner
from src.utils.validation import LLMJudge
from src.utils.langsmith_loader import load_data

os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"].rstrip()
os.environ["LANGSMITH_API_KEY"] = os.environ["LANGSMITH_API_KEY"].rstrip()
os.environ["LANGSMITH_WORKSPACE_ID"] = os.environ["LANGSMITH_WORKSPACE_ID"].rstrip()
os.environ["LANGSMITH_ENDPOINT"] = os.environ["LANGSMITH_ENDPOINT"].rstrip()
os.environ["LANGSMITH_PROJECT"] = os.environ["LANGSMITH_PROJECT"].rstrip()
os.environ["LANGSMITH_TRACING"] = os.environ["LANGSMITH_TRACING"].rstrip()
os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"].rstrip()

if __name__ == "__main__":

    EVAL_DATA_PATH = "./data/evaluation/eval_examples.json"
    CONFIG_PATH = "./configs/pipeline_config.yaml"
    data_name = "rag_test"
    judge_prompt_name = "system-llmjudge-prompt"
    judge_prompt_version = "latest"
    model = "gemini-2.5-flash"
    model_provider = "google_genai"
    temperature = 0
    top_k = 10
    top_p = 0.9

    client = Client()

    # Load data
    with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
        data = f.read()
    examples = json.loads(data)["examples"]

    # Run evaluation
    RAG_chain = ChainRunner(config_path=CONFIG_PATH)
    LLM_judge = LLMJudge(model=model,
                         model_provider=model_provider,
                         temperature=temperature,
                         top_k=top_k,
                         top_p=top_p,
                         prompt_name=judge_prompt_name,
                         prompt_version=judge_prompt_version)

    def run_inference(inputs: dict) -> dict:
        response = RAG_chain.run(inputs["text"]).content
        return {"response": response}

    results = client.evaluate(
        run_inference,
        data=dataset.name,
        evaluators=[LLM_judge.accuracy_metric],
    )