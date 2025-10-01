import os
from src.components.runner import ChainRunner
import mlflow
import time

if __name__ == "__main__":
    starttime = time.time()

    os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"].rstrip()

    mlflow.config.enable_async_logging()
    mlflow.set_experiment("first pdf RAG")
    mlflow.langchain.autolog()

    USER_QUERY_PATH = "./src/user_query/user_query.txt"
    # CONFIG_PATH = "./configs/pipeline_config.yaml" # use this for experimentation
    CONFIG_PATH = "./configs/inference_pipeline_config.yaml" # use this for inference

    with open(USER_QUERY_PATH, "r", encoding="utf-8") as f:
        user_query = f.read()

    RAG_chain = ChainRunner(config_path=CONFIG_PATH)
    response = RAG_chain.run(user_query)
    print(f"user query:\n{user_query}")
    print("*"*40)
    print(f"response:\n{response['result'].content}")
    print("*"*40)
    print(f"citation:")
    for i, citation in enumerate(response['citation']):
        print(f"#{i} \n{citation}")

    # save model as code for logging in mlflow
    # mlflow.models.set_model(RAG_chain.rag_chain)
    mlflow.flush_trace_async_logging()

    endtime = time.time()
    print(f"latency: {endtime - starttime:.4f} seconds")

