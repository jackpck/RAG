import os
from langsmith import Client
from src.components.runner import ChainRunner
import mlflow

if __name__ == "__main__":
    os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"].rstrip()
    #os.environ["LANGSMITH_API_KEY"] = os.environ["LANGSMITH_API_KEY"].rstrip()
    #os.environ["LANGSMITH_WORKSPACE_ID"] = os.environ["LANGSMITH_WORKSPACE_ID"].rstrip()
    #os.environ["LANGSMITH_ENDPOINT"] = os.environ["LANGSMITH_ENDPOINT"].rstrip()
    #os.environ["LANGSMITH_PROJECT"] = os.environ["LANGSMITH_PROJECT"].rstrip()
    #os.environ["LANGSMITH_TRACING"] = os.environ["LANGSMITH_TRACING"].rstrip()
    #os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"].rstrip()

    #client = Client()
    mlflow.set_experiment("RAG")
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
    print(f"response:\n{response['result']}")
    print("*"*40)
    print(f"citation:")
    for i, citation in enumerate(response['citation']):
        print(f"#{i} \n{citation.page_content}")

    # save model as code for logging in mlflow
    # mlflow.models.set_model(RAG_chain.rag_chain)

