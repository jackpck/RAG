import os
from langsmith import Client
from src.components.runner import ChainRunner

if __name__ == "__main__":
    os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"].rstrip()
    os.environ["LANGSMITH_API_KEY"] = os.environ["LANGSMITH_API_KEY"].rstrip()
    os.environ["LANGSMITH_WORKSPACE_ID"] = os.environ["LANGSMITH_WORKSPACE_ID"].rstrip()
    os.environ["LANGSMITH_ENDPOINT"] = os.environ["LANGSMITH_ENDPOINT"].rstrip()
    os.environ["LANGSMITH_PROJECT"] = os.environ["LANGSMITH_PROJECT"].rstrip()
    os.environ["LANGSMITH_TRACING"] = os.environ["LANGSMITH_TRACING"].rstrip()
    os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"].rstrip()

    client = Client()
    USER_QUERY_PATH = "./src/user_query/user_query.txt"
    CONFIG_PATH = "./configs/pipeline_config.yaml"

    RAG_chain = ChainRunner(config_path=CONFIG_PATH)
    response = RAG_chain.run(USER_QUERY_PATH)
    print(f"response:\n{response.content}")

