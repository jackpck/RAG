import mlflow

chain_path = "./src/set_model.py"
USER_QUERY_PATH = "./src/user_query/user_query.txt"
CONFIG_PATH = "./configs/inference_pipeline_config.yaml"  # use this for inference
model_name = "RAG"

with open(USER_QUERY_PATH, "r", encoding="utf-8") as f:
    user_query = f.read()

with mlflow.start_run():
    info = mlflow.langchain.log_model(
        lc_model=chain_path,
        name=model_name,
        input_example=user_query
    )
    mlflow.log_artifact(CONFIG_PATH)
