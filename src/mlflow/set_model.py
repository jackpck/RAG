import os
from src.components.runner import ChainRunner
import mlflow

os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"].rstrip()

CONFIG_PATH = "./configs/inference_pipeline_config.yaml"  # use this for inference

RAG_chain = ChainRunner(config_path=CONFIG_PATH)

# save model as code for logging in mlflow
mlflow.models.set_model(RAG_chain.rag_chain)

