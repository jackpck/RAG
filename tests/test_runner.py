import os

from src.components.runner import ChainRunner

os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"].rstrip()
CONFIG_PATH = "./configs/inference_pipeline_config.yaml" # use this for inference

def test_run_chain():
    user_query = "How Generative AI Works & Differences from Traditional AI?"

    RAG_chain = ChainRunner(config_path=CONFIG_PATH)
    response = RAG_chain.run(user_query)

    assert bool(response["result"])
    assert bool(response["citation"])



