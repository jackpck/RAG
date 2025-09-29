CONFIG_PATH = "./configs/inference_pipeline_config.yaml"  # use this for inference
EVAL_DATA_PATH = "./data/evaluation/eval_examples.json"
USER_QUERY_PATH = "./src/user_query/user_query.txt"
PROMPT_METADATA_PATH = "./src/prompts/prompt_metadata.json"

chain_path = "./src/mlflow/set_model.py"
model_name = "RAG"
code_paths = ["src/components", "src/utils"]

#commit_message = "initial commit"
#version_metadata = {
#    "author": "jackypck93@gmail.com",
#}

data_name = "eval_examples_test"
experiment_name = "first-experiment"
experiment_description = "first-experiment"
experiment_tags = {"mlflow.note.content": experiment_description}
#run_name = "first-run"

judge_params = {
    "judge_prompt_name": "LLMJUDGE_PROMPT",
    "judge_prompt_version": "latest",
    "judge_model": "gemini-2.5-flash",
    "judge_model_provider": "google_genai",
    "judge_temperature": 0,
    "judge_top_k": 10,
    "judge_top_p": 0.9
}
