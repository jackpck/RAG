import yaml
import importlib
import os
from langsmith import Client

os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"].rstrip()
os.environ["LANGSMITH_API_KEY"] = os.environ["LANGSMITH_API_KEY"].rstrip()
os.environ["LANGSMITH_WORKSPACE_ID"] = os.environ["LANGSMITH_WORKSPACE_ID"].rstrip()
os.environ["LANGSMITH_ENDPOINT"] = os.environ["LANGSMITH_ENDPOINT"].rstrip()
os.environ["LANGSMITH_PROJECT"] = os.environ["LANGSMITH_PROJECT"].rstrip()
os.environ["LANGSMITH_TRACING"] = os.environ["LANGSMITH_TRACING"].rstrip()
os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"].rstrip()

client = Client()

RAG_metadata = {}
class PipelineRunner:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.context = {}

    def run (self):
        for step in self.config["pipeline"]:
            cls_path = step["class"]
            method_name = step["method"]
            params = step.get("params", {})
            inputs = step.get("input", {})
            outputs = step.get("output", {})

            module_name, class_name = cls_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            if params:
                new_params = {k: self.context[k] if k in self.context else v
                              for k, v in params.items()}
                instance = cls(**new_params)
            else:
                instance = cls()

            if method_name:
                method = getattr(instance, method_name)
                args = {k: self.context[k] if k in self.context else v
                        for k, v in inputs.items()}
                result = method(**args) if args else method()
            else:
                result = instance # if method is empty in yaml, output will be the instance itself

            if isinstance(outputs, dict) and outputs:
                output_key = list(outputs.values())[0]
                self.context[output_key] = result
            elif not outputs:
                continue
            else:
                self.context[outputs] = result

        return self.context['response']

if __name__ == "__main__":
    USER_QUERY_PATH = "./src/user_query/user_query.txt"
    with open(USER_QUERY_PATH, "r", encoding="utf-8") as f:
        user_query = f.read()
    runner = PipelineRunner("./configs/pipeline_config.yaml")
    result = runner.run()
    print(f"User query: \n{user_query}")
    print(f"Answer: \n{result.content}")
    #print(f"Citations: \n{result['sources']}")