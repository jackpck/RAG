import yaml
import importlib

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
    runner = PipelineRunner("../configs/pipeline_config.yaml")
    result = runner.run()
    print("Answer: \n", result["result"])
    print("n\Citations:")
    for doc in result["source_documents"]:
        print(f"- Content: {doc.page_content.strip()}")