import yaml
import importlib

def run (config_path: str,
         start_from_step: str,
         stop_at_step: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    context = {}

    start_step = [x["name"] for x in config["pipeline"]].index(start_from_step)
    stop_step = [x["name"] for x in config["pipeline"]].index(stop_at_step)
    pipeline_steps = config["pipeline"][start_step:stop_step+1]
    for step in pipeline_steps:
        cls_path = step["class"]
        method_name = step["method"]
        params = step.get("params", {})
        inputs = step.get("input", {})
        outputs = step.get("output", {})

        module_name, class_name = cls_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        if params:
            new_params = {k: context[k] if k in context else v
                          for k, v in params.items()}
            instance = cls(**new_params)
        else:
            instance = cls()

        if method_name:
            method = getattr(instance, method_name)
            args = {k: context[k] if k in context else v
                    for k, v in inputs.items()}
            result = method(**args) if args else method()
        else:
            result = instance # if method is empty in yaml, output will be the instance itself

        if isinstance(outputs, dict) and outputs:
            output_key = list(outputs.values())[0]
            context[output_key] = result
        elif not outputs:
            continue
        else:
            context[outputs] = result

    return context

if __name__ == "__main__":
    config_path = "./configs/pipeline_config.yaml"
    start_from_step = "load_data"
    stop_at_step = "retrieving"

    response = run(config_path=config_path,
                   start_from_step=start_from_step,
                   stop_at_step=stop_at_step)
    print(type(response["retriever"]))