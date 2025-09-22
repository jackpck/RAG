import yaml
import importlib
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langsmith import Client

def chain_from_yaml(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    def get_arg(x, component_outputs, inputs):
        args = {}
        for k, v in inputs.items():
            if k in component_outputs:
                args[k] = x["context"]
            else:
                if v:  # first component, input (title) is given
                    args[k] = v
                else:  # reranker component, input (query) is empty
                    args[k] = x["question"]
        return args

    component_outputs = set()
    component_runnables = []
    for i, step in enumerate(config["pipeline"]):
        step_name = step["name"]
        cls_path = step["class"]
        method_name = step["method"]
        params = step.get("params", {})
        inputs = step.get("input", {})
        outputs = step.get("output", {})

        module_name, class_name = cls_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        if params:
            instance = cls(**params)
        else:
            instance = cls()

        method = getattr(instance, method_name)
        if outputs:
            component_lambda = lambda x, i=i, inputs=inputs, component_outputs=component_outputs, method=method: \
                {"question": x if i == 0 else x["question"],
                 "context": method(**get_arg(x=x,
                                             component_outputs=component_outputs,
                                             inputs=inputs))}
            component_runnables.append(RunnableLambda(component_lambda).with_config(run_name=step_name,
                                                                                    metadata=params))
            if isinstance(outputs, dict):
                output_key = list(outputs.values())[0]
                component_outputs.add(output_key)
            else:
                component_outputs.add(outputs)
        else: # last step, call system_prompt | llm
            component_runnables.append(method.with_config(run_name=step_name,
                                                            metadata=params))

    return RunnableSequence(*component_runnables)


if __name__ == "__main__":
    import os
    os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"].rstrip()
    os.environ["LANGSMITH_API_KEY"] = os.environ["LANGSMITH_API_KEY"].rstrip()
    os.environ["LANGSMITH_WORKSPACE_ID"] = os.environ["LANGSMITH_WORKSPACE_ID"].rstrip()
    os.environ["LANGSMITH_ENDPOINT"] = os.environ["LANGSMITH_ENDPOINT"].rstrip()
    os.environ["LANGSMITH_PROJECT"] = os.environ["LANGSMITH_PROJECT"].rstrip()
    os.environ["LANGSMITH_TRACING"] = os.environ["LANGSMITH_TRACING"].rstrip()
    os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"].rstrip()
    config_path = "./configs/pipeline_config.yaml"
    user_query_path = "./src/user_query/user_query.txt"

    client = Client()

    with open(user_query_path, "r", encoding="utf-8") as f:
        user_query = f.read()

    runnable_seq = chain_from_yaml(config_path=config_path)
    print(f"runnable_seq:\n{runnable_seq}")
    print("Invoke")
    response = runnable_seq.invoke(user_query)
    print(f"response:\n{response}")
