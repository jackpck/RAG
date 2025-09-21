import os
from langsmith import Client
from src.components.runner import ChainRunner
from src.components.chainer import ComponentChainer

if __name__ == "__main__":
    os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"].rstrip()
    os.environ["LANGSMITH_API_KEY"] = os.environ["LANGSMITH_API_KEY"].rstrip()
    os.environ["LANGSMITH_WORKSPACE_ID"] = os.environ["LANGSMITH_WORKSPACE_ID"].rstrip()
    os.environ["LANGSMITH_ENDPOINT"] = os.environ["LANGSMITH_ENDPOINT"].rstrip()
    os.environ["LANGSMITH_PROJECT"] = os.environ["LANGSMITH_PROJECT"].rstrip()
    os.environ["LANGSMITH_TRACING"] = os.environ["LANGSMITH_TRACING"].rstrip()
    os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"].rstrip()

    #client = Client()
    USER_QUERY_PATH = "./src/user_query/user_query.txt"
    with open(USER_QUERY_PATH, "r", encoding="utf-8") as f:
        user_query = f.read()

    model = "gemini-2.5-flash"
    model_provider = "google_genai"
    temperature = 0
    top_k = 10
    top_p = 0.8
    from_vs = True

    SYSTEM_PROMPT = "./src/prompts/system_prompt.txt"
    RERANKER_PROMPT = "./src/prompts/reranker_prompt.txt"

    rag_chain = ComponentChainer(model=model,
                                 model_provider=model_provider,
                                 temperature=temperature,
                                 top_k=top_k,
                                 top_p=top_p).chain(SYSTEM_PROMPT,
                                                    RERANKER_PROMPT)
    runner = ChainRunner()
    response = runner.run(rag_chain, USER_QUERY_PATH)
    print(f"response:\n{response.content}")


    #runner = PipelineRunner("./configs/pipeline_config.yaml")
    #result = runner.run(from_vs=from_vs)
    #print(f"User query: \n{user_query}")
    #print(f"Answer: \n{result.content}")
    #print(f"Citations: \n{result['sources']}")