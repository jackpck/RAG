from langsmith import Client
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langsmith.utils import LangSmithError
from langchain_core.prompts.chat import ChatPromptTemplate
import os
import json
import copy

from src.utils.langsmith_loader import load_prompt
from src.utils.validation import LLMJudge

os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"].rstrip()
os.environ["LANGSMITH_API_KEY"] = os.environ["LANGSMITH_API_KEY"].rstrip()
os.environ["LANGSMITH_WORKSPACE_ID"] = os.environ["LANGSMITH_WORKSPACE_ID"].rstrip()
os.environ["LANGSMITH_ENDPOINT"] = os.environ["LANGSMITH_ENDPOINT"].rstrip()
os.environ["LANGSMITH_PROJECT"] = os.environ["LANGSMITH_PROJECT"].rstrip()
os.environ["LANGSMITH_TRACING"] = os.environ["LANGSMITH_TRACING"].rstrip()
os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"].rstrip()

if __name__ == "__main__":

    client = Client()

    # Load data
    with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
        data = f.read()

    # Load prompts
    prompt_dict = load_prompt(client=client)

    # Run evaluation
    RAG_chain = ChainRunner(config_path=CONFIG_PATH)
    LLM_judge = LLMJudge(model=model,
                         model_provider=model_provider,
                         temperature=temperature,
                         top_k=top_k,
                         top_p=top_p)

    results = client.evaluate(
        target=RAG_chain.run,
        data=dataset.name,
        evaluators=[LLM_judge.accuracy_metric],
    )