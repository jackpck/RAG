from langchain_core.runnables.base import RunnableSerializable
from src.utils.chainer import chain_from_yaml

class ChainRunner:
    def __init__(self, config_path: str):
        self._load_chain(config_path=config_path)

    def _load_chain(self, config_path: str) -> RunnableSerializable:
        self.rag_chain = chain_from_yaml(config_path)

    def run(self, user_query: str) -> str:
        response = self.rag_chain.invoke(user_query)

        return response

