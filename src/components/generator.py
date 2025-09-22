from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

from src.utils.langsmith_loader import load_prompt

class GeneratorLLM:
    def __init__(self,
                 model: str,
                 model_provider: str,
                 temperature: float,
                 top_k: int,
                 top_p: float,
                 prompt_name: str,
                 prompt_version: str):
        self._llm = init_chat_model(model=model,
                                    model_provider=model_provider,
                                    temperature=temperature,
                                    top_k=top_k,
                                    top_p=top_p)
        self._setup_prompt(prompt_name, prompt_version)

    def _setup_prompt(self, prompt_name, prompt_version):
        prompt = load_prompt(prompt_name, prompt_version)
        system_prompt = prompt.format_messages()[0].content
        self.system_prompt = ChatPromptTemplate.from_template(system_prompt)

    @property
    def llm(self):
        return self.system_prompt | self._llm
