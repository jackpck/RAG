from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

class GeneratorLLM:
    def __init__(self,
                 model: str,
                 model_provider: str,
                 temperature: float,
                 top_k: int,
                 top_p: float,
                 system_prompt_path: str):
        self._llm = init_chat_model(model=model,
                                    model_provider=model_provider,
                                    temperature=temperature,
                                    top_k=top_k,
                                    top_p=top_p)
        self._setup_prompt(system_prompt_path=system_prompt_path)

    def _setup_prompt(self, system_prompt_path):
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            system_prompt_temp = f.read()
        self.system_prompt = ChatPromptTemplate.from_template(system_prompt_temp)

    @property
    def llm(self):
        return self.system_prompt | self._llm
