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


if __name__ == "__main__":
    import os

    os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"].rstrip()
    os.environ["LANGSMITH_API_KEY"] = os.environ["LANGSMITH_API_KEY"].rstrip()
    os.environ["LANGSMITH_WORKSPACE_ID"] = os.environ["LANGSMITH_WORKSPACE_ID"].rstrip()
    os.environ["LANGSMITH_ENDPOINT"] = os.environ["LANGSMITH_ENDPOINT"].rstrip()
    os.environ["LANGSMITH_PROJECT"] = os.environ["LANGSMITH_PROJECT"].rstrip()
    os.environ["LANGSMITH_TRACING"] = os.environ["LANGSMITH_TRACING"].rstrip()
    os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"].rstrip()

    model = "gemini-2.5-flash"
    model_provider = "google_genai"
    temperature = 0
    top_k = 10
    top_p = 0.9
    prompt_name = "system-system-prompt"
    prompt_version = "latest"

    generator = GeneratorLLM(model=model,
                             model_provider=model_provider,
                             temperature=temperature,
                             top_k=top_k,
                             top_p=top_p,
                             prompt_name=prompt_name,
                             prompt_version=prompt_version)

    query = {"question":"Where is Moscow?",
             "context":"Moscow is the capital of Russia and St Petersburg is another big city in Russia."}

    response = generator.llm.invoke(query)
    print(response)