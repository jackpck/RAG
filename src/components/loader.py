from langchain.document_loaders import TextLoader
from langchain_core.documents import Document
from typing import List

class DataLoader:
    def __init__(self, path: str):
        self.path = path

    def load(self) -> List[Document]:
        return TextLoader(self.path).load()