from langchain.document_loaders import TextLoader
from langchain_core.documents import Document
from typing import List

class DataLoader:
    def __init__(self, path: str):
        self.path = path

    def load(self, metadata: dict) -> List[Document]:
        docs = TextLoader(self.path).load()
        for doc in docs:
            doc.metadata["source"] = metadata["source"]
        return docs