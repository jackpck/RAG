from langchain.document_loaders import TextLoader
from langchain_core.documents import Document
from typing import List
from src.utils.wikipedia_api_call import WikipediaContent

class DataLoader:
    def __init__(self, metadata: dict):
        self.metadata = metadata

    def load_from_text(self, path: str) -> List[Document]:
        '''
        :param path: path to txt for the body of text from ../data
        e.g. path = "../data/battle_of_stalingrad_wiki_lite.txt"
        '''
        docs = TextLoader(path).load()
        for doc in docs:
            doc.metadata["source"] = self.metadata["source"]
        return docs

    def load_from_wikipedia_api(self, title: str) -> List[Document]:
        '''
        :param title: what you might search for on wikipedia page
        e.g. title = "Battle of Stalingrad"
        '''
        wiki_content = WikipediaContent(title)
        wiki_text = wiki_content.get_content()
        docs = [Document(page_content=wiki_text)]
        for doc in docs:
            doc.metadata["source"] = self.metadata["source"]
        return docs

if __name__ == "__main__":
    metadata = {"source": "Battle of Stalingrad"}
    title = metadata["source"]
    loader = DataLoader(metadata=metadata)
    doc = loader.load_from_wikipedia_api(title=title)

    print(doc)
