from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from typing import List
from src.utils.wikipedia_api_call import WikipediaContent
import pdfplumber

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

    def load_from_pdf(self, pdf_path: str, split_from_mid: bool) -> List[Document]:
        docs = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                if split_from_mid:
                    left_column = page.crop((0, 0, 0.5 * page.width, page.height))
                    right_column = page.crop((0.5 * page.width, 0, page.width, page.height))
                    docs.append(Document(page_content=left_column.extract_text() + right_column.extract_text()))
                else:
                    docs.append(Document(page_content=page.extract_text()))
        for page_num, doc in enumerate(docs):
            doc.metadata["source"] = page_num
        return docs

if __name__ == "__main__":
    #metadata = {"source": "Battle of Stalingrad"}
    #title = metadata["source"]
    #loader = DataLoader(metadata=metadata)
    #doc = loader.load_from_wikipedia_api(title=title)

    pdf_path = "data/pdfs/wp_generative_ai_risk_management_in_fs.pdf"
    loader = DataLoader(metadata={})
    doc = loader.load_from_pdf(pdf_path=pdf_path)
    print(doc[-2])

