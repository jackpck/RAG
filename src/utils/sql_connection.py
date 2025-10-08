import psycopg
from langchain_community.vectorstores import FAISS
import numpy as np

class NeonPostgres:
    def __init__(self, dbname: str,
                 user: str,
                 password: str,
                 host: str,
                 port: str,
                 sslmode: str):
        self._setup_connection(dbname=dbname,
                                user=user,
                                password=password,
                                host=host,
                                port=port,
                                sslmode=sslmode)

    def _setup_connection(self, dbname, user, password, host, port, sslmode):
        self.conn = psycopg.connect(dbname=dbname,
                                    user=user,
                                    password=password,
                                    host=host,
                                    port=port,
                                    sslmode=sslmode)
        self.cur = self.conn.cursor()
        return self.conn

    def create_embedding_table(self, tablename: str):
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {tablename} (
            id SERIAL PRIMARY KEY,
            document_id TEXT,
            content TEXT,
            metadata TEXT,
            embedding vector(384)
        );
        """
        self.cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        self.cur.execute(create_table_query)

    def create_log_table(self, tablename: str):
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {tablename} (
            id SERIAL PRIMARY KEY,
            datetime TEXT,
            user_query TEXT,
            response TEXT,
            user_feedback INTEGER
        );
        """
        self.cur.execute(create_table_query)

    def get_embedding_from_vectorstore(self, vectorstore: FAISS):
        doc_ids = []
        embeddings = []
        doc_metadata = []
        doc_content = []

        faiss_index = vectorstore.index
        ntotal = faiss_index.ntotal
        vectors = faiss_index.reconstruct_n(0, ntotal)
        for i, (_, doc_id) in enumerate(vectorstore.index_to_docstore_id.items()):
            doc = vectorstore.docstore.search(doc_id)
            doc_ids.append(doc.id)
            embeddings.append(vectors[i])
            doc_metadata.append(doc.metadata)
            doc_content.append(doc.page_content)

        return doc_ids, embeddings, doc_metadata, doc_content

    def insert_embedding_to_table(self, tablename: str,
                                        embeddings: list[np.ndarray],
                                        doc_ids: list,
                                        doc_content: list[str],
                                        doc_metadata: list[str]):
        for doc_id, emb, content, metadata in zip(doc_ids, embeddings, doc_content, doc_metadata):
            self.cur.execute(
                f"INSERT INTO {tablename} (document_id, embedding, content, metadata) VALUES (%s, %s, %s, %s)",
                (doc_id, emb.tolist(), content, metadata["source"])
            )

    def insert_feedback_to_table(self, tablename: str,
                                       datetime: str,
                                       user_query: str,
                                       response: str,
                                       user_feedback: str):
        self.cur.execute(
            (f"INSERT INTO {tablename} (datetime, user_query, response, user_feedback)"
             f"VALUES (%s, %s, %s, %s)"),
            (datetime, user_query, response, user_feedback)
        )

    def commit(self):
        self.conn.commit()

    def close(self):
        self.cur.close()
        self.conn.close()


if __name__ == "__main__":
    import numpy as np
    import datetime

    from credentials import db_config

    neon_db = NeonPostgres(dbname=db_config.db_name,
                           user=db_config.db_user,
                           password=db_config.db_pwd,
                           host=db_config.db_host,
                           port=db_config.db_port,
                           )

    log_tablename = "chatbot_log"
    neon_db.create_log_table(tablename=log_tablename)

    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prompt = "test prompt"
    response = "test response"
    feedback = "up"
    neon_db.insert_feedback_to_table(tablename=log_tablename,
                                     datetime=datetime_str,
                                     user_query=prompt,
                                     response=response,
                                     user_feedback=feedback)
    neon_db.commit()
    neon_db.close()

    ##############################################################
    #model_name = "all-MiniLM-L6-v2"
    #vs_name = "faiss_index_google_genai_risk_mgmt"
    #embedder = DocEmbedder(model_name=model_name,
    #                       vs_name=vs_name)

    #vectorstore = embedder.from_vs()
    #faiss_index = vectorstore.index
    #emb_vec = np.stack([faiss_index.reconstruct(i) for i in range(faiss_index.ntotal)])

    #print(emb_vec[:2])
    #print(emb_vec.shape)