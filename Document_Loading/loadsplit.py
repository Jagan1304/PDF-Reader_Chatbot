from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

FILE_PATH = "../Documents/Laws.pdf"

loader = PyPDFLoader(FILE_PATH)

pages = loader.load_and_split()

print(len(pages))

embedding_function = SentenceTransformerEmbeddings(
    model_name = "all-MiniLM-L6-v2"
)

vectordb = Chroma.from_documents(
    documents = pages,
    embedding = embedding_function,
    persist_directory = "../vector_db",
    collection_name = "laws1"
)

vectordb.persist()