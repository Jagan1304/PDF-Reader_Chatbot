from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from decouple import config
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

embedding_function = SentenceTransformerEmbeddings(
    model_name = "all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory = "../vector_db",
    collection_name = "laws1",
    embedding_function = embedding_function,
)

llm = ChatOpenAI(openai_api_key=config("OPENAI_API_KEY"), temperature = 0)

memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history"
)

QA_chain = ConversationalRetrievalChain.from_llm(
    llm = llm,
    memory = memory,
    retriever = vector_db.as_retriever(
            search_kwargs = {"fetch_k" : 4, "k" : 3},
            search_type = "mmr"
        ),
    chain_type = "refine",
)

question = "How does the MMDR Act address issues of illegal mining and unauthorized extraction of minerals?"

response = QA_chain({"question": question})

print(response.get("answer"))