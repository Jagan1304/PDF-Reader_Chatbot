from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from decouple import config
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def main():
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="../vector_db", collection_name="laws1", embedding_function=embedding_function)
    llm = ChatOpenAI(openai_api_key=config("OPENAI_API_KEY"), temperature=0)
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")


    QA_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=vector_db.as_retriever(search_kwargs={"fetch_k": 3, "k": 3}, search_type="mmr"),
        chain_type="refine",
    )

 
    updater = Updater("6855690340:AAHn5ZKkY4PSn4tdxc450pfLoCkcVaq6kVE", use_context=True)


    dp = updater.dispatcher


    def start(update, context):
        update.message.reply_text("Hello, I am here to assist you with any queries related to the Mines and Minerals (Development and Regulation) Act. Please feel free to ask your questions, and I am here to help you.")


    def handle_message(update, context):

        question = update.message.text


        response = QA_chain({"question": question})
        answer = response.get("answer")


        update.message.reply_text(answer)


    dp.add_handler(CommandHandler("start", start))


    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))


    updater.start_polling()


    updater.idle()

if __name__ == '__main__':
    main()
