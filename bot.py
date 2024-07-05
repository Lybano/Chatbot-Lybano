import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

OPENAI_API_KEY = os.getenv('OPENAI_TOKEN')

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(all_data)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

### Contextualize question ###
contextualize_q_system_prompt = (
    "Dado um histórico de bate-papo e a última pergunta do usuário " 
    "que pode fazer referência ao contexto no histórico de bate-papo, " 
    "formule uma pergunta independente que possa ser entendida " 
    "sem o histórico de bate-papo. NÃO responda à pergunta, " 
    "apenas reformule-a se necessário e devolvê-lo como está."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

### Answer question ###
system_prompt = (
    "Você é um assistente para tarefas de resposta a perguntas de uma empresa chamada alesk."
    "Utilize apenas 3 frases para responder as duvidas"
    "Inicie a conversa se apresentando como assistente virtual da alesk"
    "Use as seguintes partes do contexto recuperado para responder à " 
    "pergunta. Se você não souber a resposta, diga que " 
    "não sabe. mantenha a " 
    "resposta concisa." 
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

def run_model(query):
    resposta = conversational_rag_chain.invoke(
    {"input": query},
    config={
        "configurable": {"session_id": "abc123"}
    },
    )["answer"]
    return resposta

pergunta = run_model("Qual as informções das quais você tem acesso?")
print(pergunta)
