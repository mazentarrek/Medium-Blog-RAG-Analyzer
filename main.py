import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq


load_dotenv()

print("Initializing Components...")
embeddings= OpenAIEmbeddings(api_key=os.environ.get("OPEN_API_KEY"))
llm = ChatGroq(api_key=os.environ.get("GROQ_API_KEY"), model="llama-3.3-70b-versatile")
vectorstore = PineconeVectorStore(index_name=os.environ.get("INDEX_name"), embedding=embeddings)
retreiver = vectorstore.as_retriever(search_kwargs={"k":3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question base on only the following context
    
    Context:

    {Context}

    Question : 
    {Question}

    Provide a detailed answer:

    """
)

def format_docs(docs):
    """Format retreived dicuments into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)

def retieval_chain_without_lcel(query:str):

    # Step 1 : Retrieve relevant docs from query
    docs = retreiver.invoke(query)

    # Step 2: Format documents into context string
    context = format_docs(docs)

    # Step 3: Format prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    #Step 4: Invoke LLM
    result = llm.invoke(messages)

    return result.content



if __name__ == "__main__":
    print("Retreiving")

    query = "When did the expirement begin at?"

    result_raw = llm.invoke([HumanMessage(content=query)])
    print("Raw Result Result: \n")
    print(result_raw.content)

    print("RAG without lcel Result Result: \n")
    print(retieval_chain_without_lcel(query))
