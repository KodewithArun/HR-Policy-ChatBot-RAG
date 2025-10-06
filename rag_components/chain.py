from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def create_qa_chain(retriever, llm_model="llama-3.1-8b-instant", temperature=0.1):
    """
    Create a question-answering chain using LLM and retriever.
    
    Args:
        retriever (Retriever): The retriever object.
        llm_model (str): Model name for ChatGroq.
        temperature (float): Temperature for LLM generation.
        
    Returns:
        RunnableSequence: QA chain ready to use.
    """
    llm = ChatGroq(model=llm_model, temperature=temperature)

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an HR policy assistant. Use the context below to answer the user's question.
If the question is a casual greeting (such as "hi", "hello", "hey", "good morning", etc.), respond with a friendly greeting as an HR assistant.
If the answer is not found in the context, reply with: "Sorry, I couldn't find that in the HR policy."

Context:
{context}

Question: {question}

Answer:
"""
    )

    chain = RunnableSequence(
        {
            "context": retriever | format_docs,
            "question": lambda x: x
        },
        prompt_template,
        llm,
        StrOutputParser()
    )
    return chain