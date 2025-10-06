from rag_components.loader import load_json_documents
from rag_components.splitter import split_documents
from rag_components.embedder import get_embeddings
from rag_components.vectorestore import create_vectorstore
from rag_components.retriever import get_retriever
from rag_components.chain import create_qa_chain

def main():
    print("🧠 Loading HR Policy Chatbot...\n")

    # Paths and config
    json_file = "../data/hr_policy.json"

    # Step 1: Load documents
    docs = load_json_documents(json_file)

    # Step 2: Split documents
    split_docs = split_documents(docs)

    # Step 3: Get embeddings
    embeddings = get_embeddings()

    # Step 4: Create or load vector store
    vectorstore = create_vectorstore(split_docs, embeddings)

    # Step 5: Get retriever
    retriever = get_retriever(vectorstore)

    # Step 6: Build chain
    qa_chain = create_qa_chain(retriever)

    print("✅ Chatbot is ready! Type 'exit' to quit.\n")

    # Interactive loop
    while True:
        user_query = input("❓ Ask your HR policy question (or type 'exit'): ").strip()
        if user_query.lower() == "exit":
            print("\n👋 Exiting HR Policy Chatbot. Have a great day!")
            break

        # Step 7: Invoke chain
        response = qa_chain.invoke(user_query)
        print(f"\n🤖 Answer:\n{response}\n")


if __name__ == "__main__":
    main()