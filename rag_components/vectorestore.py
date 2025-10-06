from langchain_community.vectorstores import Chroma

def create_vectorstore(documents, embedding, persist_directory="../vector_store", collection_name="hr_policy_collection"):
    """
    Create and persist a Chroma vector store from documents.
    
    Args:
        documents (List[Document]): List of document chunks.
        embedding (Embeddings): Embedding model.
        persist_directory (str): Directory to save the vector store.
        collection_name (str): Name of the collection.
        
    Returns:
        Chroma: The created vector store.
    """
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print("Vector store created successfully.")
    return vectorstore