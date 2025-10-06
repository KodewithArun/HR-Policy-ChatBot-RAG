def get_retriever(vectorstore, search_type="mmr", k=5, lambda_mult=0.5):
    """
    Get a configured retriever from the vector store.
    
    Args:
        vectorstore (Chroma): Vector store instance.
        search_type (str): Type of search ('similarity', 'mmr').
        k (int): Number of documents to retrieve.
        lambda_mult (float): Controls diversity in MMR search.
        
    Returns:
        Retriever: Configured retriever object.
    """
    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k, "lambda_mult": lambda_mult}
    )
    return retriever