from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Get the embedding model used for encoding documents and queries.
    
    Args:
        model_name (str): Name or path of the Hugging Face embedding model.
        
    Returns:
        Embeddings: HuggingFaceEmbeddings object.
    """
    return HuggingFaceEmbeddings(model_name=model_name)