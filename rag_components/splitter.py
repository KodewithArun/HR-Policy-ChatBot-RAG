from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into smaller chunks for processing.
    
    Args:
        docs (List[Document]): List of Document objects.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap between consecutive chunks.
        
    Returns:
        List[Document]: List of split document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = text_splitter.split_documents(docs)
    print(f"Number of chunks after splitting: {len(split_docs)}")
    return split_docs