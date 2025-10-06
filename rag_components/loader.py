from langchain_community.document_loaders import JSONLoader

def load_json_documents(file_path):
    """
    Load documents from a JSON file using JSONLoader.
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        List[Document]: List of loaded LangChain Document objects.
    """
    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".[] | {text: (.title + \": \" + .description)}",
        text_content=False
    )
    docs = loader.load()
    print("Documents loaded successfully.")
    return docs