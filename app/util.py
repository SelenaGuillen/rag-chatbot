async def convert_to_documents(data: dict) -> list:
    """
    Convert to documents.
    """
    documents = []
    db_docs = data.get("data")
    for doc in db_docs:
        documents.append(doc.get("document").get("text"))
    return documents
