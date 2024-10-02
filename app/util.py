from typing import List

from llama_index.core.schema import Document


def convert_docs_to_data(docs: List[Document]) -> List[object]:
    """
    Convert documents to data.
    """
    data = []
    for doc in docs:
        data.append({"data": {"id": doc.doc_id, "text": doc.text}})
    return data
