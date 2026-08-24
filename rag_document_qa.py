"""
RAG Document Question Answering
--------------------------------
A simple Retrieval-Augmented Generation application using:
- LangChain
- HuggingFace embeddings
- FAISS vector store

Place one or more .txt documents inside the documents/ folder.
The application retrieves relevant text chunks and returns an answer
using the retrieved context.
"""

import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DOCUMENTS_PATH = "documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3


# --------------------------------------------------
# 2. LOAD DOCUMENTS
# --------------------------------------------------

def load_documents(folder_path):
    documents = []

    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder '{folder_path}' was not found. "
            "Create it and add .txt documents."
        )

    text_files = list(folder.glob("*.txt"))

    if not text_files:
        raise FileNotFoundError(
            "No .txt documents were found in the documents folder."
        )

    for file_path in text_files:
        loader = TextLoader(
            str(file_path),
            encoding="utf-8"
        )

        documents.extend(loader.load())

    return documents


# --------------------------------------------------
# 3. SPLIT DOCUMENTS INTO CHUNKS
# --------------------------------------------------

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    return splitter.split_documents(
        documents
    )


# --------------------------------------------------
# 4. CREATE VECTOR DATABASE
# --------------------------------------------------

def create_vector_store(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store


# --------------------------------------------------
# 5. RETRIEVE RELEVANT DOCUMENTS
# --------------------------------------------------

def retrieve_context(vector_store, question):

    results = vector_store.similarity_search(
        question,
        k=TOP_K
    )

    context = "\n\n".join(
        document.page_content
        for document in results
    )

    return context, results


# --------------------------------------------------
# 6. SIMPLE RAG ANSWER
# --------------------------------------------------

def generate_answer(question, context):

    answer = (
        "Question:\n"
        f"{question}\n\n"
        "Retrieved Context:\n"
        f"{context}\n\n"
        "Answer:\n"
        "The most relevant information found in the documents "
        "is shown above. Use this retrieved context to answer "
        "the question."
    )

    return answer


# --------------------------------------------------
# 7. MAIN APPLICATION
# --------------------------------------------------

def main():

    print("=" * 60)
    print("RAG DOCUMENT QUESTION ANSWERING")
    print("=" * 60)

    print("\nLoading documents...")

    documents = load_documents(
        DOCUMENTS_PATH
    )

    print(
        f"Loaded {len(documents)} document(s)."
    )

    print("\nSplitting documents into chunks...")

    chunks = split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} text chunks."
    )

    print("\nCreating vector database...")

    vector_store = create_vector_store(
        chunks
    )

    print(
        "Vector database created successfully!"
    )

    print(
        "\nYou can now ask questions about your documents."
    )

    print(
        "Type 'exit' to stop the application.\n"
    )

    while True:

        question = input(
            "Ask a question: "
        )

        if question.lower() in [
            "exit",
            "quit"
        ]:
            print("\nApplication closed.")
            break

        context, sources = retrieve_context(
            vector_store,
            question
        )

        answer = generate_answer(
            question,
            context
        )

        print("\n" + "=" * 60)
        print(answer)
        print("=" * 60)

        print("\nRetrieved Sources:")

        for index, source in enumerate(
            sources,
            start=1
        ):
            print(
                f"{index}. "
                f"{source.metadata.get('source', 'Unknown')}"
            )

        print()


if __name__ == "__main__":
    main()
