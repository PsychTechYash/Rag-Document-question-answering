# Rag-Document-question-answering
A Retrieval-Augmented Generation (RAG) application that uses LangChain and vector search to answer questions based on uploaded documents.
# 🤖 RAG Document Question Answering

A Retrieval-Augmented Generation (RAG) application that uses LangChain, HuggingFace embeddings, and FAISS vector search to answer questions based on custom documents.

## 📌 Project Overview

This project demonstrates the core workflow of a **Retrieval-Augmented Generation (RAG)** system.

Instead of relying only on a language model's existing knowledge, the application searches through user-provided documents, retrieves the most relevant information, and uses that context to support answers.

## 🎯 Objective

The objective is to build a document-based question-answering system that can:

* Load custom documents.
* Split documents into smaller chunks.
* Convert text into vector embeddings.
* Store embeddings in a vector database.
* Retrieve the most relevant document chunks.
* Answer questions using the retrieved context.

## 🔄 RAG Workflow

```text
Documents
    ↓
Document Loading
    ↓
Text Chunking
    ↓
Embedding Generation
    ↓
FAISS Vector Database
    ↓
User Question
    ↓
Similarity Search
    ↓
Relevant Context
    ↓
Answer Generation
```

## 🛠️ Technologies Used

* Python
* LangChain
* HuggingFace Embeddings
* Sentence Transformers
* FAISS

## 📂 Project Structure

```text
rag-document-question-answering/
│
├── documents/
│   ├── document1.txt
│   └── document2.txt
│
├── rag_document_qa.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/rag-document-question-answering.git
```

Navigate to the project directory:

```bash
cd rag-document-question-answering
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 📄 Adding Documents

Create a folder named:

```text
documents
```

Add one or more `.txt` files to this folder.

Example:

```text
documents/
├── company_information.txt
├── product_details.txt
└── policies.txt
```

The application will load all text documents from this directory.

## 🚀 Running the Project

Run:

```bash
python rag_document_qa.py
```

The application will:

1. Load the documents.
2. Split them into smaller chunks.
3. Generate vector embeddings.
4. Store the embeddings using FAISS.
5. Accept questions from the user.
6. Retrieve the most relevant document chunks.
7. Display the retrieved context and source files.

Type:

```text
exit
```

to close the application.

## 🧠 How Retrieval Works

When a user asks a question:

1. The question is converted into a vector embedding.
2. FAISS searches for similar document embeddings.
3. The most relevant text chunks are retrieved.
4. The retrieved information is used as context for answering the question.

This approach allows an AI system to work with information that was not necessarily part of its original training data.

## 🚀 Future Improvements

* Integrate an LLM for natural language answer generation.
* Support PDF and DOCX documents.
* Add a Streamlit web interface.
* Store the vector database permanently.
* Add conversation memory.
* Support multiple users.
* Deploy the application as an AI chatbot.

## 👨‍💻 Author

**Yash Vardhan**

B.Tech CSE (AI)
VIT Bhopal University

## 📜 License

This project is created for educational and learning purposes.
