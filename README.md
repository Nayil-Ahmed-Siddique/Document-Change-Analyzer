# 📄 Semantic Document Comparison API

A backend service that compares two versions of a document and detects whether the document has changed in a meaningful way.

This project can be used for comparing:
- Privacy Policies
- Terms & Conditions
- Company policies
- Any versioned text documents

This project focuses on semantic comparison, not simple text matching.

---

## 🔧 Tools Used

- Python
- Hugging Face Sentence Transformers
- BentoML

---

## 🤗 Hugging Face Integration

Computers do not understand natural language directly.
They understand numbers.

This project uses a Hugging Face sentence embedding model to convert text into numeric representations.

Model used:
sentence-transformers/all-MiniLM-L6-v2

Each document is converted into an embedding that represents its meaning.

---

## 🔢 How Embeddings Are Used

- Document version 1 is converted into numbers
- Document version 2 is converted into numbers
- Both numeric representations are compared using cosine similarity

If the similarity score is high, the meaning is mostly the same.
If the similarity score is low, the meaning has changed.

Based on this score, the project classifies changes as:
- Minor or no change
- Major change

---

## 🚀 BentoML Service

BentoML is used to package the document comparison logic into a production-ready API service.

With BentoML:
- The logic lives in a service.py file
- A service class is defined
- An API endpoint is exposed
- The service can be started as an HTTP server

This allows other applications to consume the document comparison logic easily.

---

## ⚙️ Service Flow

1. The service receives two document texts
2. Hugging Face converts both documents into embeddings
3. The embeddings are compared
4. A similarity score is calculated
5. The service returns a structured response indicating the type of change

---

## 📁 Project Structure

- service.py  
  Main production file containing Hugging Face and BentoML logic

- analysis.ipynb  
  Notebook used for experimentation and explanation

- Policy_v1.txt and Policy_v2.txt  
  Sample input documents

- output.json  
  Example output from the service

---

## 🎯 What This Project Demonstrates

- Practical usage of Hugging Face pretrained models
- Semantic document comparison using embeddings
- Building a production-style API with BentoML
- Clean separation between experimentation and deployment

---

## ▶️ How to Use This Project

This service can be used to compare any two text documents, such as privacy policies or terms and conditions.

- Prepare two document versions in plain text format
- Start the BentoML service
- Send both document texts to the analyze API endpoint
- Receive a response containing:
  - similarity score
  - classification of change (major or minor)

---

## 📝 Final Notes

This project does not focus on text generation or chatbots.
It demonstrates how Hugging Face models can be integrated into a backend service using BentoML to solve a real-world document comparison problem.

