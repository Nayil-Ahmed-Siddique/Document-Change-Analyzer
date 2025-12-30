📄 Document Change Analyzer

A backend service that compares two versions of a document and detects whether the document has changed in a meaningful way.

This project is useful for comparing things like:

📜 Privacy Policies

📑 Terms & Conditions

🏢 Company policies

📝 Any versioned text documents

🛠 Tools and Libraries Used

🐍 Python – main programming language

🤗 Hugging Face (Sentence Transformers) – to understand text meaning

🚀 BentoML – to turn the logic into a real API service

🤗 Hugging Face Usage

Computers do not understand English text directly.
They understand numbers.

To solve this, this project uses a Hugging Face embedding model:

Model used: sentence-transformers/all-MiniLM-L6-v2

This model converts text into numeric vectors (embeddings) that represent the meaning of the text.

🔢 What Embeddings Do in This Project

Document Version 1 → converted into numbers

Document Version 2 → converted into numbers

These numeric vectors are then compared using cosine similarity.

📈 High similarity score → meaning is mostly the same

📉 Low similarity score → meaning has changed

Based on the similarity score, the project classifies the change as:

✅ minor or no change

⚠️ major change

🚀 BentoML Usage

BentoML is used to wrap all the logic into a production-style service.

With BentoML:

The core logic is written in a service.py file

A service class is defined

An API endpoint (analyze) is exposed

The service can be started using a single command

This makes the project usable as an API, not just a script.

⚙️ How the Service Works (High Level)

The service receives two text documents as input

Hugging Face converts both documents into embeddings

The embeddings are compared

A similarity score is calculated

The service returns a JSON response containing:

similarity score

type of change (major or minor)

📁 Project Files

📄 service.py

Main production file

Loads the Hugging Face model

Contains BentoML service and API logic

📓 analysis.ipynb

Used for experimentation and explanation

Not required for running the service

📝 Policy_v1.txt, Policy_v2.txt

Sample input documents

📤 output.json

Example output format

🎯 What This Project Demonstrates

Using Hugging Face pretrained models in a practical way

Converting text into embeddings for semantic comparison

Building a production-style API using BentoML

Separating experimentation from deployment logic

▶️ How to Use This Project

This service can be used to compare any two text documents, such as privacy policies or terms and conditions.

Convert both document versions into plain text format

Start the BentoML service

Send both document texts to the analyze API endpoint

Receive a response containing:

similarity score

classification of change (major or minor)

📝 Final Note

This project focuses on semantic document comparison, not text generation or chatbots.

It demonstrates how Hugging Face models can be integrated into a real backend service using BentoML to solve a practical real-world problem.
