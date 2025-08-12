=========================================================
StudyMate AI — Streamlit RAG Chatbot
=========================================================

DESCRIPTION:
------------
StudyMate AI is an intelligent learning assistant built with:
- Streamlit (Frontend)
- Groq LLM API
- ChromaDB (Vector Store)
- HuggingFace Embeddings (e5-large-v2)
- Optional Cross-Encoder Reranker

It allows you to upload study materials (PDF, DOCX, TXT), 
store them in a persistent vector database, and chat with 
an AI that retrieves and answers based on your documents.

---------------------------------------------------------
FEATURES:
---------------------------------------------------------
1. Upload documents (PDF, DOCX, TXT) from the UI.
2. Persist data in ChromaDB so your knowledge base remains 
   available after restart.
3. Manage multiple conversations with full chat history.
4. Answers include source citations with file download 
   buttons.
5. Optional re-ranking of results for better relevance.
6. Fast and accurate answers powered by Groq LLaMA models.

---------------------------------------------------------
TECH STACK:
---------------------------------------------------------
- Streamlit
- Groq API
- Chroma Vector Store
- HuggingFace e5-large-v2 Embeddings
- Cross-Encoder Reranker (SentenceTransformers)
- PyPDF, python-docx for file reading

---------------------------------------------------------
INSTALLATION:
---------------------------------------------------------
1. Clone the repository:
   git clone https://github.com/yourusername/StudyMate-AI.git
   cd StudyMate-AI

2. Create and activate virtual environment:
   python -m venv venv
   venv\Scripts\activate   (Windows)
   source venv/bin/activate (Mac/Linux)

3. Install dependencies:
   pip install -r requirements.txt

4. Create a .env file in the project root:
   GROQ_API_KEY=your_groq_api_key_here

5. Make sure .env is in .gitignore to avoid committing secrets.

---------------------------------------------------------
USAGE:
---------------------------------------------------------
Run the Streamlit app:
   streamlit run app.py

In the web interface:
1. Upload your PDF/DOCX/TXT files.
2. Click "Add to Knowledge Base" to embed and store them.
3. Ask questions in the input box.
4. View answers and download source documents.

---------------------------------------------------------
CONFIGURATION:
---------------------------------------------------------
In the sidebar:
- Enable/disable re-ranker (slower but more accurate).
- Set top_k retrieval values for better performance.

---------------------------------------------------------
FOLDER STRUCTURE:
---------------------------------------------------------
app.py                -> Main application script

uploads/              -> Stores uploaded documents

chroma_store/         -> Persistent Chroma database

requirements.txt      -> List of dependencies

.env                  -> API keys and environment variables

readme.txt            -> Project documentation



=========================================================
