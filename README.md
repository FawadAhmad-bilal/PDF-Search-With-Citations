# PDF Search With Citations

A document search app that finds the most relevant lines for your question and shows the **file name and page number** for each result. No LLM and no API key needed, so answers come straight from your documents and nothing is made up.

**Live demo:** <add your Streamlit link here>

## Features
- Upload multiple files at once (PDF, DOCX, TXT, MD)
- Semantic search: matches by meaning, not just keywords
- Top 3 matching sentences with file name and page number
- Sources section with the full matching passages
- Removes the References section from PDFs so it doesn't pollute results
- Dark themed UI

## How it works
1. Load the files and split the text into chunks
2. Convert the chunks into embeddings and store them in FAISS
3. Retrieve the closest chunks for your question
4. Split them into sentences and rank by similarity
5. Show the best sentences with their citations

## Tech Stack
Python, Streamlit, LangChain, FAISS, FastEmbed (all-MiniLM-L6-v2)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limitations
- Returns existing sentences, it does not write summaries
- Scanned PDFs (images) are not supported
- Old `.doc` files are not supported, only `.docx`

Made by **Fawad Ahmad Bilal**
