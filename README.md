# Setup Instructions:
1. Clone my forked repository
2. Make sure you are in the res389-assignment1 branch
3. Install dependencies: `pip install -r requirements.txt`
4. Run the streamlit app: API_KEY="<your_api_key>" streamlit run chat_with_pdf.py

# Accepted Files
You can upload .txt, .pdf, and .html (or .htm) files (I added .html files because I saw beautifulsoup was in the requirements.txt). There is no limit of how many files you can upload, but the maximum size is 200 mb per file.

# Chatbot features
This chatbot is provides enhanced Q&A to the user. You can upload any supported documents and ask questions about them. The chatbot will cite the top 5 most related excerpts that it used to respond to your question. You can also clear your chat with the 'Clear Chat' button in the sidebar, and you can remove any of your documents at will. The LLM being used is OpenAI's gpt-4o model, and the embeddings are created with OpenAI's text embedding 3 large model.

# Chunking strategy
Each chunk is 800 characters NOT tokens, and the chunks overlap by 100 characters to preserve some context. They are also separated by natural English punctuation (i.e. paragraphs, sentences, words) to minimize any lost context when chunking. 

# Modifications to requirements.txt and devcontainer.json
I changed pandas to 2.1.4 because of a compatibility error with numpy. I made no modifications to devcontainer.json.

