
# Document Q&A with LangChain

This Streamlit application enables users to interact with a chatbot trained on a single PDF document. The chatbot utilizes LangChain, OpenAI, and FAISS to perform document-based question and answer tasks. The application is built to read a PDF, split its text into manageable chunks, create embeddings for these chunks, store them in a vector store, and then use a retriever and language model to answer user queries.

## Features

- **PDF Text Extraction**: Reads and extracts text from a provided PDF document.
- **Text Chunking**: Splits the extracted text into smaller, manageable chunks for processing.
- **Vector Store**: Creates embeddings from text chunks and stores them in a FAISS vector store for efficient retrieval.
- **Language Model**: Uses OpenAI's language model to generate responses to user queries.
- **Memory**: Maintains a conversation history to provide context for follow-up questions.

## Requirements
- Python 3.10
- Streamlit
- dotenv
- PyPDF2
- LangChain
- langchain-community
- langchain-openai
- OpenAI
- FAISS

## Installation

1. Clone the repository:

\`\`\`sh
git clone <repository_url>
cd <repository_name>
\`\`\`

2. Install the required dependencies:

\`\`\`sh
pip install streamlit python-dotenv PyPDF2 langchain openai faiss
or pip install -r requirements.txt
\`\`\`

3. Set up environment variables:

Create a `.env` file in the root directory and add your OpenAI API key:

\`\`\`env
OPENAI_API_KEY=your_openai_api_key
\`\`\`

4. Place your PDF document in the specified path or update the `pdf_path` variable in the script to point to your document:

\`\`\`python
pdf_path = r"C:\path\to\your\document.pdf"
\`\`\`

## Usage

1. Run the Streamlit application:

\`\`\`sh
streamlit run main.py
\`\`\`

2. Open your web browser and navigate to the local Streamlit server URL, usually `http://localhost:8501`.

3. Interact with the chatbot by entering questions about the content of your document in the provided text input field.

## Code Overview

- **get_pdf_text**: Extracts text from the provided PDF document.
- **get_text_chunks**: Splits the extracted text into smaller chunks.
- **get_vectorstore**: Creates embeddings from the text chunks and stores them in a FAISS vector store.
- **ll_retriver**: Sets up a language model-based retriever using OpenAI.
- **chain**: Creates a QA chain that uses the language model and retriever to answer questions.
- **main**: The main function that sets up the Streamlit interface and handles user interactions.

## Notes

- The application currently supports only a single document for training and querying.
- The conversation history is limited to the last two interactions to provide context for follow-up questions.

## Troubleshooting

If you encounter issues, ensure that all dependencies are correctly installed and that your environment variables are properly configured.

For further assistance, refer to the documentation of the respective libraries:

- [Streamlit](https://docs.streamlit.io/)
- [LangChain](https://langchain.readthedocs.io/)
- [OpenAI](https://beta.openai.com/docs/)
- [FAISS](https://github.com/facebookresearch/faiss)

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License.
