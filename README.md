# RAG Textbook Summarizer

A Retrieval-Augmented Generation (RAG) system for summarizing and querying textbook content using LangChain and OpenAI.

## Features

- **Document Processing**: Load and chunk markdown documents with semantic awareness
- **Vector Storage**: Store document embeddings in ChromaDB for efficient retrieval
- **Smart Querying**: Intelligent retrieval based on query type (structural vs content)
- **Improved Chunking**: Preserves document structure and headers for better context
- **OpenAI Integration**: Uses OpenAI embeddings and GPT models for generation

## Project Structure

```
├── create_database.py              # Basic database creation
├── create_database_improved.py     # Improved chunking with header preservation
├── query_data.py                   # Basic querying
├── query_data_improved.py          # Enhanced querying with structural awareness
├── compare_embeddings.py           # Embedding comparison utilities
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (create this)
├── data/books/                    # Place your markdown documents here
└── chroma/                        # Vector database (auto-generated)
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rag-textbook-summarizer
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install "unstructured[md]"
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

### 1. Prepare Your Documents

Place your markdown documents in the `data/books/` directory.

### 2. Create the Vector Database

**Basic version**:
```bash
python create_database.py
```

**Improved version** (recommended):
```bash
python create_database_improved.py
```

### 3. Query Your Documents

**Basic querying**:
```bash
python query_data.py "Your question here"
```

**Improved querying** (recommended):
```bash
python query_data_improved.py "Your question here"
```

## Example Queries

- **Structural questions**: "How many chapters are there?", "What is chapter 3 about?"
- **Content questions**: "Explain search algorithms", "What is an agent in AI?"
- **Comparative questions**: "Compare depth-first and breadth-first search"

## Key Improvements

### Enhanced Chunking Strategy
- Uses `MarkdownHeaderTextSplitter` to preserve document structure
- Larger chunk sizes (1000 chars) with more overlap (200 chars)
- Preserves header metadata for better context

### Intelligent Retrieval
- Detects structural vs content queries
- Prioritizes header chunks for structural questions
- Retrieves more context for complex queries

### Better Prompting
- Specialized prompts for different query types
- Lower temperature for more factual responses
- Enhanced context formatting

## Dependencies

- `langchain` - Core RAG framework
- `langchain-community` - Community integrations
- `langchain-openai` - OpenAI integrations
- `langchain-text-splitters` - Text chunking utilities
- `chromadb` - Vector database
- `openai` - OpenAI API client
- `python-dotenv` - Environment variable management
- `unstructured` - Document processing
- `onnxruntime` - ML runtime for ChromaDB

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're using the virtual environment
2. **API key errors**: Verify your `.env` file contains the correct OpenAI API key
3. **Chunking issues**: Use the improved version for better document structure handling
4. **Memory issues**: Reduce chunk size in `create_database_improved.py` if needed

### Performance Tips

- Use the improved scripts for better results
- Ensure documents have proper markdown headers
- For large documents, consider preprocessing to add consistent headers

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
