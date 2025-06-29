from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil

# Load environment variables
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text_improved(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents

def split_text_improved(documents: list[Document]):
    # First, split by markdown headers to preserve structure
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    
    # Then use recursive character splitter for smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Larger chunks for better context
        chunk_overlap=200,  # More overlap
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""]  # Better separators
    )
    
    all_chunks = []
    
    for doc in documents:
        # First split by headers
        header_splits = markdown_splitter.split_text(doc.page_content)
        
        for header_split in header_splits:
            # Then split into smaller chunks
            chunks = text_splitter.split_text(header_split.page_content)
            
            for i, chunk in enumerate(chunks):
                # Preserve header information in metadata
                metadata = {
                    "source": doc.metadata.get("source", ""),
                    "start_index": i * 1000,  # Approximate start index
                }
                
                # Add header information if available
                for header_key, header_value in header_split.metadata.items():
                    if header_value:
                        metadata[header_key] = header_value
                
                all_chunks.append(Document(
                    page_content=chunk,
                    metadata=metadata
                ))
    
    print(f"Split {len(documents)} documents into {len(all_chunks)} chunks.")
    
    # Show a sample chunk with header info
    if all_chunks:
        sample_chunk = all_chunks[10]
        print("\nSample chunk:")
        print(sample_chunk.page_content[:200] + "...")
        print("Metadata:", sample_chunk.metadata)
    
    return all_chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main() 