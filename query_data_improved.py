import argparse
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

# Improved prompt template for structural queries
STRUCTURAL_PROMPT_TEMPLATE = """
You are analyzing a textbook document. Answer the question based on the following context, paying special attention to document structure, chapter information, and organizational details.

Context:
{context}

Question: {question}

Instructions:
- If asked about chapters, look for chapter numbers, titles, and structure
- If asked about content, provide specific information from the context
- If the information is not in the context, say so clearly
- Be precise and factual in your response

Answer:"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Determine if this is a structural query
    structural_keywords = ["chapter", "section", "how many", "structure", "organization", "table of contents"]
    is_structural = any(keyword in query_text.lower() for keyword in structural_keywords)
    
    # Adjust retrieval strategy based on query type
    if is_structural:
        # For structural queries, get more chunks and prioritize header information
        k = 8  # More chunks for structural understanding
        results = db.similarity_search_with_relevance_scores(query_text, k=k)
        
        # Filter for chunks with header information
        header_chunks = [doc for doc, score in results if any(key.startswith('Header') for key in doc.metadata.keys())]
        content_chunks = [doc for doc, score in results if not any(key.startswith('Header') for key in doc.metadata.keys())]
        
        # Prioritize header chunks for structural queries
        if header_chunks:
            results = header_chunks[:4] + content_chunks[:4]
        else:
            results = [doc for doc, score in results[:8]]
    else:
        # For content queries, use standard retrieval
        k = 5
        results = db.similarity_search_with_relevance_scores(query_text, k=k)
        results = [doc for doc, score in results]

    if len(results) == 0:
        print("Unable to find matching results.")
        return

    # Build context with metadata information
    context_parts = []
    for doc in results:
        # Include header information in context
        header_info = ""
        for key, value in doc.metadata.items():
            if key.startswith('Header') and value:
                header_info += f"[{key}: {value}] "
        
        context_parts.append(f"{header_info}\n{doc.page_content}")
    
    context_text = "\n\n---\n\n".join(context_parts)
    
    # Use appropriate prompt template
    if is_structural:
        prompt_template = ChatPromptTemplate.from_template(STRUCTURAL_PROMPT_TEMPLATE)
    else:
        prompt_template = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

{context}

Question: {question}

Answer:""")
    
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Print the prompt for debugging (optional)
    print("=== PROMPT ===")
    print(prompt)
    print("=== END PROMPT ===\n")

    # Get response from LLM
    model = ChatOpenAI(temperature=0.1)  # Lower temperature for more factual responses
    response_text = model.predict(prompt)

    # Format response
    sources = [doc.metadata.get("source", "Unknown") for doc in results]
    formatted_response = f"Response: {response_text}\n\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main() 