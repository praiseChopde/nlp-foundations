"""
LangChain RAG Demo
Retrieval-Augmented Generation Pipeline

This demo builds a complete RAG system that:
1. Loads documents (PDF or text)
2. Splits them into chunks
3. Embeds chunks with sentence-transformers
4. Stores embeddings in a FAISS vector store
5. Retrieves relevant chunks for a query
6. Generates an answer using an LLM

This architecture directly mirrors the NSERC project's 
self-correcting LM pipeline for educational question generation.
"""

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline as hf_pipeline
import os

# ─────────────────────────────────────────────────────────
# STEP 1: DOCUMENT LOADING
# ─────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1: Loading Documents")
print("=" * 60)

# Create a sample document for the demo
sample_text = """
Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems 
to learn and improve from experience without being explicitly programmed. 
It focuses on developing computer programs that can access data and use it 
to learn for themselves.

Types of Machine Learning

Supervised learning uses labeled training data to learn a mapping from 
inputs to outputs. Common algorithms include linear regression for continuous 
outputs and logistic regression for binary classification. Decision trees 
and random forests are ensemble methods that combine multiple weak learners.

Unsupervised learning finds hidden patterns in data without labeled responses. 
Clustering algorithms like K-means group similar data points together. 
Principal component analysis reduces dimensionality while preserving variance.

Reinforcement learning trains agents to make sequences of decisions by 
rewarding desired behaviors and penalizing undesired ones. The agent learns 
a policy that maximizes cumulative reward over time.

Neural Networks and Deep Learning

Neural networks are computing systems inspired by biological neural networks. 
They consist of layers of interconnected nodes that process information. 
Deep learning uses neural networks with many layers to learn hierarchical 
representations of data.

Convolutional neural networks excel at image recognition tasks. Recurrent 
neural networks handle sequential data like text and time series. Transformers 
use attention mechanisms to process sequences in parallel, revolutionizing NLP.

Applications of Machine Learning

Machine learning is used in natural language processing for translation, 
summarization, and question answering. Computer vision applications include 
object detection, facial recognition, and medical image analysis. 
Recommendation systems power streaming services and e-commerce platforms.
"""

# Save sample document
os.makedirs("data/sample_docs", exist_ok=True)
with open("data/sample_docs/ml_intro.txt", "w") as f:
    f.write(sample_text)

loader = TextLoader("data/sample_docs/ml_intro.txt")
documents = loader.load()
print(f"Loaded {len(documents)} document(s)")
print(f"Total characters: {sum(len(d.page_content) for d in documents)}")

# ─────────────────────────────────────────────────────────
# STEP 2: TEXT SPLITTING
# ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2: Splitting Documents into Chunks")
print("=" * 60)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,          # Each chunk is ~400 characters
    chunk_overlap=50,        # 50-char overlap preserves context at boundaries
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)

chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")
print(f"Chunk sizes: {[len(c.page_content) for c in chunks]}")
print(f"\nExample chunk:\n  '{chunks[0].page_content[:200]}...'")

# ─────────────────────────────────────────────────────────
# STEP 3: EMBEDDING
# ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3: Embedding Chunks with sentence-transformers")
print("=" * 60)

# all-MiniLM-L6-v2: fast, good quality, 384-dim embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Test embedding
sample_embedding = embedding_model.embed_query("What is machine learning?")
print(f"Embedding model: sentence-transformers/all-MiniLM-L6-v2")
print(f"Embedding dimensions: {len(sample_embedding)}")
print(f"Sample values (first 5): {[round(x, 4) for x in sample_embedding[:5]]}")

# ─────────────────────────────────────────────────────────
# STEP 4: VECTOR STORE (FAISS)
# ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: Storing Embeddings in FAISS Vector Store")
print("=" * 60)

vectorstore = FAISS.from_documents(chunks, embedding_model)
vectorstore.save_local("data/faiss_index")
print(f"Vector store created with {len(chunks)} vectors")
print(f"Index saved to: data/faiss_index/")

# ─────────────────────────────────────────────────────────
# STEP 5: RETRIEVAL
# ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5: Semantic Retrieval")
print("=" * 60)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

test_queries = [
    "What is supervised learning?",
    "How do neural networks work?",
    "What are applications of machine learning?",
]

for query in test_queries:
    docs = retriever.invoke(query)
    print(f"\nQuery: '{query}'")
    print(f"Top retrieved chunk:")
    print(f"  '{docs[0].page_content[:150]}...'")

# ─────────────────────────────────────────────────────────
# STEP 6: LLM + RAG CHAIN
# ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6: Full RAG Chain with LLM")
print("=" * 60)

# Use a small local model for demonstration
# In production, replace with OpenAI/Groq/Anthropic API for better quality
llm_pipeline = hf_pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=200,
    temperature=0.1,
)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Custom prompt template
prompt_template = """Use the following pieces of context to answer the question.
If you don't know the answer based on the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Build the RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True,
)

# Test it
questions = [
    "What is reinforcement learning?",
    "What are convolutional neural networks used for?",
]

for q in questions:
    print(f"\nQ: {q}")
    result = qa_chain.invoke({"query": q})
    print(f"A: {result['result']}")
    print(f"Sources used: {len(result['source_documents'])} chunks")

print("\n" + "=" * 60)
print("RAG Demo Complete!")
print("=" * 60)
print("\nPipeline summary:")
print("  1. Load document         -> TextLoader / PyPDFLoader")
print("  2. Split into chunks     -> RecursiveCharacterTextSplitter")
print("  3. Embed chunks          -> sentence-transformers/all-MiniLM-L6-v2")
print("  4. Store in vector DB    -> FAISS")
print("  5. Retrieve on query     -> similarity search (cosine distance)")
print("  6. Generate answer       -> LLM conditioned on retrieved context")
print("\nThis architecture is the foundation of EduQGen's question generation pipeline.")
