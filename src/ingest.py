#PDF → chunks → embeddings → Pinecone

from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from pinecone import Pinecone,ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()
def load_pdf(path:str):
    loader= PyPDFLoader(path)
    pages=loader.load()
    print(f"Loaded {len(pages)} from {path}")
    return pages
    pass

def chunk_documents(pages:list):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks=splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks from {len(pages)} pages")
    return chunks

def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("HuggingFace embeddings model loaded")
    return embeddings

def clear_index():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")
    index = pc.Index(index_name)
    index.delete(delete_all=True)       # wipes every vector in the index
    print("Cleared existing vectors from Pinecone")

def upsert_to_pinecone(chunks: list, embeddings):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")

    existing = [i.name for i in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Created index: {index_name}")
    else:
        # ← ADD THIS: wipe old vectors before upserting new ones
        index = pc.Index(index_name)
        index.delete(delete_all=True)
        print("Cleared old vectors")

    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )
    print(f"Upserted {len(chunks)} chunks to Pinecone")
    return vectorstore


if __name__=="__main__":
    pages = load_pdf("data/USHA's_report.pdf")
    chunks = chunk_documents(pages)
    embeddings= create_embeddings()
    vectorstore=upsert_to_pinecone(chunks,embeddings)


    # inspect a few chunks
    #for i, chunk in enumerate(chunks[:3]):
    #    print(f"\n--- Chunk {i+1} ---")
    #    print(chunk.page_content)
    #    print("Metadata:", chunk.metadata)

    #test on one chunk
    #test_vector=embeddings.embed_query(chunks[0].page_content)
    #print(f"\nVector dimensions:{len(test_vector)}")
    #print(f"First 5 values: {test_vector[:5]}")
    print("\nDay 53 complete — pipeline indexed and ready!")
