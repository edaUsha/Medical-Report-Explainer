# run this as a quick sanity check
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
import argparse
from langchain_groq import ChatGroq
import os

load_dotenv()
"""print("OpenAI key loaded:", bool(os.getenv("OPENAI_API_KEY")))
print("Pinecone key loaded:", bool(os.getenv("PINECONE_API_KEY")))
print("All imports OK")"""

'''from pinecone import Pinecone

pc = Pinecone(api_key="pcsk_3tFkVG_DD6WiTCDSo4wNiGVNK2ndH7yTEfcCM1WApBDrzpXSfQLxtudu27LWTVSkbNhsfc")
index = pc.Index("medical-rag")'''

def run_ingest(pdf_path:str):
    from src.ingest import load_pdf,chunk_documents,create_embeddings,upsert_to_pinecone
    print("====INGESTION PIPELINE=====")
    pages = load_pdf("data/sample medical rag report.pdf")
    chunks = chunk_documents(pages)
    embeddings= create_embeddings()
    upsert_to_pinecone(chunks,embeddings)
    print("====INGESTION PIPEPINE COMPLETED!====")

def run_query(question:str):
    from src.query import load_vectorstore,load_llm, build_qa_chain,ask
    print("====QUERY PIPELINE====")
    vectorstore=load_vectorstore()
    llm=load_llm()
    chain=build_qa_chain(vectorstore,llm)
    ask(chain,question)

def run_chat():
    from src.query import load_vectorstore,load_llm, build_qa_chain,ask
    print("=====MEDICAL REPORT QA====")
    print("====TYPE YOUR QUESTION OR 'quit' TO EXIT\n")
    vectorstore=load_vectorstore()
    llm=load_llm()
    chain=build_qa_chain(vectorstore,llm)

    while True:
        question=input("Your Question:").strip()
        if question.lower() in ("quit","exit","q"):
            print("Goodbye")
            break
        if not question:
            continue
        ask(chain,question)
        print()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Medical RAG Q&A System")
    subparser=parser.add_subparsers(dest="command")

    #ingest command
    ingest_parser=subparser.add_parser("ingest",help="Index a PDF")
    ingest_parser.add_argument('pdf',help="Path to PDF file")

    #query command
    query_parser=subparser.add_parser("query",help="Ask a single question")
    query_parser.add_argument("question",help="Your Question in quotes")

    #chat command
    subparser.add_parser("chat",help="Interactive Q&A Session")

    args= parser.parse_args()

    if args.command=="ingest":
        run_ingest(args.pdf)
    elif args.command == "query":
        run_query(args.question)
    elif args.command == "chat":
        run_chat()
    else:
        parser.print_help()