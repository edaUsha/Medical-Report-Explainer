#question → embed → Pinecone search → Claude → plain-English answer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

def load_vectorstore():
    embeddings= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore=PineconeVectorStore.from_existing_index(index_name=os.getenv("PINECONE_INDEX_NAME"),
                                                        embedding=embeddings)
    # connects to your already-populated Pinecone index.
    print(f"Connected to Pinecone index")
    return vectorstore


def retrieve_chunks(vectorstore, question:str, k:int=5):
    #wraps the vectorstore as a LangChain retriever. k=5 means fetch the 5 nearest chunks by cosine similarity.
    retriever=vectorstore.as_retriever(search_kwargs={"k":k})
    chunks=retriever.invoke(question)
    print(f"\n Top {k} chunks retrieved for:'{question}'")
    return chunks

def load_llm():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )
    print("GPT-4o loaded")
    return llm

def build_prompt():
    template="""
        You are a helpful medical report explainer. Your job is to help patients
        understand their medical reports in simple, clear language.

        Use ONLY the information in the context below to answer the question.
        If the answer is not in the context, say "I don't see that in your report."
        Never guess or make up medical information.

        If you use a medical term, always explain it in plain English in parentheses.

        Context from medical report:
        {context}

        Patient's question: {question}

        Answer in plain, friendly English:"""
    return PromptTemplate(input_variables=["context","question"],template=template)

def build_qa_chain(vectorstore,llm):
    prompt=build_prompt()
    chain=RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                      retriever=vectorstore.as_retriever(search_kwargs={"k":5}),
                                      chain_type_kwargs={"prompt":prompt},
                                      return_source_documents=True)
    print("QA Chain is ready")
    return chain

def ask(chain,question:str):
    result=chain.invoke({"query":question})
    print(f"\nQ:{question}")
    print(f"\nA: {result['result']}")
    print(f"\n------Sources used------")
    for doc in result["source_documents"]:
        print(f"  Page {doc.metadata.get('page', '?')}: {doc.page_content[:80]}...")
    return result

    
    


if __name__=="__main__":
    vectorstore=load_vectorstore()
    llm=load_llm()
    chain=build_qa_chain(vectorstore,llm)

    #question="What does my hemoglobin level mean?"
    #chunks=retrieve_chunks(vectorstore,question)

    ask(chain, "What does my hemoglobin level mean?")
    ask(chain, "Are my kidney results normal?")
    ask(chain, "Should I be worried about anything in this report?")

    #for i, chunk in enumerate(chunks):
    #    print(f"\n--- Chunk {i+1} ---")
    #    print(chunk.page_content)
    #    print("Source:", chunk.metadata)
    



