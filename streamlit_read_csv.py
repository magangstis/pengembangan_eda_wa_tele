import os
import pandas as pd
import streamlit as st
import io
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from API_GEMINI import GOOGLE_API_KEY
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_core.documents import Document
from uuid import uuid4
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss

genai.configure(api_key=GOOGLE_API_KEY)

def load_csv_files_with_metadata(csv_files):
    """Load CSV files, add metadata, and return documents with metadata."""
    all_documents = []
    
    for file in csv_files:
        try:
            # Read the CSV data from the UploadedFile object
            file_content = file.read()  # This returns the content of the file as bytes
            
            # Convert the file content to a pandas DataFrame
            df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
            
            # Extract the file name without extension
            file_name = os.path.basename(file.name).split('.')[0]
            
            # Iterate over each row in the dataframe and create Document objects with metadata
            for index, row in df.iterrows():
                metadata = {
                    "source": file_name,
                }

                # Ensure 'turvar' is properly handled
                turvar = row['turvar'] if 'turvar' in df.columns and pd.notna(row['turvar']) and row['turvar'] != "" else None

                # Handle 'vervar', 'datacontent', and 'tahun'
                vervar = row['vervar'] if 'vervar' in df.columns and pd.notna(row['vervar']) else "Wilayah tidak tersedia"
                datacontent = row['datacontent'] if 'datacontent' in df.columns and pd.notna(row['datacontent']) else "Data tidak tersedia"
                tahun = row['tahun'] if 'tahun' in df.columns and pd.notna(row['tahun']) else "Tahun tidak tersedia"

                # Determine the content based on the presence of 'turvar'
                if turvar:
                    content = f"{file_name}, {tahun}, {turvar} untuk {vervar}, {datacontent}."
                else:
                    content = f"{file_name}, {tahun}, {vervar}, {datacontent}."

                # Create a Document object with content and metadata
                document = Document(page_content=content, metadata=metadata)
                
                # Add document to the list
                all_documents.append(document)
        
        except Exception as e:
            st.error(f"Error loading {file.name}: {e}")
    
    return all_documents

def create_or_update_vector_store(documents, vector_store_path="faiss_index", batch_size=1000):
    """Create or update a vector store with the given documents in batches."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
    
    try:
        if os.path.exists(vector_store_path):
            vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
            st.info("Existing vector store loaded.")
        else:
            index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
            vector_store = FAISS(embedding_function=embeddings, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})
            st.info("New vector store created.")
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            uuids = [str(uuid4()) for _ in range(len(batch_docs))]
            vector_store.add_documents(documents=batch_docs, ids=uuids)
        
        vector_store.save_local(vector_store_path)
        st.success("Vector store updated successfully.")
        
    except Exception as e:
        st.error(f"Error creating or updating vector store: {e}")
        return None
    
    return vector_store

def get_conversational_chain():
    """Create and return a QA chain."""
    prompt_template = """
    Anda adalah EDA (Electronic Data Assistance) pada aplikasi WhatsApp yang membantu pengguna berkonsultasi dengan pertanyaan statistik dan permintaan data khususnya dari BPS Provinsi Sumatera Utara. Sebagai kaki tangan BPS Provinsi Sumatera Utara, Anda tidak boleh mendiskreditkan BPS Provinsi Sumatera Utara. Kepala BPS Provinsi Sumatera Utara adalah Asim Saputra, SST, M.Ec.Dev. Kantor BPS Provinsi Sumatera Utara berlokasi di Jalan Asrama No. 179, Dwikora, Medan Helvetia, Medan, Sumatera Utara 20123.

    Visi BPS pada tahun 2024 adalah menjadi penyedia data statistik berkualitas untuk Indonesia
    Maju.
    Misi BPS pada tahun 2024 meliputi: 1) Menyediakan statistik berkualitas yang berstandar
    nasional dan internasional; 2) Membina K/L/D/I melalui Sistem Statistik Nasional yang
    berkesinambungan; 3) Mewujudkan pelayanan prima di bidang statistik untuk terwujudnya
    Sistem Statistik Nasional; 4) Membangun SDM yang unggul dan adaptif berlandaskan nilai
    profesionalisme, integritas, dan amanah.

    Anda tidak menerima input berupa audio dan gambar.

    Jawaban Anda harus sesuai dengan konteks dan tidak memberikan informasi yang salah atau di luar konteks. Jika ada permintaan data di luar konteks, arahkan pengguna ke https://sumut.bps.go.id untuk informasi lebih lanjut.
    
    Context:\n {context}\n
    Question: \n{question}\n
    
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-exp-0827", temperature=0.1, google_api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def handle_user_input(user_question):
    """Handle user input and get response."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
    
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        st.write("Reply:", response["output_text"])
        
    except Exception as e:
        st.error(f"Error processing user input: {e}")

def main():
    st.set_page_config(page_title="Chat CSV")
    st.header("Chat with CSV using Gemini")
    
    user_question = st.text_input("Ask a Question from the CSV Files")
    
    if user_question:
        handle_user_input(user_question)
    
    with st.sidebar:
        st.title("Menu: ")
        
        # Check if CSV files are stored in session state
        if 'csv_files' not in st.session_state:
            st.session_state.csv_files = []

        # File uploader widget
        uploaded_files = st.file_uploader("Upload your CSV Files", accept_multiple_files=True)
        
        if uploaded_files:
            st.session_state.csv_files = uploaded_files
        
        if st.button("Submit & Process"):
            with st.spinner("Processing: "):
                if st.session_state.csv_files:
                    documents = load_csv_files_with_metadata(st.session_state.csv_files)
                    if documents:
                        create_or_update_vector_store(documents)
                        
                        # Display file info
                        st.write(f"Number of files uploaded: {len(st.session_state.csv_files)}")
                        st.write("Files uploaded:")
                        for file in st.session_state.csv_files:
                            st.write(f"- {file.name}")
                        
                        st.success("Processing Complete")
                else:
                    st.error("Please upload CSV files.")
                
if __name__ == "__main__":
    main()
