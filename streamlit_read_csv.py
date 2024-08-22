import os  # Mengimpor modul os untuk berinteraksi dengan sistem operasi
import pandas as pd  # Mengimpor pandas untuk manipulasi dan analisis data
import streamlit as st  # Mengimpor streamlit untuk membuat antarmuka web
import io  # Mengimpor io untuk menangani input/output file
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Mengimpor text splitter untuk membagi teks menjadi bagian yang lebih kecil
from API_GEMINI import GOOGLE_API_KEY  # Mengimpor kunci API Google dari file API_GEMINI
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Mengimpor embedding dari Google Generative AI
import google.generativeai as genai  # Mengimpor library generative AI dari Google
from langchain.vectorstores import FAISS  # Mengimpor FAISS untuk pencarian berbasis vektor
from langchain_google_genai import ChatGoogleGenerativeAI  # Mengimpor model chat dari Google Generative AI
from langchain.prompts import PromptTemplate  # Mengimpor template prompt dari LangChain
from langchain.chains.question_answering import load_qa_chain  # Mengimpor chain untuk penjawaban pertanyaan
from langchain_core.documents import Document  # Mengimpor kelas Document untuk menyimpan teks dan metadata
from uuid import uuid4  # Mengimpor uuid4 untuk menghasilkan ID unik

# Mengonfigurasi API Google Generative AI dengan kunci API yang disediakan
genai.configure(api_key=GOOGLE_API_KEY)

def preprocess_text(text):
    """
    Memproses teks yang diekstrak dari PDF agar dapat digunakan oleh LangChain.
    
    Args:
        text (str): Teks yang akan diproses.
    
    Returns:
        List[str]: Daftar bagian teks yang telah dipecah menjadi beberapa chunk.
    """
    if not text:
        return []
    
    # Membagi teks menjadi chunk berdasarkan ukuran tertentu dengan overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000, separators=[",","\n","\n\n"], length_function=len)
    return text_splitter.split_text(text)

def load_csv_files_with_metadata(csv_files):
    """
    Memuat file CSV, menambahkan metadata, dan mengembalikan dokumen dengan metadata.
    
    Args:
        csv_files (list): Daftar file CSV yang diunggah.
    
    Returns:
        List[Document]: Daftar objek dokumen dengan konten dan metadata.
    """
    all_documents = []
    
    for file in csv_files:
        try:
            # Membaca konten file CSV dari UploadedFile object
            file_content = file.read()  # Mengembalikan konten file dalam bentuk bytes
            
            # Mengonversi konten file menjadi pandas DataFrame
            df = pd.read_csv(io.StringIO(file_content.decode('utf-8')), skiprows=1)
            
            # Mengekstrak nama file tanpa ekstensi
            file_name = os.path.basename(file.name).split('.')[0]
            
            # Iterasi setiap baris dalam DataFrame dan membuat objek Document dengan metadata
            for index, row in df.iterrows():
                metadata = {
                    "source": file_name,
                }
                
                # Menggabungkan data baris menjadi teks yang bermakna
                content = f"{file_name} dari {row['input']} tahun {row['tahun']} adalah {row['output']}."
                
                # Membuat objek Document dengan konten dan metadata
                document = Document(page_content=content, metadata=metadata)
                
                # Menambahkan dokumen ke daftar
                all_documents.append(document)
        
        except Exception as e:
            st.error(f"Error loading {file.name}: {e}")
    
    return all_documents

def create_or_update_vector_store(documents, vector_store_path="faiss_index"):
    """
    Membuat atau memperbarui vector store dengan dokumen yang diberikan.
    
    Args:
        documents (list): Daftar objek dokumen yang akan disimpan dalam vector store.
        vector_store_path (str): Path untuk menyimpan vector store.
    
    Returns:
        FAISS: Objek FAISS yang berisi vector store.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    
    try:
        # Memeriksa apakah vector store sudah ada
        if os.path.exists(vector_store_path):
            # Memuat vector store yang sudah ada
            vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
            st.info("Existing vector store loaded.")
        else:
            # Membuat vector store baru
            vector_store = FAISS(embedding=embeddings)
            st.info("New vector store created.")
        
        # Menghasilkan ID unik untuk dokumen
        uuids = [str(uuid4()) for _ in range(len(documents))]
        
        # Menambahkan dokumen baru ke vector store
        vector_store.add_documents(documents=documents, ids=uuids)
        
        # Menyimpan vector store yang diperbarui
        vector_store.save_local(vector_store_path)
        st.success("Vector store updated successfully.")
        
    except Exception as e:
        st.error(f"Error creating or updating vector store: {e}")
        return None
    
    return vector_store

def get_conversational_chain():
    """
    Membuat dan mengembalikan chain untuk penjawab pertanyaan.
    
    Returns:
        Chain: Objek chain untuk menjawab pertanyaan berdasarkan konteks.
    """
    prompt_template = """
    Anda adalah EDA (Electronic Data Assistance) pada aplikasi WhatsApp yang membantu pengguna berkonsultasi dengan pertanyaan statistik dan permintaan data khususnya dari BPS Provinsi Sumatera Utara. Sebagai kaki tangan BPS Provinsi Sumatera Utara, Anda tidak boleh mendiskreditkan BPS Provinsi Sumatera Utara.

    Anda tidak menerima input berupa audio dan gambar.

    Jawaban Anda harus sesuai dengan konteks dan tidak memberikan informasi yang salah atau di luar konteks. Jika ada permintaan data di luar konteks, arahkan pengguna ke https://sumut.bps.go.id untuk informasi lebih lanjut.
    
    Context:\n {context}\n
    Question: \n{question}\n
    
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.1, google_api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def handle_user_input(user_question):
    """
    Mengolah input pengguna dan mendapatkan respons dari model.
    
    Args:
        user_question (str): Pertanyaan dari pengguna.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
    
    try:
        # Memuat vector store yang sudah ada
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)  # Mencari dokumen yang mirip dengan pertanyaan pengguna
        
        # Mendapatkan chain untuk penjawab pertanyaan
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        st.write("Reply:", response["output_text"])  # Menampilkan jawaban dari model
        
    except Exception as e:
        st.error(f"Error processing user input: {e}")

def main():
    """
    Fungsi utama untuk menjalankan aplikasi Streamlit.
    """
    st.set_page_config(page_title="Chat CSV")  # Mengatur konfigurasi halaman
    st.header("Chat with CSV using Gemini")  # Menampilkan header aplikasi
    
    user_question = st.text_input("Ask a Question from the CSV Files")  # Menampilkan input teks untuk pertanyaan pengguna
    
    if user_question:
        handle_user_input(user_question)  # Mengolah input pengguna jika ada
    
    with st.sidebar:
        st.title("Menu: ")
        
        # Memeriksa apakah file CSV sudah disimpan di session state
        if 'csv_files' not in st.session_state:
            st.session_state.csv_files = []

        # Widget untuk mengunggah file
        uploaded_files = st.file_uploader("Upload your CSV Files", accept_multiple_files=True)
        
        if uploaded_files:
            st.session_state.csv_files = uploaded_files
        
        if st.button("Submit & Process"):
            with st.spinner("Processing: "):
                if st.session_state.csv_files:
                    documents = load_csv_files_with_metadata(st.session_state.csv_files)
                    if documents:
                        create_or_update_vector_store(documents)  # Membuat atau memperbarui vector store dengan dokumen yang diunggah
                        
                        # Menampilkan informasi file yang diunggah
                        st.write(f"Number of files uploaded: {len(st.session_state.csv_files)}")
                        st.write("Files uploaded:")
                        for file in st.session_state.csv_files:
                            st.write(f"- {file.name}")
                        
                        st.success("Processing Complete")  # Menampilkan pesan sukses setelah proses selesai
                else:
                    st.error("Please upload CSV files.")  # Menampilkan pesan kesalahan jika tidak ada file yang diunggah
                
if __name__ == "__main__":
    main()  # Menjalankan aplikasi Streamlit
