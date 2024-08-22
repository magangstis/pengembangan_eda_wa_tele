import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from API_GEMINI import GOOGLE_API_KEY
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import PyPDF2
from uuid import uuid4
from langchain_core.documents import Document

# Konfigurasi API Google Generative AI dengan API key yang telah disediakan
genai.configure(api_key=GOOGLE_API_KEY)

def load_pdf_files(pdf_files):
    """Memuat dan mengekstrak teks dari file PDF menggunakan PyPDF2.
    
    Args:
        pdf_files (list): Daftar file PDF yang diunggah pengguna.
        
    Returns:
        str: Gabungan teks dari semua file PDF yang valid.
    """
    all_text = []
    for file in pdf_files:
        try:
            # Membaca file PDF menggunakan PyPDF2
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            # Mengekstrak teks dari setiap halaman dalam PDF
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
                
            # Menambahkan teks yang berhasil diambil ke dalam list
            if text.strip():
                all_text.append(text)
            else:
                st.warning(f"File {file.name} kosong atau tidak mengandung teks yang dapat dibaca.")
                
        except Exception as e:
            st.warning(f"Gagal membaca file PDF {file.name}: {e}")
    
    if all_text:
        return "\n".join(all_text)
    else:
        st.error("Tidak ada file PDF yang valid diunggah.")
        return ""

def preprocess_text(text):
    """Memproses teks yang diekstrak dari PDF untuk digunakan dalam LangChain.
    
    Args:
        text (str): Teks yang diekstrak dari file PDF.
        
    Returns:
        list: Daftar potongan teks setelah dipecah menjadi chunk.
    """
    if not text:
        return []
    
    # Memecah teks menjadi chunk dengan ukuran maksimal 10.000 karakter dan overlap 1.000 karakter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000, length_function=len)
    return text_splitter.split_text(text)

def create_or_update_vector_store(text_chunks, vector_store_path="faiss_index"):
    """Membuat atau memperbarui vektor store dengan potongan teks yang diberikan.
    
    Args:
        text_chunks (list): Daftar potongan teks untuk dimasukkan ke vektor store.
        vector_store_path (str): Jalur penyimpanan vektor store (default: "faiss_index").
        
    Returns:
        FAISS: Objek vektor store yang diperbarui.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    
    try:
        # Memeriksa apakah vektor store sudah ada
        if os.path.exists(vector_store_path):
            # Memuat vektor store yang ada
            vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
            st.info("Vektor store yang ada berhasil dimuat.")
        else:
            # Membuat vektor store baru jika tidak ada
            vector_store = FAISS(embedding=embeddings)
            st.info("Vektor store baru berhasil dibuat.")
        
        # Membuat objek Document untuk setiap chunk teks
        documents = [Document(page_content=chunk) for chunk in text_chunks]
        
        # Menghasilkan UUID untuk setiap dokumen
        uuids = [str(uuid4()) for _ in range(len(documents))]
        
        # Menambahkan dokumen ke vektor store
        vector_store.add_documents(documents=documents, ids=uuids)
        
        # Menyimpan vektor store
        vector_store.save_local(vector_store_path)
        st.success("Vektor store berhasil diperbarui.")
        
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membuat atau memperbarui vektor store: {e}")
        return None
    
    return vector_store

def get_conversational_chain():
    """Membuat dan mengembalikan rantai QA untuk menjawab pertanyaan pengguna.
    
    Returns:
        LLMChain: Objek rantai QA untuk penjawaban pertanyaan.
    """
    prompt_template = """
    Anda adalah EDA (Electronic Data Assistance) pada aplikasi WhatsApp yang membantu pengguna berkonsultasi dengan pertanyaan statistik dan permintaan data khususnya dari BPS Provinsi Sumatera Utara. Sebagai kaki tangan BPS Provinsi Sumatera Utara, Anda tidak boleh mendiskreditkan BPS Provinsi Sumatera Utara.

    Anda tidak menerima input berupa audio dan gambar.

    Jawaban Anda harus sesuai dengan konteks dan tidak memberikan informasi yang salah atau di luar konteks. Jika ada permintaan data di luar konteks, arahkan pengguna ke https://sumut.bps.go.id untuk informasi lebih lanjut.
    
    Context:\n {context}\n
    Question: \n{question}\n
    
    Answer:
    """
    
    # Menginisialisasi model generatif Google untuk chat
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.1, google_api_key=GOOGLE_API_KEY)
    # Membuat template prompt dengan variabel input
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    # Membuat rantai QA dengan model dan template prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def handle_user_input(user_question):
    """Memproses input pengguna dan menghasilkan respons.
    
    Args:
        user_question (str): Pertanyaan yang diajukan oleh pengguna.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
    
    try:
        # Memuat vektor store yang ada
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        # Mencari dokumen yang relevan berdasarkan pertanyaan pengguna
        docs = new_db.similarity_search(user_question)
        
        # Mendapatkan rantai QA
        chain = get_conversational_chain()
        # Mendapatkan respons dari rantai QA
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        # Menampilkan respons di aplikasi Streamlit
        st.write("Reply:", response["output_text"])
        
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses input pengguna: {e}")

def main():
    # Mengatur konfigurasi halaman aplikasi Streamlit
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat dengan PDF menggunakan Gemini")
    
    # Widget input teks untuk pertanyaan pengguna
    user_question = st.text_input("Ajukan Pertanyaan dari File PDF")
    
    if user_question:
        handle_user_input(user_question)
    
    with st.sidebar:
        st.title("Menu: ")
        
        # Memeriksa apakah file PDF disimpan dalam session state
        if 'pdf_files' not in st.session_state:
            st.session_state.pdf_files = []

        # Widget pengunggah file
        uploaded_files = st.file_uploader("Unggah File PDF Anda", accept_multiple_files=True)
        
        if uploaded_files:
            st.session_state.pdf_files = uploaded_files
        
        if st.button("Submit & Process"):
            with st.spinner("Memproses: "):
                if st.session_state.pdf_files:
                    # Memuat, memproses, dan menyimpan teks dari file PDF
                    pdf_text = load_pdf_files(st.session_state.pdf_files)
                    text_chunks = preprocess_text(pdf_text)
                    
                    if text_chunks:
                        create_or_update_vector_store(text_chunks)
                        
                        # Menampilkan informasi file
                        st.write(f"Jumlah file yang diunggah: {len(st.session_state.pdf_files)}")
                        st.write("File yang diunggah:")
                        for file in st.session_state.pdf_files:
                            st.write(f"- {file.name}")
                        
                        st.success("Proses Selesai")
                else:
                    st.error("Harap unggah file PDF.")
                
if __name__ == "__main__":
    main()
