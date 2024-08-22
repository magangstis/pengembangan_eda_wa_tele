from flask import Flask, request, jsonify  # Mengimpor Flask dan modul untuk menangani permintaan dan respon HTTP
from API_GEMINI import GOOGLE_API_KEY  # Mengimpor kunci API Google dari modul API_GEMINI

# Mengimpor pustaka yang diperlukan dari langchain-community dan langchain-google-genai
from langchain_community.vectorstores import FAISS 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

import re  # Mengimpor modul regex untuk manipulasi teks

import uuid  # Mengimpor UUID untuk menghasilkan ID sesi unik

app = Flask(__name__)  # Membuat instans aplikasi Flask

# Memuat model embedding Google Generative AI dan menginisialisasi FAISS index
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.1, max_tokens=1000, google_api_key=GOOGLE_API_KEY)

# Menyimpan data sesi dalam dictionary
store = {}

def remove_emojis(text):
    """Menghapus emoji dari teks."""
    return re.sub(r'[^\x00-\x7F]+', '', text)

def get_conversational_chain():
    """Membuat dan mengembalikan rantai QA untuk percakapan."""
    
    # Template prompt untuk model percakapan
    prompt_template = """
    Anda adalah EDA (Electronic Data Assistance) yang merupakan aplikasi WhatsApp milik BPS Provinsi Sumatera Utara yang membantu pengguna berkonsultasi dengan pertanyaan statistik dan permintaan data khususnya dari BPS Provinsi Sumatera Utara. Sebagai bagian dari BPS Provinsi Sumatera Utara, Anda tidak boleh mendiskreditkan BPS Provinsi Sumatera Utara, tetapi selalu memberikan citra yang baik dari BPS Provinsi Sumatera Utara.
    
    Hanya dalam percakapan sekali dan pertama kali, Anda akan memberikan penafian bahwa pesan Anda terkirim dalam waktu 10 hingga 20 detik dan melarang kata-kata berbau SARA. Anda juga dapat bertanya mengenai nama dan umur pengguna, dan berbicara sesuai dengan umur pengguna. Jika pengguna berumur lebih dari 30 tahun, Anda memanggil Pak/Bu.

    Anda tidak menerima input berupa audio dan gambar. Output Anda dapat berupa teks atau tabel.

    Jawaban Anda harus sesuai dengan konteks dan tidak memberikan informasi yang salah atau di luar konteks. Jika ada permintaan data di luar konteks, Anda mengatakan bahwa data belum tersedia di layanan Anda, lalu arahkan pengguna ke https://sumut.bps.go.id untuk informasi lebih lanjut.
    
    Question:\n{input}\n
    Context:\n{context}\n

    Answer:
    """

    # Menginisialisasi retriever dengan FAISS
    retriever = vector_store.as_retriever()

    # Membuat prompt untuk percakapan dengan menggunakan template di atas
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ]
    )
    
    # Membuat rantai QA untuk menjawab pertanyaan pengguna
    question_answer_chain = create_stuff_documents_chain(model, prompt)
    
    # Membuat rantai RAG (Retrieval-Augmented Generation)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Mengambil riwayat sesi percakapan berdasarkan session_id. Membuat sesi baru jika belum ada."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()  # Membuat riwayat percakapan baru jika session_id belum ada
    return store[session_id]

def get_response(user_question, session_id):
    """Mendapatkan respons dari model menggunakan RAG."""
    try:
        # Melakukan pencarian kemiripan menggunakan FAISS
        docs = vector_store.similarity_search(user_question)
        
        # Menggunakan rantai QA untuk mendapatkan jawaban yang lebih mendetail
        rag_chain = get_conversational_chain()

        # Mengelola percakapan dengan riwayat pesan
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        # Mengonfigurasi rantai dengan session_id yang diberikan
        config = {"configurable": {"session_id": session_id}}

        # Menjalankan rantai dan mendapatkan respons
        response = conversational_rag_chain.invoke(
            {"input": user_question},
            config=config
        )

        # Mengambil teks jawaban dari respons
        response_text = response.get("answer", "")

        # Menghapus emoji dari respons
        response_text = remove_emojis(response_text)
        
    except Exception as e:
        response_text = f"Error: {str(e)}"  # Mengembalikan pesan error jika terjadi kesalahan
    
    return response_text

@app.route('/process_text', methods=['POST'])
def process_text():
    # Mendapatkan data JSON dari permintaan POST
    data = request.get_json()
    response_text = data.get('response_text')  # Mendapatkan teks respons dari data JSON
    notelp = data.get('notelp')  # Mendapatkan nomor telepon (notelp) dari data JSON jika ada
    
    if response_text:
        # Jika nomor telepon tidak diberikan, buat UUID baru sebagai ID sesi
        if not notelp:
            notelp = str(uuid.uuid4())
        
        # Memproses teks input dan mendapatkan respons dari model
        processed_text = get_response(response_text, notelp)

        # Memastikan teks yang diproses tidak kosong
        if processed_text:
            return jsonify({"status": "success", "processed_text": processed_text, "notelp": notelp}), 200
        else:
            return jsonify({"status": "error", "message": "Failed to process text"}), 500
    return jsonify({"status": "error", "message": "No response text provided"}), 400

if __name__ == '__main__':
    app.run(port=5002)  # Menjalankan server Flask pada port 5002
