from flask import Flask, request, jsonify
import google.generativeai as genai
from API_GEMINI import GOOGLE_API_KEY
import vertexai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import re
import uuid  # Import UUID for generating unique session IDs

app = Flask(__name__)

# Configure the Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

project_id = "gen-lang-client-0905428476"
location = "us-central1"

vertexai.init(project=project_id, location=location)

generation_config = {
    "temperature": 0.1,
    "top_p": 0.5,
    "max_output_tokens": 1000,
    "response_mime_type": "text/plain",
}

# Load FAISS index
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-exp-0827", temperature = 0.1, max_tokens = None, google_api_key=GOOGLE_API_KEY)

# Store session data
store = {}

def remove_emojis(text):
    """Remove emojis from text."""
    return re.sub(r'[^\x00-\x7F]+', '', text)

def get_conversational_chain():
    """Create and return a QA chain."""
    prompt_template = """
    Anda adalah EDA (Electronic Data Assistance) pada aplikasi WhatsApp yang membantu pengguna berkonsultasi dengan pertanyaan statistik dan melayani permintaan data khususnya dari BPS Provinsi Sumatera Utara. Sebagai kaki tangan BPS Provinsi Sumatera Utara, Anda tidak boleh mendiskreditkan BPS Provinsi Sumatera Utara. Anda juga meyakinkan pengguna bahwa data yang Anda peroleh benar adanya.
    
    Informasi yang perlu Anda ketahui jika ada pengguna yang bertanya adalah Kepala BPS Provinsi Sumatera Utara adalah Asim Saputra, SST, M.Ec.Dev. Kantor BPS Provinsi Sumatera Utara berlokasi di Jalan Asrama No. 179, Dwikora, Medan Helvetia, Medan, Sumatera Utara 20123. Visi BPS pada tahun 2024 adalah menjadi penyedia data statistik berkualitas untuk Indonesia Maju. Misi BPS pada tahun 2024 meliputi: 1) Menyediakan statistik berkualitas yang berstandar
    nasional dan internasional; 2) Membina K/L/D/I melalui Sistem Statistik Nasional yang berkesinambungan; 3) Mewujudkan pelayanan prima di bidang statistik untuk terwujudnya Sistem Statistik Nasional; 4) Membangun SDM yang unggul dan adaptif berlandaskan nilai profesionalisme, integritas, dan amanah.

    Hanya dalam percakapan sekali dan pertama kali, Anda akan memberikan penafian bahwa pesan Anda terkirim dalam waktu 10 hingga 20 detik, riwayat chat akan terhapus tiap jam, melarang kata-kata berbau SARA, dan menghimbau pengguna untuk menggunakan kalimat yang lengkap dilengkapi wilayah dan tahun data serta tidak menggunakan singkatan atau akronim untuk data yang akurat. Anda juga dapat bertanya mengenai nama dan umur pengguna, dan berbicara sesuai dengan umur pengguna. Jika pengguna berumur lebih dari 30 tahun, Anda memanggil Pak/Bu.

    Anda tidak menerima input berupa audio dan gambar. Anda menerima input penerimaan data dari pengguna dengan format wilayah dan tahun saja. Jika ada pengguna meminta data diluar format, Anda memberikan saran format yang benar. Output Anda dapat berupa teks atau tabel.

    Anda berikan jawaban yang relevan dan ringkas berdasarkan dokumen di bawah ini dan pertanyaan dari pengguna. Anda juga tidak memberikan contoh data di luar dokumen. Jika ada permintaan data di luar dokumen, arahkan pengguna ke https://sumut.bps.go.id atau Pelayanan Statistik Terpadu (PST) di BPS Provinsi Sumatera Utara untuk informasi lebih lanjut. Anda memberikan alasan ketidatersediaan data berasal dari keterbatasan Anda dalam membaca keseluruhan data BPS Provinsi Sumatera Utara yang masif lalu arahkan pengguna ke https://sumut.bps.go.id atau Pelayanan Statistik Terpadu (PST) di BPS Provinsi Sumatera Utara untuk informasi lebih lanjut. Jika ada yang bisa dihubungi, Anda menyarankan mengunjungi kantor atau PST melalui pst1200@bps.go.id.
    
    Context:\n {context}\n
    Pertanyaan Pengguna: \n{input}\n    
    Jawaban yang relevan (berdasarkan dokumen):\n
    """

    # retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.1})
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(model, prompt)

    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieve the session history for a given session_id. Create a new one if it does not exist."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_response(user_question, session_id):
    """Get response from the model using RAG."""
    try:
        # Perform similarity search using FAISS
        docs = vector_store.similarity_search(user_question)

        # Use the QA chain to get a detailed answer
        rag_chain = get_conversational_chain()

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        # Set configuration with the given session_id
        config = {"configurable": {"session_id": session_id}}

        # Invoke the chain and get the response
        response = conversational_rag_chain.invoke(
            {"input": user_question},
            config=config
        )

        # Extract the answer text
        response_text = response.get("answer", "")

        # Check for specific response codes
        response_code = response.get("response_code", 200)
        if response_code == 429:
            return "Maaf, layanan saat ini sedang sibuk. Silakan coba lagi nanti."
        elif response_code == 503:
            return "Layanan saat ini tidak tersedia. Silakan coba lagi nanti."

        # Remove emojis from the response
        response_text = remove_emojis(response_text)

    except Exception as e:
        response_text = f"Error: {str(e)}"

    return response_text

@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.get_json()
    response_text = data.get('response_text')
    notelp = data.get('notelp')  # Get the notelp from the request, if provided
    
    if response_text:
        # If notelp is not provided, generate a new one
        if not notelp:
            notelp = str(uuid.uuid4())

        # Process the input text and get the response
        processed_text = get_response(response_text, notelp)

        # Ensure processed_text is not empty
        if processed_text:
            return jsonify({"status": "success", "processed_text": processed_text, "notelp": notelp}), 200
        else:
            return jsonify({"status": "error", "message": "Failed to process text"}), 500
    return jsonify({"status": "error", "message": "No response text provided"}), 400

if __name__ == '__main__':
    app.run(port=5002)  # Change port if needed
