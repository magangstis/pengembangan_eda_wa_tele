from flask import Flask, request, jsonify  # Mengimpor Flask untuk membuat aplikasi web, dan request, jsonify untuk menangani dan merespons permintaan HTTP
import requests  # Mengimpor modul requests untuk melakukan permintaan HTTP ke server lain
import logging  # Mengimpor modul logging untuk mencatat log aplikasi

app = Flask(__name__)  # Membuat instans aplikasi Flask

# Konfigurasi Logging
logging.basicConfig(level=logging.DEBUG)  # Mengatur logging untuk mencatat semua pesan debug dan pesan yang lebih serius

def send_to_main(response_text, session_id):
    """
    Mengirim response_text dan session_id ke server utama (main.py) untuk diproses.
    
    Args:
        response_text (str): Teks respons yang akan diproses.
        session_id (str): ID sesi untuk identifikasi sesi pengguna.

    Returns:
        str: Teks yang telah diproses oleh server utama, atau None jika terjadi kesalahan.
    """
    main_url = 'http://localhost:5002/process_text'  # URL server utama (main.py), sesuaikan dengan URL server Anda
    payload = {'response_text': response_text, 'notelp': session_id}  # Data yang akan dikirim ke server utama

    try:
        logging.debug(f"Sending to main.py. Payload: {payload}")  # Mencatat payload sebelum dikirim
        response = requests.post(main_url, json=payload)  # Mengirim permintaan POST ke server utama dengan payload JSON
        logging.debug(f"Main.py response status: {response.status_code}, Response: {response.text}")  # Mencatat status dan respons dari server utama
        
        if response.status_code == 200:  # Memeriksa apakah permintaan berhasil
            return response.json().get('processed_text', '')  # Mengembalikan teks yang telah diproses dari respons JSON
        else:
            logging.error(f"Failed to get processed response from main.py. Status code: {response.status_code}, Response: {response.text}")
            return None  # Mengembalikan None jika terjadi kesalahan pada server utama
    except Exception as e:
        logging.error(f"Error sending data to main.py: {str(e)}")  # Mencatat kesalahan jika terjadi masalah saat mengirim data ke server utama
        return None  # Mengembalikan None jika terjadi kesalahan

@app.route('/get_response', methods=['POST'])
def get_response():
    """
    Menerima teks dari pengguna, mengirimkannya ke server utama untuk diproses, 
    dan mengembalikan teks yang telah diproses sebagai respons.
    
    Returns:
        JSON: Respons berisi teks yang telah diproses atau pesan kesalahan.
    """
    data = request.get_json()  # Mengambil data JSON dari permintaan POST
    response_text = data.get('response_text')  # Mendapatkan teks respons dari data JSON
    session_id = data.get('id')  # Mendapatkan session_id dari data JSON
    
    if response_text and session_id:  # Memeriksa apakah teks respons dan session_id tersedia
        logging.debug(f"Received response text: {response_text} with notelp: {session_id}")  # Mencatat teks respons dan session_id yang diterima

        # Mengirim teks respons dan session_id ke server utama untuk diproses
        processed_text = send_to_main(response_text, session_id)
        
        # Mengembalikan teks yang telah diproses sebagai respons
        return jsonify({"status": "success", "response_text": processed_text}), 200

    logging.error("No response text or session_id provided")  # Mencatat kesalahan jika teks respons atau session_id tidak tersedia
    return jsonify({"status": "error", "message": "No response text or session_id provided"}), 400  # Mengembalikan pesan kesalahan jika input tidak lengkap

if __name__ == '__main__':
    app.run(port=5001)  # Menjalankan server Flask pada port 5001, sesuaikan jika diperlukan
