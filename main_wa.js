const express = require('express'); // Mengimpor modul Express untuk membuat server web
const bodyParser = require('body-parser'); // Mengimpor body-parser untuk mem-parsing body dari HTTP request
const axios = require('axios'); // Mengimpor axios untuk melakukan HTTP request ke layanan eksternal

const app = express(); // Membuat instance aplikasi Express
const port = process.env.PORT || 3002; // Menentukan port server, menggunakan environment variable PORT jika tersedia, atau default ke 3002

app.use(bodyParser.json()); // Menggunakan body-parser untuk mem-parsing JSON pada body request

// Endpoint untuk menerima pesan dari WhatsApp
app.post('/webhook', async (req, res) => {
    // Mengambil nomor telepon dan teks pesan dari body request
    const { notelp, text } = req.body;

    console.log(`Received message: ${text} from ${notelp}`); // Logging pesan yang diterima

    // Jika nomor telepon atau teks pesan tidak tersedia, kirim respons error
    if (!notelp || !text) {
        console.error('No phone number or message provided');
        return res.status(400).send('No phone number or message provided');
    }

    try {
        // Langkah 1 dan 2: Kirim teks ke Flask untuk diproses
        const flaskUrl = 'http://localhost:5001/get_response'; // URL endpoint Flask
        const flaskPayload = { response_text: text, id: notelp }; // Payload yang dikirim ke Flask
        const flaskResponse = await axios.post(flaskUrl, flaskPayload); // Mengirim request ke Flask

        // Jika Flask tidak merespons dengan status 200, kirim error
        if (flaskResponse.status !== 200) {
            console.error('Failed to get response from Flask');
            return res.status(500).send('Failed to get response from Flask');
        }

        // Langkah 3: Ambil hasil generate content dari Flask
        let generatedContent = flaskResponse.data.response_text; // Mengambil teks yang dihasilkan dari respons Flask

        // Jika tidak ada konten yang dihasilkan dari Flask, kirim error
        if (!generatedContent) {
            console.error('No generated content received from Flask');
            return res.status(500).send('No generated content received from Flask');
        }

        // Mengganti dua tanda ** dengan satu * untuk formatting teks
        generatedContent = generatedContent
            .replace(/\*\*/g, '*')
            .replace(/__/g, '_')
            .replace(/~~/g, '~');

        // Mengganti link markdown dengan URL saja
        generatedContent = generatedContent.replace(/\[.*?\]\((.*?)\)/g, '$1');

        // Langkah 4: Kirim hasil generate content ke WhatsApp melalui API WACONNECT
        const whatsappApiUrl = "https://api-waconnect.bps.web.id/kirim-text"; // URL endpoint API WhatsApp
        const whatsappData = {
            token: "<<Isi sesuai token yang didapatkan dari WAConnect>>", // Token otentikasi API
            notelp: notelp, // Nomor telepon tujuan
            text: generatedContent // Pesan yang akan dikirim
        };

        // Mengirim request ke API WhatsApp
        const wa = await axios.post(whatsappApiUrl, whatsappData, {
            headers: { 'Content-Type': 'application/json' }
        });
        
        if(wa){ 
            console.log('Generated content sent to WhatsApp'); // Log jika pesan berhasil dikirim
        }else{
            console.log('Ngk bisa kirim wa'); // Log jika pengiriman pesan gagal
        }

        res.sendStatus(200); // Kirim respons sukses ke klien
    } catch (error) {
        console.error('Error processing message:', error); // Log error jika terjadi kesalahan
        res.status(500).send('Error processing message'); // Kirim respons error ke klien
    }
});

// Menjalankan server pada port yang ditentukan
app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
