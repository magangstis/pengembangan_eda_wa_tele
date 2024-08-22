const express = require('express'); // Mengimpor modul Express untuk membuat server web
const bodyParser = require('body-parser'); // Mengimpor body-parser untuk mem-parsing body dari HTTP request
const axios = require('axios'); // Mengimpor axios untuk melakukan HTTP request ke layanan eksternal
const { Telegraf } = require('telegraf'); // Mengimpor Telegraf untuk berinteraksi dengan Telegram API

const app = express(); // Membuat instance aplikasi Express
const port = process.env.PORT || 3003; // Menentukan port server, menggunakan environment variable PORT jika tersedia, atau default ke 3003

// Inisialisasi Telegram bot
const TELEGRAM_BOT_TOKEN = '7294945219:AAFGWI5FippG8nE4bS4Pog2o2v9D8V5IcWo'; // Token untuk bot Telegram
const bot = new Telegraf(TELEGRAM_BOT_TOKEN); // Membuat instance Telegraf dengan token bot

app.use(bodyParser.json()); // Menggunakan body-parser untuk mem-parsing JSON pada body request

// Endpoint untuk menerima pesan dari Telegram
bot.on('text', async (ctx) => {
    // Mengambil teks pesan dan username pengirim dari objek context
    const text = ctx.message.text;
    const username = ctx.message.from.username;

    console.log(`Received message: ${text} from @${username}`); // Logging pesan yang diterima

    // Jika username atau teks pesan tidak tersedia, kirim respons kesalahan
    if (!username || !text) {
        console.error('No username or message provided');
        return ctx.reply('No username or message provided');
    }

    try {
        // Langkah 1 dan 2: Kirim teks ke Flask untuk diproses
        const flaskUrl = 'http://localhost:5001/get_response'; // URL endpoint Flask
        const flaskPayload = { response_text: text, id: username }; // Payload yang dikirim ke Flask
        const flaskResponse = await axios.post(flaskUrl, flaskPayload); // Mengirim request ke Flask

        // Jika Flask tidak merespons dengan status 200, kirim pesan kesalahan ke Telegram
        if (flaskResponse.status !== 200) {
            console.error('Failed to get response from Flask');
            return ctx.reply('Failed to get response from Flask');
        }

        // Langkah 3: Ambil hasil generate content dari Flask
        let generatedContent = flaskResponse.data.response_text; // Mengambil teks yang dihasilkan dari respons Flask

        // Jika tidak ada konten yang dihasilkan dari Flask, kirim pesan kesalahan ke Telegram
        if (!generatedContent) {
            console.error('No generated content received from Flask');
            return ctx.reply('No generated content received from Flask');
        }

        // Mengganti dua tanda ** dengan satu * untuk formatting teks
        generatedContent = generatedContent
            .replace(/\*\*/g, '*')
            .replace(/__/g, '_')
            .replace(/~~/g, '~');

        // Mengganti link markdown dengan URL saja
        generatedContent = generatedContent.replace(/\[.*?\]\((.*?)\)/g, '$1');

        // Langkah 4: Kirim hasil generate content ke pengguna Telegram
        await ctx.reply(generatedContent); // Mengirimkan pesan yang telah diproses ke Telegram
        console.log('Generated content sent to Telegram'); // Logging konfirmasi bahwa pesan berhasil dikirim

    } catch (error) {
        // Jika terjadi error selama pemrosesan, kirim pesan kesalahan ke Telegram
        console.error('Error processing message:', error);
        await ctx.reply('Error processing message');
    }
});

// Meluncurkan bot Telegram dan logging bahwa bot sedang berjalan
bot.launch().then(() => {
    console.log('Telegram bot is running...');
});

// Menjalankan server pada port yang ditentukan
app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
