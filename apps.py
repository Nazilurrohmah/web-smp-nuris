import gradio as gr
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import json
import random

# 1. Muat Model dan Tokenizer IndoBERT
MODEL_PATH = "./model_bert_final_smp"
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model.eval()

# 2. Muat Peta Label Index
with open('label_map.json', 'r') as f:
    label_map = json.load(f)

# 3. Muat Basis Data Jawaban dari JSON asli
with open('dataakhirchatbot.json', 'r') as f:
    intents_file = json.load(f)
    intents_data = intents_file['intents'] if 'intents' in intents_file else intents_file

# 4. Fungsi Utama Prediksi Chatbot
def tanggapan_bot(pesan_user, history):
    # Tokenisasi teks input
    inputs = tokenizer(pesan_user, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
        
    # Ubah indeks angka ke nama tag string
    predicted_tag = label_map.get(str(prediction), "unknown")
    
    # Cari balasan teks yang cocok di intents.json
    balasan = "Maaf, bot belum memahami maksud pertanyaan Anda. Silakan hubungi admin sekolah."
    for intent in intents_data:
        if intent['tag'] == predicted_tag:
            balasan = random.choice(intent['responses'])
            break
            
    return balasan

# 5. Desain Antarmuka Chat Otomatis Premium (Gradio)
demo = gr.ChatInterface(
    fn=tanggapan_bot,
    title="🤖 Chatbot AI - SMP Nurul Islam Probolinggo",
    description="Sistem Informasi Pelayanan Akademik Berbasis NLP IndoBERT. Silakan tanyakan seputar pendaftaran, biaya, atau profil sekolah.",
    textbox=gr.Textbox(placeholder="Ketik pertanyaan Anda di sini...", container=False, scale=7),
    theme="soft",
    submit_btn="Kirim ➔",
    clear_btn="Hapus Riwayat 🗑️",
    retry_btn=None,
    undo_btn=None
)

if __name__ == "__main__":
    demo.launch()
