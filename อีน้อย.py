# Updated version for Google Colab
from flask import Flask, request, jsonify, render_template_string
import torch
import os
import gc
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from google.colab import output, drive

# 1. Mount Google Drive เพื่อเข้าถึงไฟล์โมเดล
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

app = Flask(__name__)

# แก้ไข Path ให้ตรงกับที่อยู่ไฟล์ของคุณใน Google Drive
MODEL_PATH = '/content/drive/MyDrive/fine_tuned_tinyllama.pth'

model = None
tokenizer = None
last_activity_time = time.time()

def ensure_folders():
    if not os.path.exists("offload"):
        os.makedirs("offload")

def load_model():
    print("กำลังโหลดโมเดล TinyLLaMA...")
    try:
        global model, tokenizer
        ensure_folders()

        tokenizer = AutoTokenizer.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            offload_folder="offload"
        )

        # ตรวจสอบว่ามีไฟล์ fine-tuned weights หรือไม่
        if os.path.exists(MODEL_PATH):
            print(f"พบไฟล์ Fine-tuned กำลังโหลดจาก: {MODEL_PATH}")
            state_dict = torch.load(MODEL_PATH, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            del state_dict
        else:
            print("⚠️ ไม่พบไฟล์ fine-tuned ใน Drive จะใช้ Base Model แทน")

        model.eval()
        torch.cuda.empty_cache()
        gc.collect()
        print("โหลดโมเดลสำเร็จ!")
        return True
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
        return False

def generate_response(user_input):
    global model, tokenizer
    try:
        # ปรับ Prompt ให้เข้ากับ TinyLlama Chat Format
        prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # ตัดเอาเฉพาะส่วนคำตอบของ AI
        if "<|assistant|>" in full_response:
            response = full_response.split("<|assistant|>")[-1].strip()
        else:
            response = full_response.replace(user_input, "").strip()
        
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# --- UI HTML ---
CHAT_HTML = """
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <title>YaDa AI Chatbot</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { margin: 0; font-family: sans-serif; background: #121212; color: white; display: flex; justify-content: center; height: 100vh; }
        .chat-container { width: 100%; max-width: 600px; display: flex; flex-direction: column; background: #1e1e1e; }
        .chat-header { padding: 20px; background: #7928ca; text-align: center; font-weight: bold; }
        .chat-box { flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; }
        .msg { padding: 10px 15px; border-radius: 15px; max-width: 80%; }
        .user { background: #0072ff; align-self: flex-end; }
        .bot { background: #333; align-self: flex-start; }
        .input-area { padding: 20px; display: flex; border-top: 1px solid #333; }
        input { flex: 1; padding: 10px; border-radius: 5px; border: none; outline: none; }
        button { margin-left: 10px; padding: 10px 20px; background: #7928ca; color: white; border: none; border-radius: 5px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">💬 YaDa AI Chatbot</div>
        <div class="chat-box" id="chatBox">
            <div class="msg bot">สวัสดี! มีอะไรให้ช่วยไหมคะ?</div>
        </div>
        <div class="input-area">
            <input type="text" id="userInput" placeholder="พิมพ์ข้อความ...">
            <button onclick="sendMessage()">ส่ง</button>
        </div>
    </div>
    <script>
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const text = input.value.trim();
            if(!text) return;
            
            const chatBox = document.getElementById('chatBox');
            chatBox.innerHTML += `<div class="msg user">${text}</div>`;
            input.value = '';

            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ input: text })
            });
            const data = await res.json();
            chatBox.innerHTML += `<div class="msg bot">${data.response}</div>`;
            chatBox.scrollTop = chatBox.scrollHeight;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(CHAT_HTML)

@app.route('/api/chat', methods=['POST'])
def chat():
    if model is None:
        if not load_model():
            return jsonify({'response': 'โหลดโมเดลไม่สำเร็จ'})
    
    data = request.json
    user_input = data.get('input', '')
    response = generate_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    # กำหนด Port
    PORT = 5000
    # สร้าง URL สำหรับคลิกเข้าชมหน้าเว็บใน Colab
    output.serve_kernel_port_as_window(PORT)
    print(f"คลิกที่ลิงก์ด้านบนเพื่อเปิดหน้าเว็บ (รอจนกว่า Flask จะทำงาน)")
    app.run(port=PORT)
