# Updated version for Public GitHub Repository
# Supports both Google Colab and Local Environments

from flask import Flask, request, jsonify, render_template_string
import torch
import os
import gc
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

# --- Configuration ---
# หากรันใน Colab ให้ใช้ Path ของ Drive / หากรันทั่วไปให้วางไฟล์ไว้ที่เดียวกับโค้ด
MODEL_PATH = '/content/drive/MyDrive/fine_tuned_tinyllama.pth' if os.path.exists('/content/drive') else 'fine_tuned_tinyllama.pth'

model = None
tokenizer = None
last_activity_time = time.time()

def setup_colab_tunnel(port):
    """ฟังก์ชันสำหรับสร้างลิงก์เข้าหน้าเว็บเมื่อรันบน Google Colab"""
    try:
        from google.colab import output
        output.serve_kernel_port_as_window(port)
        from google.colab.output import eval_js
        print(f"--- Colab Environment Detected ---")
        print(f"คลิกที่ลิงก์นี้เพื่อเปิดหน้าเว็บ: {eval_js(f'google.colab.kernel.proxyPort({port})')}")
    except ImportError:
        print(f"--- Local Environment Detected ---")
        print(f"เข้าใช้งานผ่าน: http://127.0.0.1:{port}")

def ensure_folders():
    if not os.path.exists("offload"):
        os.makedirs("offload")

def load_model():
    print("Loading TinyLLaMA model...")
    global model, tokenizer
    try:
        ensure_folders()
        
        # ใช้ Base Model จาก HuggingFace
        repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        
        # ตั้งค่าการใช้ Hardware (GPU/CPU)
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
            low_cpu_mem_usage=True,
            offload_folder="offload"
        )

        # โหลดไฟล์ Weights ถ้ามีอยู่จริง
        if os.path.exists(MODEL_PATH):
            print(f"Loading weights from {MODEL_PATH}")
            state_dict = torch.load(MODEL_PATH, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            del state_dict
        else:
            print("Notice: Running with Base Model (Weights file not found)")

        model.eval()
        torch.cuda.empty_cache()
        gc.collect()
        print("Model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def generate_response(user_input):
    global model, tokenizer
    try:
        prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("<|assistant|>")[-1].strip() if "<|assistant|>" in response else response.strip()
    except Exception as e:
        print(f"Generation Error: {e}")
        return "ขออภัย เกิดข้อผิดพลาดในการประมวลผล"

# --- UI HTML ---
CHAT_HTML = """
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <title>🦄 TinyLLaMA Chatbot | YaDa</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { margin: 0; font-family: 'Segoe UI', sans-serif; background: #000; color: white; display: flex; justify-content: center; align-items: center; height: 100vh; }
        .chat-container { width: 100%; max-width: 750px; height: 90vh; background: #111; border-radius: 15px; display: flex; flex-direction: column; overflow: hidden; box-shadow: 0 0 30px rgba(255,0,255,0.3); }
        .chat-header { padding: 20px; background: linear-gradient(90deg, #ff0080, #7928ca, #2afadf); font-size: 1.5em; text-align: center; font-weight: bold; }
        .chat-box { flex: 1; padding: 15px; overflow-y: auto; background: #1a1a1a; display: flex; flex-direction: column; }
        .user-message { background: #0072ff; color: white; padding: 10px 15px; border-radius: 15px; align-self: flex-end; margin: 5px; max-width: 80%; }
        .bot-message { background: #333; color: white; padding: 10px 15px; border-radius: 15px; align-self: flex-start; margin: 5px; max-width: 80%; }
        .input-area { display: flex; padding: 20px; background: #111; }
        input { flex: 1; padding: 12px; border-radius: 30px; border: none; background: #222; color: white; outline: none; }
        button { background: #7928ca; border: none; padding: 10px 20px; margin-left: 10px; border-radius: 30px; color: white; cursor: pointer; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">💬 YaDa AI Chatbot</div>
        <div class="chat-box" id="chatBox">
            <div class="bot-message">สวัสดี! ถามอะไรฉันก็ได้เลย 😊</div>
        </div>
        <div class="input-area">
            <input type="text" id="userInput" placeholder="พิมพ์ข้อความ..." onkeypress="if(event.key==='Enter') sendMessage()">
            <button onclick="sendMessage()">ส่ง</button>
        </div>
    </div>
    <script>
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const text = input.value.trim();
            if(!text) return;
            const chatBox = document.getElementById('chatBox');
            chatBox.innerHTML += `<div class="user-message">${text}</div>`;
            input.value = '';
            chatBox.scrollTop = chatBox.scrollHeight;
            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ input: text })
                });
                const data = await res.json();
                chatBox.innerHTML += `<div class="bot-message">${data.response}</div>`;
                chatBox.scrollTop = chatBox.scrollHeight;
            } catch {
                chatBox.innerHTML += `<div class="bot-message">เกิดข้อผิดพลาด</div>`;
            }
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
            return jsonify({'response': 'Model not ready'})
    data = request.json
    response = generate_response(data.get('input', ''))
    return jsonify({'response': response})

if __name__ == '__main__':
    PORT = 5000
    setup_colab_tunnel(PORT)
    app.run(port=PORT)
