# Updated version using TinyLLaMA model and general-purpose chatbot UI with enhanced response handling

from flask import Flask, request, jsonify, render_template_string
import torch
import os
import gc
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from google.colab import output  # สำหรับ Colab เท่านั้น

app = Flask(__name__)

MODEL_PATH = '/content/drive/MyDrive/fine_tuned_tinyllama.pth'

model = None
tokenizer = None
last_activity_time = time.time()

def setup_colab_tunnel(port):
    try:
        output.serve_kernel_port_as_window(port)
        print(f"Colab tunnel created for port {port}.")
    except Exception as e:
        print(f"ไม่สามารถเปิด tunnel ได้: {e}")

def ensure_folders():
    if not os.path.exists("offload"):
        os.makedirs("offload")
        print("สร้างโฟลเดอร์ offload สำเร็จ")

def load_model():
    print("Loading TinyLLaMA model...")
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

        if os.path.exists(MODEL_PATH):
            print(f"Loading fine-tuned weights from {MODEL_PATH}")
            state_dict = torch.load(MODEL_PATH, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            del state_dict
            torch.cuda.empty_cache()
            gc.collect()
        else:
            print("ใช้ base model เพราะไม่พบไฟล์ fine-tuned")

        model.eval()
        print("Model loaded successfully")
        return True

    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def check_idle_time():
    global last_activity_time
    return time.time() - last_activity_time

def update_activity_time():
    global last_activity_time
    last_activity_time = time.time()

def ensure_resources_loaded():
    global model, tokenizer
    if model is None or tokenizer is None:
        return load_model()
    return True

def generate_response(user_input):
    global model, tokenizer
    try:
        prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

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
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        else:
            response = response.strip()

        torch.cuda.empty_cache()
        gc.collect()

        return response if response else "ขออภัย ระบบไม่สามารถตอบกลับได้ในตอนนี้"

    except Exception as e:
        print(f"Error generating response: {e}")
        return "ขออภัย ระบบประมวลผลไม่สำเร็จ กรุณาลองใหม่อีกครั้ง"

def unload_model_if_idle(idle_time_threshold=180):
    global model, tokenizer
    idle_time = check_idle_time()
    if idle_time > idle_time_threshold and model is not None:
        print(f"Unloading model after {idle_time:.1f} seconds of inactivity...")
        model = None
        tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        print("Model unloaded")

CHAT_HTML = """
<!DOCTYPE html>
<html lang=\"th\">
<head>
    <meta charset=\"UTF-8\">
    <title>🦄 TinyLLaMA Chatbot | YaDa</title>
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <style>
        body {
            margin: 0;
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #000, #222);
            color: white;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .chat-container {
            width: 100%;
            max-width: 750px;
            height: 90vh;
            background: #111;
            border-radius: 15px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 0 30px rgba(255,0,255,0.3);
        }
        .chat-header {
            padding: 20px;
            background: linear-gradient(90deg, #ff0080, #7928ca, #2afadf);
            font-size: 1.5em;
            text-align: center;
            font-weight: bold;
            color: white;
        }
        .chat-box {
            flex: 1;
            padding: 15px;
            overflow-y: auto;
            background: #1a1a1a;
        }
        .user-message, .bot-message {
            margin: 10px 0;
            padding: 12px 18px;
            border-radius: 20px;
            max-width: 80%;
            line-height: 1.5;
        }
        .user-message {
            background: linear-gradient(145deg, #00c6ff, #0072ff);
            color: white;
            margin-left: auto;
        }
        .bot-message {
            background: #333;
            color: #eee;
            margin-right: auto;
        }
        .input-area {
            display: flex;
            padding: 20px;
            background: #111;
            border-top: 1px solid #444;
        }
        .input-area input {
            flex: 1;
            padding: 12px 20px;
            border: none;
            border-radius: 30px;
            font-size: 1em;
            background: #1e1e1e;
            color: white;
        }
        .input-area button {
            background: linear-gradient(145deg, #ff0080, #7928ca, #2afadf);
            border: none;
            padding: 12px 20px;
            margin-left: 10px;
            border-radius: 30px;
            color: white;
            font-weight: bold;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class=\"chat-container\">
        <div class=\"chat-header\">💬 YaDa - AI ผู้ช่วยอัจฉริยะ</div>
        <div class=\"chat-box\" id=\"chatBox\">
            <div class=\"bot-message\">สวัสดี! ถามอะไรกับฉันก็ได้เลย 😊</div>
        </div>
        <div class=\"input-area\">
            <input type=\"text\" id=\"userInput\" placeholder=\"พิมพ์ข้อความของคุณ...\" onkeypress=\"if(event.key==='Enter') sendMessage()\">
            <button onclick=\"sendMessage()\">ส่ง</button>
        </div>
    </div>
    <script>
        async function sendMessage() {
            const inputBox = document.getElementById('userInput');
            const userInput = inputBox.value.trim();
            if (!userInput) return;
            const chatBox = document.getElementById('chatBox');
            chatBox.innerHTML += `<div class='user-message'>${userInput}</div>`;
            inputBox.value = '';
            chatBox.scrollTop = chatBox.scrollHeight;
            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ input: userInput })
                });
                const data = await res.json();
                chatBox.innerHTML += `<div class='bot-message'>${data.response}</div>`;
                chatBox.scrollTop = chatBox.scrollHeight;
            } catch (err) {
                chatBox.innerHTML += `<div class='bot-message'>เกิดข้อผิดพลาด กรุณาลองใหม่</div>`;
            }
        }
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(CHAT_HTML)

@app.route('/api/chat', methods=['POST'])
def chat():
    update_activity_time()
    if not ensure_resources_loaded():
        return jsonify({ 'status': 'error', 'response': 'ไม่สามารถโหลดโมเดลได้' })
    try:
        data = request.json
        user_input = data.get('input', '')[:500]
        if not user_input:
            return jsonify({'status': 'error', 'response': 'กรุณาใส่ข้อความ'}), 400
        response = generate_response(user_input)
        return jsonify({ 'status': 'success', 'response': response })
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({ 'status': 'error', 'response': 'เกิดข้อผิดพลาด' })

@app.route('/api/unload', methods=['POST'])
def unload():
    unload_model_if_idle()
    return jsonify({'status': 'success'})

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'online',
        'model_loaded': model is not None,
        'idle_time': check_idle_time()
    })

if __name__ == '__main__':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:32'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    port = 5000
    setup_colab_tunnel(port)
    app.run()
