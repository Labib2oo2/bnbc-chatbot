# BNBC AI Chatbot (100% Free)

## যা যা একদম ফ্রি লাগবে
- GitHub account (ফ্রি)
- Groq API key (console.groq.com — কোনো কার্ড লাগে না)
- Streamlit Community Cloud (ফ্রি hosting)

## Local run (টেস্ট করার জন্য)
```
pip install -r requirements.txt
export GROQ_API_KEY=your_key
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. এই ফোল্ডারের সব ফাইল (app.py, chunks.pkl, bm25.pkl, requirements.txt) GitHub repo-তে push করো
2. https://share.streamlit.io -তে গিয়ে GitHub দিয়ে sign in করো
3. "Create app" → repo সিলেক্ট করো → Main file: `app.py`
4. Deploy করার আগে "Advanced settings" → Secrets-এ লিখো:
   ```
   GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxx"
   ```
5. Deploy চাপো — কয়েক মিনিটে `xxxx.streamlit.app` লিংক পাবে

## Groq API Key কিভাবে পাবে (ফ্রি)
1. console.groq.com -এ যাও
2. Google/GitHub দিয়ে sign up করো
3. "API Keys" → Create → কপি করো

## Files
- `app.py` — Streamlit web app (Groq + Llama 3.3 70B)
- `chat.py` — Terminal (TUI) version
- `chunks.pkl`, `bm25.pkl` — BNBC 2020 indexed sections (4544 chunks, 2464 pages)
