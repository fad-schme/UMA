# UMA-RLM Example Chatbot

Simple interactive chatbot demonstrating UMA-RLM memory usage.

Usage:

1. Ensure `config/uma.yaml` is configured for your LLMs/embedder.
2. (Optional) install extra dependency for PDF support: `pip install PyPDF2`
3. Run:

```bash
python examples/chatbot_app/main.py --config config/uma.yaml --user user:local
```

Commands inside the REPL:
- `/load` — load documents from `/material` into UMA (document ingestion pipeline).
- `/setprompt` — set the system prompt used for the assistant.
- `/quit` — exit.
