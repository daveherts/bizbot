# rag/bot.py

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from rag.prompt_template import format_prompt

CHROMA_DB_PATH = os.path.expanduser("~/bb/bizbotapp/rag/vector_store/chroma_db")
BASE_MODEL_PATH = os.path.expanduser("~/bb/bizbotapp/models/base/Llama-3.2-1B-Instruct")
ADAPTER_PATH = os.path.expanduser("~/bb/bizbotapp/models/adapters/llamaft")
QNA_FILE_PATH = os.path.expanduser("~/bb/bizbotapp/rag/cleaned_faq_pairs.txt")

class BizBot:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("🧠 Loading fine-tuned model (LoRA)...")

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        self.model.to(self.device)
        self.model.eval()

        print("✅ Model loaded on:", self.device)

        client = PersistentClient(path=CHROMA_DB_PATH)
        self.collection = client.get_or_create_collection("company_docs")

        self.qna_pairs = self.load_qna_pairs(QNA_FILE_PATH)
        print(f"📚 Loaded {len(self.qna_pairs)} Q&A pairs from file.")

    def load_qna_pairs(self, path):
        if not os.path.exists(path):
            return []
        pairs = []
        with open(path, "r", encoding="utf-8") as f:
            current_q, current_a = None, None
            for line in f:
                line = line.strip()
                if line.startswith("Q:"):
                    current_q = line[2:].strip()
                elif line.startswith("A:"):
                    current_a = line[2:].strip()
                    if current_q and current_a:
                        pairs.append((current_q.lower(), current_a))
                        current_q, current_a = None, None
        return pairs

    def check_exact_match(self, query):
        query_lower = query.strip().lower()
        for q, a in self.qna_pairs:
            if q == query_lower:
                print("✅ Exact match found in Q&A file.")
                return a
        return None

    def answer(self, query: str) -> str:
        # Step 1: Exact Q&A match
        match = self.check_exact_match(query)
        if match:
            return match

        # Step 2: RAG-based generation
        print("🔍 Embedding query and retrieving documents...")
        results = self.collection.query(query_texts=[query], n_results=3)
        context = "\n".join(results["documents"][0]) if results["documents"] else ""
        print("📄 Retrieved context:\n", context)

        prompt = format_prompt(query, context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.0,
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.split("Answer:")[-1].strip()
