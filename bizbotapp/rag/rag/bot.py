import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from rag.prompt_template import format_prompt

CHROMA_DB_PATH = os.path.expanduser("~/bb/bizbotapp/scripts/vector_store/chroma_db")
BASE_MODEL_PATH = os.path.expanduser("~/bb/bizbotapp/scripts/models/base/Llama-3.2-1B-Instruct")
ADAPTER_PATH = os.path.expanduser("~/bb/bizbotapp/models/adapters/llamaft")

class BizBot:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        self.model.eval()

        client = PersistentClient(path=CHROMA_DB_PATH)
        self.collection = client.get_or_create_collection("company_docs")

    def answer(self, query: str, n_results: int = 3) -> str:
        print("🔍 Embedding query and retrieving documents...")
        results = self.collection.query(query_texts=[query], n_results=n_results)
        context = "\n".join(results["documents"][0]) if results["documents"] and results["documents"][0] else ""

        prompt = format_prompt(query, context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return answer.split("Answer:")[-1].strip()