from sentence_transformers import SentenceTransformer, util

# Load this once globally
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def evaluate_tone(model_output, reference_response):
    embedding1 = model.encode(model_output, convert_to_tensor=True)
    embedding2 = model.encode(reference_response, convert_to_tensor=True)
    cosine_score = util.cos_sim(embedding1, embedding2).item()
    return round(cosine_score, 3)
