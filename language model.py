
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import gradio as gr

# Load and preprocess data
def load_data(file_path):
    df = pd.read_excel(file_path)
    df = df[['PMID', 'Title', 'Abstract']].dropna()
    df['text'] = "Title: " + df['Title'] + "\nAbstract: " + df['Abstract']
    return df['text'].tolist()

# Initialize models and vector DB
class GliomaAssistant:
    def __init__(self, data_path):
        # Load data
        self.documents = load_data(data_path)
        
        # Initialize models
        self.embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl")
        
        # Create vector database
        self._create_vector_db()
    
    def _create_vector_db(self):
        # Generate embeddings
        embeddings = self.embedder.encode(self.documents, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings).astype('float32'))
    
    def _retrieve_docs(self, query, k=5):
        query_embedding = self.embedder.encode([query])
        distances, indices = self.index.search(query_embedding, k)
        return [self.documents[i] for i in indices[0]]
    
    def generate_answer(self, query):
        # Retrieve relevant documents
        context = self._retrieve_docs(query)
        
        # Create prompt
        prompt = f"""You are a Glioma specialist focusing on AGPS protein. Answer the question using only the context below.
        
        Context:
        {'\n\n'.join(context)}
        
        Question: {query}
        Answer:"""
        
        # Generate answer
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=4096, truncation=True)
        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            max_length=512,
            temperature=0.7,
            num_beams=5,
            early_stopping=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Initialize system
assistant = GliomaAssistant("abstracts.xlsx")

# Create Gradio interface
def respond(query, history):
    response = assistant.generate_answer(query)
    return response

interface = gr.ChatInterface(
    fn=respond,
    title="Glioma Research Assistant",
    description="Ask any question about glioma and AGPS protein",
    examples=[
        "Why is AGPS a potential target for glioma therapy?",
        "What are the latest biomarkers for glioma?",
        "Explain the role of AGPS in glioma progression",
        "What animal models are used in glioma research?"
    ]
)

# Run the application
if __name__ == "__main__":
    interface.launch()