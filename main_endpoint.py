import os
import uuid
from typing import List, Optional, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

app = FastAPI(title="Document Processing API")

# Configurações iniciais
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_DIM = 384  # Dimensão do modelo all-MiniLM-L6-v2
CLASSIFICATION_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# Inicialização de modelos
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
classifier_tokenizer = AutoTokenizer.from_pretrained(CLASSIFICATION_MODEL)
classifier_model = AutoModelForSequenceClassification.from_pretrained(CLASSIFICATION_MODEL)

# Estruturas de dados para armazenamento
vector_index = faiss.IndexFlatL2(VECTOR_DIM)
document_store = []  # Armazena metadados e chunks
bm25_corpus = []  # Para reranking BM25

class ChunkingParams(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 50

class QueryInput(BaseModel):
    question: str
    use_bm25: bool = False
    top_k: int = 3

class ClassificationInput(BaseModel):
    text: str
    return_logprobs: bool = False

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text: str, params: ChunkingParams) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + params.chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - params.chunk_overlap
        
        if start < 0:  # Evitar índice negativo
            start = 0
            
    return chunks

@app.post("/upload/")
async def upload_files(
    files: List[UploadFile] = File(...),
    params: ChunkingParams = ChunkingParams()
):
    global vector_index, document_store, bm25_corpus
    
    try:
        for file in files:
            if file.content_type != "application/pdf":
                raise HTTPException(status_code=400, detail="Only PDF files are supported")
            
            # Extrair texto do PDF
            text = extract_text_from_pdf(file.file)
            
            # Chunknização
            chunks = chunk_text(text, params)
            
            # Gerar embeddings
            embeddings = embedding_model.encode(chunks, convert_to_tensor=False)
            
            # Armazenar no FAISS
            if len(document_store) == 0:
                vector_index.add(embeddings)
            else:
                # Se o índice já tem vetores, precisamos reconstruir com a nova dimensão
                current_vectors = vector_index.reconstruct_n(0, vector_index.ntotal)
                updated_vectors = np.vstack([current_vectors, embeddings])
                vector_index.reset()
                vector_index.add(updated_vectors)
            
            # Armazenar metadados
            doc_id = str(uuid.uuid4())
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                document_store.append({
                    "doc_id": doc_id,
                    "file_name": file.filename,
                    "chunk_id": i,
                    "chunk_text": chunk,
                    "embedding": embedding.tolist()
                })
            
            # Atualizar corpus para BM25
            bm25_corpus.extend([chunk.split() for chunk in chunks])
        
        return {"message": f"Processed {len(files)} files", "total_chunks": len(document_store)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/")
async def naive_rag(query: QueryInput):
    try:
        # Embedding da pergunta
        query_embedding = embedding_model.encode(query.question, convert_to_tensor=False)
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        # Busca por similaridade
        D, I = vector_index.search(query_embedding, query.top_k)
        
        # Recuperar chunks relevantes
        results = [document_store[i] for i in I[0]]
        
        # Aplicar BM25 se solicitado
        if query.use_bm25 and len(bm25_corpus) > 0:
            bm25 = BM25Okapi(bm25_corpus)
            tokenized_query = query.question.split()
            bm25_scores = bm25.get_scores(tokenized_query)
            
            # Combinar scores (simples média ponderada)
            for i, res in enumerate(results):
                original_idx = I[0][i]
                combined_score = (1 - D[0][i] + bm25_scores[original_idx]) / 2
                res["combined_score"] = combined_score
            
            # Reordenar por combined_score
            results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Construir contexto para resposta (simplificado)
        context = "\n\n".join([res["chunk_text"] for res in results])
        answer = f"Based on the documents, here's what I found:\n\nContext:\n{context}\n\n[This is a placeholder - in a real implementation you would generate a proper answer using an LLM]"
        
        return {
            "question": query.question,
            "answer": answer,
            "relevant_chunks": results,
            "used_bm25": query.use_bm25
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify/")
async def classify_text(input: ClassificationInput):
    try:
        inputs = classifier_tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = classifier_model(**inputs)
        
        probs = F.softmax(outputs.logits, dim=-1)
        predicted_class_idx = torch.argmax(probs).item()
        predicted_class = classifier_model.config.id2label[predicted_class_idx]
        
        response = {
            "text": input.text,
            "classification": predicted_class,
            "confidence": probs[0][predicted_class_idx].item()
        }
        
        if input.return_logprobs:
            logprobs = torch.log(probs)
            response["logprobs"] = {
                "positive": logprobs[0][1].item(),
                "negative": logprobs[0][0].item()
            }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)