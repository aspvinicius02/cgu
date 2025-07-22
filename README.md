# Document Processing API

Uma API completa para processamento de documentos com capacidade de upload de PDFs, geração de embeddings, busca vetorial e classificação de texto.

## 📌 Visão Geral

Esta API oferece três funcionalidades principais:

- **📄 Processamento de Documentos**: Upload de PDFs, chunking, geração de embeddings e armazenamento vetorial
- **🔍 Naive RAG**: Sistema de recuperação aumentada por geração baseado nos documentos processados
- **🍿️ Classificação de Texto**: Análise de sentimentos e classificação textual com suporte a logprobs

## 🚀 Começando Rápido

### Pré-requisitos

- Python 3.8+
- Pipenv (recomendado) ou pip
- 4GB+ de RAM (8GB recomendado para modelos maiores)

### Instalação em 3 Passos

1. Clone o repositório:

```bash
git clone https://github.com/aspvinicius02/cgu.git
cd cgu
```

2. Instale as dependências:

```bash
pipenv install
```

3. Inicie o ambiente virtual e rode a API:

```bash
pipenv shell
uvicorn main_endpoint:app --reload
```

A API estará disponível em: [http://localhost:8000](http://localhost:8000)

## ⚙️ Configuração

Crie um arquivo `.env` na raiz (baseado no `chaves.env` em caso de usar chave da API da OPENAI 'OPENAI_API_KEY' ):

```env
# Modelos
EMBEDDING_MODEL=all-MiniLM-L6-v2
CLASSIFICATION_MODEL=distilbert-base-uncased-finetuned-sst-2-english

# Configurações de Chunking
DEFAULT_CHUNK_SIZE=500
DEFAULT_CHUNK_OVERLAP=50

# FAISS
FAISS_INDEX_PATH=./data/faiss_index
```

## 📚 Documentação Interativa

Acessos automaticamente gerados:

| Ferramenta   | URL                          |
|--------------|-------------------------------|
| Swagger UI   | http://localhost:8000/docs   |
| ReDoc        | http://localhost:8000/redoc  |

## 🔍 Endpoints Principais

### 1. Upload e Processamento de Documentos

**POST /upload/**

**Descrição**: Processa múltiplos PDFs, dividindo em chunks e gerando embeddings vetoriais.

**Parâmetros**:

- `files` (obrigatório): Lista de arquivos PDF
- `chunk_size`: Tamanho em tokens (padrão: 500)
- `chunk_overlap`: Sobreposição entre chunks (padrão: 50)

**Exemplo**:

```bash
curl -X POST "http://localhost:8000/upload/" \
  -H "accept: application/json" \
  -F "files=@relatorio.pdf" \
  -F "chunk_size=600"
```

**Resposta**:

```json
{
  "status": "success",
  "documents_processed": 2,
  "total_chunks": 45,
  "processing_time": 12.3
}
```

### 2. Consulta RAG

**POST /query/**

**Descrição**: Busca semântica nos documentos processados.

**Corpo (JSON)**:

```json
{
  "question": "Quais são os principais riscos mencionados?",
  "use_bm25": true,
  "top_k": 5
}
```

**Resposta**:

```json
{
  "results": [
    {
      "text": "Os principais riscos incluem...",
      "score": 0.87,
      "source": "relatorio.pdf (página 12)"
    }
  ]
}
```

### 3. Classificação de Texto

**POST /classify/**

**Descrição**: Classifica texto e retorna probabilidades.

**Exemplo**:

```bash
curl -X POST "http://localhost:8000/classify/" \
  -d '{"text":"O produto atendeu às expectativas","return_logprobs":true}'
```

**Resposta**:

```json
{
  "classification": "POSITIVE",
  "confidence": 0.92,
  "logprobs": {
    "POSITIVE": -0.08,
    "NEGATIVE": -2.41
  }
}
```

## 🏗 Diagrama simplificado da arquitetura utilizada

┌─────────┐
│  Client │
└────┬────┘
     │
     ▼
┌──────────┐
│  FastAPI │
└────┬─────┘
     │
 ┌───┼────────────────────┬─────────────────────┐
 │   │                    │                     │
 ▼   ▼                    ▼                     ▼
PDF Processing   Embedding Generation   Text Classification
 │   │                    │                     │
 │   ▼                    ▼                     ▼
 │  FAISS Vector DB    FAISS Vector DB     PostgreSQL (Opcional)
 │      │                  │
 │      └──────┬───────────┘
 │             ▼
 │       Naive RAG Service
 │             │
 ▼             ▼
Resposta RAG   Resultados Classificação


## 🛠️ Stack Tecnológica

| Categoria   | Tecnologias                                   |
|-------------|-----------------------------------------------|
| Framework   | FastAPI, Uvicorn                              |
| NLP         | Sentence-Transformers, HuggingFace            |
| Vector DB   | FAISS                                         |
| PDF         | PyPDF2, pdf2image                             |
| Utilitários | Pydantic, Loguru                              |

## 📂 Estrutura do Projeto

```text
.
├── document-processing-api/
    ├── chaves.env            # Exemplo de variáveis de ambiente
    ├── main_endpoint.py      # Código principal da API
    ├── requirements.txt      # Dependências do projeto
    ├── README.md             # Este arquivo
    ├── CV 1-5.pdf            # documento de extensão .pdf utilizado no upload
    └── DAM.pdf               # documento de extensão .pdf utilizado no upload   
    
```

## 🤝 Como Contribuir

- Reporte issues com detalhes
- Faça fork e crie sua branch (`git checkout -b feature/nova-feature`)
- Commit changes (`git commit -am 'Adiciona nova feature'`)
- Push (`git push origin feature/nova-feature`)
- Abra um Pull Request
