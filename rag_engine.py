"""
RAG Engine - Core retrieval-augmented generation logic
Uses FAISS for vector search + HuggingFace for embeddings
"""

import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
)
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    content: str
    source: str
    page: int
    score: float
    chunk_id: int


@dataclass
class RAGResponse:
    answer: str
    sources: List[RetrievedChunk]
    query: str
    context_used: str
    model_used: str


class DocumentIngestionPipeline:
    """Handles document loading, splitting, and preprocessing"""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_document(self, file_path: str) -> List[Document]:
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}. Supported: {self.SUPPORTED_EXTENSIONS}")

        loaders = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader
        }

        loader_cls = loaders[ext]
        loader = loader_cls(file_path)
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} pages from {path.name}")
        return docs

    def ingest_documents(self, file_paths: List[str]) -> List[Document]:
        all_docs = []
        for fp in file_paths:
            try:
                docs = self.load_document(fp)
                chunks = self.splitter.split_documents(docs)
                # Attach metadata
                for i, chunk in enumerate(chunks):
                    chunk.metadata["chunk_id"] = i
                    chunk.metadata["source_file"] = Path(fp).name
                all_docs.extend(chunks)
                logger.info(f"Split {Path(fp).name} into {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Failed to ingest {fp}: {e}")

        logger.info(f"Total chunks across all documents: {len(all_docs)}")
        return all_docs


class EmbeddingEngine:
    """Generates dense embeddings using HuggingFace sentence-transformers"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dim: {self.embedding_dim}")

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )
        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype("float32")


class VectorStore:
    """FAISS-based vector store with metadata support"""

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        # Inner product on normalized vectors == cosine similarity
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.documents: List[Document] = []
        self.is_built = False

    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        assert len(documents) == len(embeddings), "Mismatch between docs and embeddings"
        self.index.add(embeddings)
        self.documents.extend(documents)
        self.is_built = True
        logger.info(f"Vector store now holds {self.index.ntotal} vectors")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[Document, float]]:
        if not self.is_built:
            raise RuntimeError("Vector store is empty. Ingest documents first.")

        scores, indices = self.index.search(query_vector, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append((self.documents[idx], float(score)))
        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        logger.info(f"Vector store saved to {path}")

    def load(self, path: str):
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)
        self.is_built = True
        logger.info(f"Vector store loaded: {self.index.ntotal} vectors")


class LLMInference:
    """
    LLM wrapper supporting local HuggingFace models and OpenAI-compatible APIs.
    Falls back to a mock for demo/testing without GPU.
    """

    def __init__(self, model_name: str = "mock", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._pipeline = None

        if model_name == "mock":
            logger.info("Using mock LLM (no API key / GPU required)")
        elif model_name.startswith("gpt"):
            self._setup_openai()
        else:
            self._setup_hf_pipeline(model_name)

    def _setup_openai(self):
        try:
            import openai
            self._client = openai.OpenAI(api_key=self.api_key)
            logger.info(f"OpenAI client ready: {self.model_name}")
        except ImportError:
            logger.warning("openai package not installed; falling back to mock")
            self.model_name = "mock"

    def _setup_hf_pipeline(self, model_name: str):
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._pipeline = pipeline(
                "text2text-generation",   # flan-t5 needs this, not "text-generation"
                model=model_name,
                tokenizer=tokenizer,
                max_new_tokens=256,
                do_sample=False
            )
            logger.info(f"HuggingFace pipeline ready: {model_name}")
        except Exception as e:
            logger.warning(f"HF pipeline failed ({e}); falling back to mock")
            self.model_name = "mock"

    def generate(self, prompt: str) -> str:
        if self.model_name == "mock":
            return self._mock_generate(prompt)
        elif self.model_name.startswith("gpt"):
            return self._openai_generate(prompt)
        else:
            return self._hf_generate(prompt)

    def _mock_generate(self, prompt: str) -> str:
        # Extracts answer from context to simulate RAG
        context_start = prompt.find("CONTEXT:") + 8
        context_end = prompt.find("QUESTION:")
        if context_start > 8 and context_end > 0:
            context = prompt[context_start:context_end].strip()
            lines = [l.strip() for l in context.split("\n") if l.strip()]
            if lines:
                summary = lines[0][:300]
                return (
                    f"Based on the provided documents: {summary}... "
                    f"[This is a demo response. Connect a real LLM via OPENAI_API_KEY or "
                    f"set LLM_MODEL to a HuggingFace model name for actual generation.]"
                )
        return "I could not find relevant information in the provided context."

    def _openai_generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer based only on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=512
        )
        return response.choices[0].message.content

    def _hf_generate(self, prompt: str) -> str:
        result = self._pipeline(prompt)
        # text2text-generation returns the answer directly (no prompt echo)
        generated = result[0]["generated_text"]
        import re
        # Collapse newlines, bullet artifacts, excess whitespace
        cleaned = re.sub(r'\s*\n\s*', ' ', generated)
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)
        cleaned = re.sub(r'\s●\s', '. ', cleaned)  # fix bullet points
        return cleaned.strip()


RAG_PROMPT_TEMPLATE = """You are an expert AI assistant. Answer the user's question using ONLY the information from the provided context. 
If the context does not contain enough information, say so clearly.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer concisely and accurately based on the context above.
- Cite which document sections support your answer.
- If the answer isn't in the context, respond: "The provided documents don't contain sufficient information to answer this question."

ANSWER:"""


class RAGPipeline:
    """Full end-to-end RAG pipeline"""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "mock",
        vector_store_path: str = "./vector_store",
        top_k: int = 5,
        score_threshold: float = 0.3,
    ):
        self.embedding_engine = EmbeddingEngine(embedding_model)
        self.vector_store = VectorStore(self.embedding_engine.embedding_dim)
        self.llm = LLMInference(llm_model)
        self.ingestion = DocumentIngestionPipeline()
        self.vector_store_path = vector_store_path
        self.top_k = top_k
        self.score_threshold = score_threshold

        # Load existing store if available
        if os.path.exists(os.path.join(vector_store_path, "index.faiss")):
            self.vector_store.load(vector_store_path)

    def ingest(self, file_paths: List[str]) -> Dict:
        docs = self.ingestion.ingest_documents(file_paths)
        if not docs:
            return {"status": "error", "message": "No documents ingested"}

        texts = [d.page_content for d in docs]
        embeddings = self.embedding_engine.embed_texts(texts)
        self.vector_store.add_documents(docs, embeddings)
        self.vector_store.save(self.vector_store_path)

        return {
            "status": "success",
            "documents_ingested": len(file_paths),
            "chunks_created": len(docs),
            "total_vectors": self.vector_store.index.ntotal
        }

    def query(self, question: str) -> RAGResponse:
        if not self.vector_store.is_built:
            return RAGResponse(
                answer="No documents have been ingested yet. Please upload documents first.",
                sources=[],
                query=question,
                context_used="",
                model_used=self.llm.model_name
            )

        # Embed the query
        query_vec = self.embedding_engine.embed_query(question)

        # Retrieve top-k chunks
        results = self.vector_store.search(query_vec, top_k=self.top_k)

        # Filter by score threshold
        filtered = [(doc, score) for doc, score in results if score >= self.score_threshold]

        if not filtered:
            filtered = results[:2]  # fallback: take top 2 anyway

        # Build context
        context_parts = []
        sources = []
        for i, (doc, score) in enumerate(filtered):
            meta = doc.metadata
            source_name = meta.get("source_file", meta.get("source", "unknown"))
            page = meta.get("page", 0)
            context_parts.append(f"[Source {i+1}: {source_name}, page {page}]\n{doc.page_content}")
            sources.append(RetrievedChunk(
                content=doc.page_content,
                source=source_name,
                page=page,
                score=score,
                chunk_id=meta.get("chunk_id", i)
            ))

        context = "\n\n---\n\n".join(context_parts)

        # Generate answer
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
        answer = self.llm.generate(prompt)

        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            context_used=context,
            model_used=self.llm.model_name
        )

    def get_stats(self) -> Dict:
        return {
            "total_vectors": self.vector_store.index.ntotal if self.vector_store.is_built else 0,
            "embedding_model": self.embedding_engine.model_name,
            "embedding_dim": self.embedding_engine.embedding_dim,
            "llm_model": self.llm.model_name,
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
        }
