import os
import logging
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VECTOR_DB_PATH = os.path.join(BASE_DIR, 'data', 'vector_store')

class RAGEngine:
    def __init__(self):
        # Es pot canviar a embeddings locals (e.g. HuggingFace) si l'usuari ho prefereix
        # Requereix base d'entorn OPENAI_API_KEY
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        
        # Carregar vector store existent si ni ha
        if os.path.exists(VECTOR_DB_PATH):
            try:
                self.vector_store = FAISS.load_local(VECTOR_DB_PATH, self.embeddings, allow_dangerous_deserialization=True)
                logger.info("S'ha carregat el Vector Store existent")
            except Exception as e:
                logger.error(f"Error carregant FAISS: {e}")
                
    def process_pdf(self, pdf_path: str) -> bool:
        """Llegeix un PDF, el divideix, crea els embeddings i els guarda a FAISS."""
        try:
            logger.info(f"Processant el document {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Dividir en chunks perquè la IA pugui buscar per rellevància i encabir en context
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(splits, self.embeddings)
            else:
                self.vector_store.add_documents(splits)
                
            # Persistir
            self.vector_store.save_local(VECTOR_DB_PATH)
            return True
        except Exception as e:
            logger.error(f"Error processant el document PDF: {e}")
            return False
            
    def similarity_search(self, query: str, k: int = 3) -> str:
        """Busca informació rellevant a la documentació (p.ex. criteris d'inversió)."""
        if self.vector_store is None:
            return "No s'ha configurat cap document de coneixement previ."
            
        docs = self.vector_store.similarity_search(query, k=k)
        text_context = "\n\n".join([d.page_content for d in docs])
        return text_context
