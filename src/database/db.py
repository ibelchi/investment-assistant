import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# Definir la ruta base del projecte (un directori per sobre d'aquest fitxer)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, 'data', 'investment_assistant.db')

# Crear el directori data si no existeix
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

engine = create_engine(f'sqlite:///{DB_PATH}', echo=False)
Base = declarative_base()

class Opportunity(Base):
    __tablename__ = 'opportunities'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    date_detected = Column(DateTime, default=datetime.utcnow)
    strategy_name = Column(String(50), nullable=False)
    current_price = Column(Float, nullable=False)
    target_price = Column(Float, nullable=True) # Fase 2
    stop_loss = Column(Float, nullable=True)    # Fase 2
    strategy_config = Column(JSON, nullable=True) # Guardar la configuració usada
    market_context = Column(Text, nullable=True)  # Generat per IA
    explanation = Column(Text, nullable=True)     # Generat per IA

class StrategyConfig(Base):
    __tablename__ = 'strategy_configs'
    
    id = Column(Integer, primary_key=True)
    strategy_name = Column(String(50), unique=True, nullable=False)
    parameters = Column(JSON, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Crear o actualitzar l'esquema
Base.metadata.create_all(engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Generador per obtenir la sessió de la base de dades"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
