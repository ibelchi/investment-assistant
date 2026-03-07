import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from src.ai.rag_engine import RAGEngine

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self):
        # S'assumeix OPENAI_API_KEY a l'entorn
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2)
        self.rag = RAGEngine()
        
    def generate_report(self, symbol: str, strategy_name: str, tech_reason: str, current_price: float) -> str:
        """Agrupa el coneixement RAG amb la troballa tècnica per oferir un resum."""
        
        # Consultem a la bdd de coneixement del RAG quins criteris té l'usuari
        # o consells generals sobre swing trading i l'estratègia en concret.
        user_knowledge = self.rag.similarity_search(f"Criterios de inversión y factores de riesgo para {strategy_name} en accions", k=3)
        
        prompt = PromptTemplate.from_template("""
        Ets un Assistent d'Anàlisi d'Inversions Expert.
        
        T'acabo d'enviar una oportunitat de Swing Trading detectada pel meu sistema automàtic:
        - Acció: {symbol}
        - Estratègia usada pel sistema: {strategy_name}
        - Preu actual: ${current_price}
        - Lògica tècnica de detecció: {tech_reason}
        
        Coneixement teòric proveït per l'usuari recuperat de la seva base de dades (RAG Context): 
        {user_knowledge}
        
        TASCA: Escriu un petit informe dirigit a mi (l'inversor) en format Markdown.
        1. Resum curtet del motiu de la detecció donat pel sistema tècnic.
        2. Relaciona i contrasta amb el coneixement clau procedent dels meus documents si n'hi ha.
        3. Fes una anàlisi de probables factors positius (catalitzadors) i negatius (riscos d'entrar malament).
        4. Recomanació final d'observació o precaucions base de purament anàlisi i en l'estratègia descrita sense donar consells concloents.
        
        Escriu en català i de manera molt professional.
        """)
        
        formatted_prompt = prompt.format(
            symbol=symbol,
            strategy_name=strategy_name,
            current_price=current_price,
            tech_reason=tech_reason,
            user_knowledge=user_knowledge
        )
        
        try:
            response = self.llm.invoke(formatted_prompt)
            return response.content
        except Exception as e:
            logger.error(f"Error a l'invocar LLM per reports: {e}")
            return f"Error des de l'IA per a generar l'informe ({e}). Considereu configurar la API Key d'OpenAI."
