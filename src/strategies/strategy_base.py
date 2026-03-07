from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class StrategyBase(ABC):
    """
    Classe base per a totes les estratègies d'inversió (plugins).
    Garanteix un contracte comú perquè el Market Scanner pugui fer
    servir múltiples estratègies de format intercanviable.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Nom identificador de l'estratègia."""
        pass
        
    @property
    @abstractmethod
    def default_parameters(self) -> Dict[str, Any]:
        """Paràmetres per defecte si no s'han configurat prèviament o a la DB."""
        pass

    @abstractmethod
    def analyze(self, symbol: str, hist_data, info_data, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analitza les dades d'una acció i determina si hi ha una oportunitat o no.
        
        Args:
            symbol (str): Ticker de l'acció
            hist_data (pd.DataFrame): Dades històriques EOD (End of Day)
            info_data (dict): Dades fonamentals com capitalització, sector
            config (dict): Configuració (paràmetres) a utilitzar (sobreescriu per defecte)
            
        Returns:
            Dict amb el resultat:
            {
                "is_opportunity": bool,
                "confidence": float,
                "current_price": float,
                "reason": str (explicació tècnica per als reports)
            }
        """
        pass
