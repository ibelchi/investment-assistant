import logging
import pandas as pd
from typing import Optional, List, Dict
from datetime import timedelta

from src.data.ingestion import get_historical_data, get_company_info
from src.strategies.buy_the_dip import BuyTheDipStrategy

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self):
        self.strategy = BuyTheDipStrategy()
        
    def run_backtest(self, symbols: List[str], start_date: str, end_date: str, custom_config: Optional[Dict] = None) -> List[Dict]:
        """
        Emula el pas del temps tallant els DataFrames històrics dia a dia per 
        veure quan la nostra estratègia hauria donat senyal.
        """
        logger.info(f"Iniciant Backtest de {start_date} a {end_date} per {len(symbols)} actius.")
        config = custom_config or self.strategy.default_parameters
        results = []
        
        for sym in symbols:
            # Descarreguem un any extra de dades abans del start_date per assegurar tenir
            # prou info per les mitges mòbils i "lookback_days".
            hist_start = pd.to_datetime(start_date) - timedelta(days=365)
            # Fem servir yfinance 'history' descarregant tot el necessari, ací optem
            # por simplificar baixant el màxim i talant.
            hist_data = get_historical_data(sym, period="2y")
            info_data = get_company_info(sym)
            
            if hist_data.empty:
                continue
                
            # Filtrem entre start_date i end_date per simular
            mask = (hist_data.index >= pd.to_datetime(start_date, utc=True)) & (hist_data.index <= pd.to_datetime(end_date, utc=True))
            sim_indexes = hist_data[mask].index
            
            for sim_date in sim_indexes:
                # El "passat" al punt d'avaluació
                past_data = hist_data[hist_data.index <= sim_date]
                
                # Intentem analitzar com si fóssim a eixe dia
                res = self.strategy.analyze(sym, past_data, info_data, config)
                
                if res.get("is_opportunity"):
                    # Compra "virtual"
                    entry_price = res["current_price"]
                    
                    # Verifiquem si hauria sigut un éxit o fracàs (ex: buscar rendimient a +30 dies)
                    future_data = hist_data[hist_data.index > sim_date].head(20) # Mirem les seguents 20 sessions (aprox 1 mes)
                    if not future_data.empty:
                        max_future_price = future_data["High"].max()
                        profit_pct = ((max_future_price - entry_price) / entry_price) * 100
                    else:
                        profit_pct = 0.0
                        
                    results.append({
                        "sim_date": sim_date.strftime('%Y-%m-%d'),
                        "symbol": sym,
                        "entry_price": entry_price,
                        "confidence": res.get("confidence"),
                        "max_1mo_profit_pct": round(profit_pct, 2)
                    })
                    
        return results
