import pandas as pd
from typing import Dict, Any, Optional
from .strategy_base import StrategyBase

class BuyTheDipStrategy(StrategyBase):
    
    @property
    def name(self) -> str:
        return "Buy the Dip (Swing)"
        
    @property
    def default_parameters(self) -> Dict[str, Any]:
        return {
            "min_drop_pct": 15.0,        # Caiguda mínima des del màxim
            "lookback_days": 60,         # Finestra de dies a observar el màxim
            "min_rebound_pct": 2.0,      # Han de pujar un mínim 2% des del fons per confirmar canvi tendència
            "min_market_cap_b": 10.0,    # Capitalització mínima en Billions
            "min_volume_m": 1.0          # Volum diari mínim en Millions
        }
        
    def analyze(self, symbol: str, hist_data: pd.DataFrame, info_data: dict, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detecta accions que han caigut respecte al seu recent màxim però que
        ja han començat a rebotar des d'un mínim.
        """
        p = self.default_parameters
        if config:
            p.update(config)
            
        result = {
            "is_opportunity": False,
            "confidence": 0.0,
            "current_price": 0.0,
            "reason": ""
        }
        
        if hist_data.empty or len(hist_data) < p["lookback_days"]:
            result["reason"] = f"Manca dades històriques per a {symbol}"
            return result
            
        current_price = hist_data['Close'].iloc[-1]
        result["current_price"] = current_price
        
        # 1. Comprovació de filtres d'empresa (Market Cap & Volume)
        market_cap = info_data.get("market_cap", 0) / 1e9 # A Billions
        avg_volume_10d = hist_data['Volume'].tail(10).mean() / 1e6 # A Millions
        
        if market_cap < p["min_market_cap_b"]:
            result["reason"] = f"Capitalització massa baixa ({market_cap:.1f}B < {p['min_market_cap_b']}B)"
            return result
            
        if avg_volume_10d < p["min_volume_m"]:
            result["reason"] = f"Volum de negociació molt baix ({avg_volume_10d:.1f}M < {p['min_volume_m']}M)"
            return result
            
        # 2. Càlcul de caiguda i fons
        # Històric de X dies anteriors sense incloure avui (o depèn de preferències)
        recent_data = hist_data.tail(p["lookback_days"])
        period_high = recent_data['High'].max()
        period_low = recent_data['Low'].min()
        
        # Calcular devaluació des del màxim
        drop_pct = ((period_high - period_low) / period_high) * 100
        
        if drop_pct < p["min_drop_pct"]:
            result["reason"] = f"No ha caigut prou ({drop_pct:.1f}% < {p['min_drop_pct']}%) des del recent màxim de ${period_high:.2f}."
            return result
            
        # 3. Calcular el rebot actual
        # Comprovem si des del fons (period_low) ha rebotat indicant inici de "recovery"
        rebound_pct = ((current_price - period_low) / period_low) * 100
        
        if rebound_pct < p["min_rebound_pct"]:
            result["reason"] = f"No hi ha signe que la sagnia hagi acabat (rebut del {rebound_pct:.1f}% < {p['min_rebound_pct']}%)."
            return result
            
        # També potser no ens interessa si ja ha pujat massa des del fons (> per exemple un percent de la caiguda).
        # Implementació senzilla per ara.
        
        # Si hem arribat aquí, és una "oposició d'entrada!"
        result["is_opportunity"] = True
        
        # Formula rudimentària de "Confiança"
        # Més caiguda i més rebot robust (però no excessiu) = més confiança.
        conf_drop = min(drop_pct / 30.0, 1.0) * 0.6  # 60% peso
        conf_rebound = min(rebound_pct / 5.0, 1.0) * 0.4 # 40% peso
        result["confidence"] = round((conf_drop + conf_rebound) * 100, 2)
        
        result["reason"] = (
            f"Oportunitat Buy the Dip detectada en {symbol}. "
            f"El valor ha caigut un {drop_pct:.1f}% des del seu màxim dels últims {p['lookback_days']} dies (${period_high:.2f}). "
            f"Actualment es troba a ${current_price:.2f}, havent rebotat un {rebound_pct:.1f}% des del mínim local (${period_low:.2f}), validant inici de recuperació."
        )
        return result
