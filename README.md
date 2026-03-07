# Assistent Personal d'Anàlisi d'Inversió (Swing Trading)

Un assistent autònom dissenyat per escanejar el mercat de valors de l'S&P 500, trobar oportunitats paramètriques per swing trading ("Buy the Dip") i explicar l'escat de mercat combinant anàlisi tècnica amb documents de coneixement d'inversió (RAG).

## Funcionalitats

* **Market Scanner:** Escàner automàtic de condicions tècniques per filtratge.
* **Sistema d'Estratègies:** Connectable. De base inclou l'estratègia configurables de recuperació post-caiguda.
* **Generació d'informes IA:** Combina la lògica calculada amb consells RAG pujats des de PDFs de l'inversor gràcies a LangChain i models OpenAI.
* **Streamlit UI:** Base de dades SQLite automatitzada i un quadre de comandaments interactiu de fàcil ús.
* **Simulació històrica (Backtesting):** Capacitat per avaluar regles al passat.

## Requisits

* Python 3.9+
* API Key de OpenAI (per a que els reports usin ChatGPT per llegir).

## Instal·lació

```bash
git clone <aquest-repo>
cd investment_assistant
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

*(Pendent de generar el llistat requirements.txt si s'escau)*

Obrir la interfície via PowerShell:
```powershell
streamlit run app.py
```
O fer servir la drecera script .BAT (Windows) creat al directori d'arrel.
