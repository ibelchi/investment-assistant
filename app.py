import streamlit as st
import pandas as pd
import tempfile
import os
import json
from src.database.db import SessionLocal, Opportunity, StrategyConfig
from src.scanner.market_scanner import MarketScanner
from src.ai.rag_engine import RAGEngine
from src.ai.report_generator import ReportGenerator

# --- CONFIGURACIÓ PÀGINA ---
st.set_page_config(page_title="Assistent Anàlisi Inversió", layout="wide")
st.title("📈 Assistent Personal d'Anàlisi d'Inversió (Swing Trading)")

# --- TABS ---
tab_scanner, tab_history, tab_config, tab_knowledge = st.tabs([
    "🔍 Market Scanner", 
    "📚 Historial i Informes", 
    "⚙️ Configuració d'Estratègies",
    "🧠 Coneixement de l'Inversor (RAG)"
])

# --- TAB SCANNER ---
with tab_scanner:
    st.header("Escaneig de Mercat Diari")
    st.write("Cerca oportunitats en base a les estratègies actives.")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        limit = st.number_input("Límit símbols (deixar 0 o buit per tot el S&P500)", min_value=0, max_value=505, value=10)
        start_btn = st.button("Executar Escàner", type="primary")
        
    with col2:
        if start_btn:
            with st.spinner("Escanejant el mercat i avaluant estratègies... Això pot trigar una mica."):
                scanner = MarketScanner()
                scanner.run_scan(limit_symbols=limit if limit > 0 else None)
            st.success("Escaneig completat! Revisa l'Historial per veure si hi ha noves oportunitats detectades.")
            st.balloons()

# --- TAB HISTORY ---
with tab_history:
    st.header("Oportunitats Detectades")
    db = SessionLocal()
    try:
        opportunities = db.query(Opportunity).order_by(Opportunity.date_detected.desc()).limit(50).all()
        if not opportunities:
            st.info("De moment no s'ha detectat cap oportunitat al mercat.")
        else:
            # Mostrar tabla resumen
            data = []
            for op in opportunities:
                data.append({
                    "ID": op.id,
                    "Data": op.date_detected.strftime('%Y-%m-%d %H:%M'),
                    "Símbol": op.symbol,
                    "Estratègia": op.strategy_name,
                    "Preu": f"${op.current_price:.2f}"
                })
            
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
            
            st.divider()
            st.subheader("Generar Informe (IA)")
            
            selected_id = st.selectbox("Selecciona un ID per generar l'informe profund", df["ID"].tolist())
            if st.button("Generar Informe amb LangChain"):
                op = db.query(Opportunity).filter(Opportunity.id == selected_id).first()
                if op:
                    if op.explanation and not op.market_context:
                        with st.spinner("L'IA està analitzant la troballa tècnica contrastant amb el RAG..."):
                            gen = ReportGenerator()
                            # El informe es guarda a 'market_context' o el mostrem directament
                            informe = gen.generate_report(op.symbol, op.strategy_name, op.explanation, op.current_price)
                            
                            # Caching de l'informe generat
                            op.market_context = informe
                            db.commit()
                            
                    elif op.market_context:
                        informe = op.market_context
                    else:
                        informe = "No hi ha dades tècniques per arrencar l'informe."
                        
                    st.markdown(informe)
    finally:
        db.close()

# --- TAB CONFIG ---
with tab_config:
    st.header("Paràmetres de l'Estratègia")
    st.write("Ajusta els valors de simulació sense necessitat de tocar el codi.")
    
    # Truc de lectura directa (només tenim 1 plugin de moment)
    from src.strategies.buy_the_dip import BuyTheDipStrategy
    btd = BuyTheDipStrategy()
    
    db = SessionLocal()
    try:
        conf_record = db.query(StrategyConfig).filter(StrategyConfig.strategy_name == btd.name).first()
        actual_params = conf_record.parameters if conf_record else btd.default_parameters
        
        with st.form("btd_config"):
            st.subheader(btd.name)
            
            col1, col2 = st.columns(2)
            with col1:
                min_drop = st.slider("Caiguda Mínima (%)", 5.0, 50.0, actual_params.get("min_drop_pct", 15.0), 0.5)
                lookback = st.slider("Finestra Històrica (Dies)", 20, 250, actual_params.get("lookback_days", 60), 5)
                min_rebound = st.slider("Rebot Mínim (%)", 0.0, 15.0, actual_params.get("min_rebound_pct", 2.0), 0.5)
            
            with col2:
                mc = st.number_input("Capitalització Mínima (B $)", 0.0, 1000.0, actual_params.get("min_market_cap_b", 10.0))
                vol = st.number_input("Volum Mínim (M)", 0.0, 100.0, actual_params.get("min_volume_m", 1.0))
                
            guardar = st.form_submit_button("Guardar Canvis")
            if guardar:
                new_p = {
                    "min_drop_pct": min_drop,
                    "lookback_days": lookback,
                    "min_rebound_pct": min_rebound,
                    "min_market_cap_b": mc,
                    "min_volume_m": vol
                }
                if not conf_record:
                    conf_record = StrategyConfig(strategy_name=btd.name, parameters=new_p)
                    db.add(conf_record)
                else:
                    conf_record.parameters = new_p
                db.commit()
                st.success("Configuració guardada i aplicada pel següent escaneig.")
    finally:
        db.close()

# --- TAB KNOWLEDGE (RAG) ---
with tab_knowledge:
    st.header("Documentació Guia de l'Inversor")
    st.write("Puja llibres, mètodes o fitxers de notes sobre Swing Trading per que la IA ho interioritzi al generar informes.")
    
    pdf_docs = st.file_uploader("Puja els teus llibres en PDF ací", accept_multiple_files=True, type=['pdf'])
    if st.button("Processar i Injectar Coneixement"):
        if pdf_docs:
            with st.spinner("Fragmentant i vectoritzant els documents (FAISS + OpenAI Embeddings)..."):
                eng = RAGEngine()
                for pdf_file in pdf_docs:
                    # Guardem el text pujat
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(pdf_file.getvalue())
                        tmp_path = tmp.name
                    
                    ok = eng.process_pdf(tmp_path)
                    os.unlink(tmp_path)
                    
                    if ok:
                        st.success(f"Document '{pdf_file.name}' indexat exitosament.")
                    else:
                        st.error(f"He fallat indexant l'arxiu '{pdf_file.name}'.")
        else:
            st.warning("Selecciona com a mínim un document.")
