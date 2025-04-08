# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import camelot
import tempfile
import os
import io
import numpy as np
import re
import base64
from PIL import Image
import fitz  # PyMuPDF
import json
# Ensure google.generativeai is installed and imported correctly
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# --- Configuration ---
GEMINI_IMAGE_RESOLUTION_FACTOR = 3
# Select the Gemini model to use (e.g., 'gemini-1.5-flash', 'gemini-pro-vision')
# Note: 'gemini-2.0-flash-exp' from the example might be experimental or specific.
# 'gemini-1.5-flash' is a good general-purpose choice.
GEMINI_MODEL_NAME = 'gemini-1.5-flash'

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Conversor PDF para Excel",
    page_icon="📊",
    layout="wide"
)

# --- PDF Processing Functions ---

def extract_images_from_pdf_page(pdf_path, page_num):
    """Extracts a high-resolution image of a specific PDF page."""
    temp_img_path = None # Initialize path
    doc = None # Initialize doc
    try:
        doc = fitz.open(pdf_path)
        if page_num >= len(doc):
            st.warning(f"Número de página {page_num + 1} está fora do intervalo (Total: {len(doc)}).")
            return None
        page = doc[page_num]
        matrix = fitz.Matrix(GEMINI_IMAGE_RESOLUTION_FACTOR, GEMINI_IMAGE_RESOLUTION_FACTOR)
        pix = page.get_pixmap(matrix=matrix)
        temp_img_path = f"temp_page_{page_num + 1}.png"
        pix.save(temp_img_path)
        return temp_img_path
    except Exception as e:
        st.error(f"Erro ao extrair imagem da página {page_num + 1}: {e}")
        return None
    finally:
        # Ensure the document is closed
        if doc:
            doc.close()

def parse_page_input(page_input_str, max_pages):
    """Parses the user's page input string into a list of 0-based page indices."""
    # (Implementation remains the same as the previous version)
    if not page_input_str or page_input_str.strip().lower() == 'all':
        return list(range(max_pages))
    page_numbers = set()
    parts = page_input_str.split(',')
    for part in parts:
        part = part.strip()
        if not part: continue
        if '-' in part:
            try:
                start, end = map(str.strip, part.split('-'))
                start_idx = int(start) - 1
                end_idx = int(end)
                if start_idx < 0 or end_idx > max_pages or start_idx >= end_idx:
                    st.warning(f"Intervalo de páginas inválido '{part}' (páginas de 1 a {max_pages}). Ignorando.")
                    continue
                page_numbers.update(range(start_idx, end_idx))
            except ValueError:
                st.warning(f"Formato de intervalo inválido '{part}'. Ignorando.")
        else:
            try:
                page_idx = int(part) - 1
                if 0 <= page_idx < max_pages:
                    page_numbers.add(page_idx)
                else:
                    st.warning(f"Número de página inválido '{part}' (páginas de 1 a {max_pages}). Ignorando.")
            except ValueError:
                st.warning(f"Formato de página inválido '{part}'. Ignorando.")
    return sorted(list(page_numbers))

# --- Updated Gemini Function ---
def detect_tables_with_gemini(pdf_path, page_input, api_key):
    """Detects and transcribes tables using Gemini VLM via genai.Client."""
    if not GEMINI_AVAILABLE:
        st.error("Gemini VLM não está disponível. Verifique a instalação: pip install google-generativeai")
        return []
    if not api_key:
         st.error("Chave API Gemini não fornecida na barra lateral.")
         return []

    # 1. Create the Gemini Client
    try:
        client = genai.Client(api_key=api_key)
        # Optionally, verify the client/key here, e.g., by trying to list models
        # client.list_models()
    except Exception as e:
         st.error(f"Erro ao criar o cliente Gemini: {e}. Verifique sua chave API.")
         return []

    # Get page count and parse requested pages
    try:
        doc = fitz.open(pdf_path)
        max_pages = len(doc)
        doc.close()
    except Exception as e:
        st.error(f"Erro ao abrir o PDF para contagem de páginas: {e}")
        return []

    page_numbers_to_process = parse_page_input(page_input, max_pages)
    if not page_numbers_to_process:
         st.warning("Nenhuma página válida selecionada para processamento.")
         return []

    all_detected_tables = []
    temp_img_files = []

    # --- Gemini Prompt (same as before) ---
    prompt = """
    Analise a imagem fornecida, que é uma página de um documento.
    Identifique TODAS as tabelas presentes nesta página.
    Para CADA tabela encontrada, transcreva seu conteúdo (cabeçalhos e dados).
    Formate TODA a saída EXCLUSIVAMENTE como um único objeto JSON.
    O objeto JSON deve ter uma chave principal 'tables'. O valor de 'tables' deve ser uma LISTA de objetos, onde cada objeto representa uma tabela.
    Cada objeto de tabela deve ter DUAS chaves:
    1. 'headers': Uma lista de strings representando os nomes das colunas da tabela.
    2. 'data': Uma lista de listas, onde cada lista interna representa uma linha de dados da tabela, com os valores correspondendo aos cabeçalhos.

    Exemplo de formato JSON esperado:
    {
      "tables": [
        {
          "headers": ["Coluna A", "Coluna B", "Coluna C"],
          "data": [
            ["Linha1 A", "Linha1 B", "Linha1 C"],
            ["Linha2 A", "Linha2 B", "Linha2 C"]
          ]
        },
        {
          "headers": ["ID", "Nome"],
          "data": [
            ["1", "Alice"],
            ["2", "Bob"]
          ]
        }
      ]
    }

    Certifique-se de que a resposta contenha APENAS o objeto JSON e nada mais (sem texto introdutório, explicações ou marcadores como ```json).
    Se nenhuma tabela for encontrada na página, retorne: {"tables": []}
    """

    with st.spinner(f"Processando {len(page_numbers_to_process)} página(s) com Gemini VLM..."):
        for page_num in page_numbers_to_process:
            st.info(f"Processando página {page_num + 1}/{max_pages} com Gemini VLM...")

            temp_img_path = extract_images_from_pdf_page(pdf_path, page_num)
            if not temp_img_path:
                st.warning(f"Pulando página {page_num + 1} devido a erro na extração da imagem.")
                continue
            temp_img_files.append(temp_img_path)

            image_bytes = None
            try:
                # 2. Read image bytes and prepare the Part
                with open(temp_img_path, "rb") as image_file:
                    image_bytes = image_file.read()

                # Create the image Part using types.Part.from_bytes
                image_part = genai.types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png" # Since we save as PNG
                )

                # 3. Call generate_content using the client
                response = client.generate_content(  # Use client.generate_content
                    model=GEMINI_MODEL_NAME,         # Specify model here
                    contents=[prompt, image_part],   # Pass contents list
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=8192,
                    ),
                    request_options={'timeout': 600},
                    stream=False
                )

                # 4. Process response (same JSON parsing as before)
                response_text = response.text
                match = re.search(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', response_text, re.DOTALL | re.IGNORECASE)

                if match:
                    json_str = match.group(1) or match.group(2)
                    try:
                        detected_data = json.loads(json_str)
                        if isinstance(detected_data, dict) and 'tables' in detected_data and isinstance(detected_data['tables'], list):
                            page_tables = detected_data['tables']
                            if not page_tables: st.info(f"Nenhuma tabela detectada pelo Gemini na página {page_num + 1}.")
                            else: st.info(f"Gemini detectou {len(page_tables)} tabela(s) na página {page_num + 1}.")

                            for i, table_data in enumerate(page_tables):
                                if isinstance(table_data, dict) and 'headers' in table_data and 'data' in table_data:
                                     if isinstance(table_data['headers'], list) and isinstance(table_data['data'], list):
                                        df = pd.DataFrame(table_data['data'], columns=table_data['headers'])
                                        all_detected_tables.append({
                                            'page': page_num + 1, 'table_index_on_page': i + 1,
                                            'dataframe': df, 'source': 'Gemini'
                                        })
                                     else: st.warning(f"Formato headers/data inválido na Tabela {i+1}/Página {page_num + 1} (Gemini).")
                                else: st.warning(f"Estrutura JSON Tabela {i+1}/Página {page_num + 1} incorreta (Gemini).")
                        else: st.warning(f"Estrutura JSON principal ('tables') não encontrada na Página {page_num + 1} (Gemini).")
                    except json.JSONDecodeError as json_e:
                        st.error(f"Erro ao decodificar JSON da Página {page_num + 1}: {json_e}")
                        st.text_area("Resposta Bruta (Erro JSON)", response_text, height=150)
                else:
                    st.warning(f"Nenhum bloco JSON encontrado na resposta da Página {page_num + 1} (Gemini).")
                    st.text_area("Resposta Bruta (JSON não encontrado)", response_text, height=150)

            except Exception as e:
                 # Catch potential API errors, timeouts, etc.
                 st.error(f"Erro durante chamada à API Gemini para página {page_num + 1}: {type(e).__name__} - {e}")
            finally:
                # Clean up image bytes variable if needed (though Python's GC handles it)
                image_bytes = None

    # --- Cleanup Temporary Image Files ---
    for img_path in temp_img_files:
        try:
            if os.path.exists(img_path):
                os.remove(img_path)
        except Exception as e:
            st.warning(f"Não foi possível remover o arquivo de imagem temporário {img_path}: {e}")

    return all_detected_tables

# --- Camelot and Data Processing Functions (remain unchanged) ---

def format_pages_for_camelot(page_input_str):
    if not page_input_str or page_input_str.strip().lower() == 'all': return 'all'
    return page_input_str.strip()

def extract_tables_with_camelot(pdf_path, page_input='all'):
    extracted_tables = []
    pages_formatted = format_pages_for_camelot(page_input)
    st.info(f"Tentando extrair com Camelot (páginas: {pages_formatted})...")
    try: # Lattice
        st.write("Executando Camelot - modo Lattice...")
        lattice_tables = camelot.read_pdf(pdf_path, pages=pages_formatted, flavor='lattice', suppress_warnings=True)
        if lattice_tables.n > 0:
             st.write(f"  -> {lattice_tables.n} tabelas encontradas com Lattice.")
             extracted_tables.extend(lattice_tables)
        else: st.write("  -> Nenhuma tabela encontrada com Lattice.")
    except Exception as e: st.warning(f"Erro/Aviso Camelot Lattice: {e}")
    try: # Stream
        st.write("Executando Camelot - modo Stream...")
        stream_tables = camelot.read_pdf(pdf_path, pages=pages_formatted, flavor='stream', suppress_warnings=True)
        if stream_tables.n > 0:
            st.write(f"  -> {stream_tables.n} tabelas encontradas com Stream.")
            extracted_tables.extend(stream_tables)
        else: st.write("  -> Nenhuma tabela encontrada com Stream.")
    except Exception as e: st.warning(f"Erro/Aviso Camelot Stream: {e}")
    camelot_results = [{'page': t.page, 'table_index_on_page': i + 1, 'dataframe': t.df, 'source': f"Camelot ({t.flavor})"}
                       for i, t in enumerate(extracted_tables)]
    return camelot_results

def fix_duplicate_columns(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

def clean_dataframe(df):
    if df.empty: return df
    df = df.replace(r'^\s*$', np.nan, regex=True).dropna(axis=1, how='all').dropna(axis=0, how='all')
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.reset_index(drop=True)
    df = fix_duplicate_columns(df)
    return df

def process_extracted_tables(tables_list):
    processed = []
    for i, table_info in enumerate(tables_list):
        try:
            cleaned_df = clean_dataframe(table_info['dataframe'].copy())
            if not cleaned_df.empty:
                 processed.append({**table_info, 'dataframe': cleaned_df,
                                   'table_id': f"{table_info['source'].replace(' ','')}_Pg{table_info['page']}_T{table_info['table_index_on_page']}"})
            else: st.write(f"Tabela {i+1} ({table_info['source']}, P{table_info['page']}) descartada após limpeza.")
        except Exception as e: st.error(f"Erro ao limpar Tabela {i+1} ({table_info['source']}, P{table_info['page']}): {e}")
    return processed


# --- Streamlit UI (remains largely the same) ---
st.title("📊 Conversor de PDF para Excel - Extrator de Tabelas")
st.markdown("Extraia tabelas de arquivos PDF e salve-as em formato Excel. Use **Camelot** ou **Gemini VLM**.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurações")
    extraction_method = st.radio("Método:", ["Camelot", "Gemini VLM", "Ambos"], index=2)
    page_input = st.text_input("Páginas:", placeholder="Ex: 1, 3, 5-7 (ou deixe em branco)")
    st.markdown("---")
    st.subheader("🔑 Chave API Gemini")
    api_key_input = st.text_input("Cole sua Chave API Gemini:", type="password", key="api_key_input_sidebar")
    st.markdown("[Obtenha uma Chave API Gemini](https://aistudio.google.com/app/apikey)")
    if extraction_method in ["Gemini VLM", "Ambos"] and not GEMINI_AVAILABLE:
         st.warning("SDK Google Gemini não encontrado. `pip install google-generativeai`")

# Main Area
uploaded_file = st.file_uploader("1. Escolha um arquivo PDF", type=['pdf'])

if uploaded_file is not None:
    st.success(f"Arquivo '{uploaded_file.name}' carregado!")
    st.write(f"Tamanho: {uploaded_file.size / 1024:.2f} KB")

    gemini_needed = extraction_method in ["Gemini VLM", "Ambos"]
    # Pre-checks for Gemini
    if gemini_needed and not api_key_input:
        st.error("⚠️ Chave API Gemini necessária (na barra lateral) para usar Gemini VLM.")
        st.stop()
    if gemini_needed and not GEMINI_AVAILABLE:
        st.error("⚠️ SDK Gemini (`google-generativeai`) não instalado.")
        st.stop()

    temp_pdf_path = None # Define scope outside 'with'
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.getvalue())
            temp_pdf_path = temp_pdf.name # Store path

        st.info("2. Iniciando extração...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        all_results = []

        # Extraction Execution
        if extraction_method in ["Camelot", "Ambos"]:
            status_text.text("Executando Camelot...")
            camelot_raw_tables = extract_tables_with_camelot(temp_pdf_path, page_input)
            all_results.extend(camelot_raw_tables)
            progress_bar.progress(30 if extraction_method == "Camelot" else 15)

        if extraction_method in ["Gemini VLM", "Ambos"]:
            status_text.text("Executando Gemini VLM...")
            # Pass the API key from the sidebar input
            gemini_raw_tables = detect_tables_with_gemini(temp_pdf_path, page_input, api_key_input)
            all_results.extend(gemini_raw_tables)
            progress_bar.progress(70 if extraction_method == "Gemini VLM" else (50 if all_results else 30))

        # Processing and Display
        status_text.text("Processando tabelas...")
        if not all_results:
            st.warning("Nenhuma tabela foi extraída.")
        else:
            st.success(f"Extração inicial: {len(all_results)} tabelas brutas encontradas.")
            processed_tables = process_extracted_tables(all_results)
            progress_bar.progress(90)

            if not processed_tables:
                st.warning("Nenhuma tabela válida após processamento.")
            else:
                st.success(f"Processamento concluído! {len(processed_tables)} tabelas válidas.")
                st.markdown("---")
                st.header("📊 Tabelas Extraídas")

                output_excel = io.BytesIO()
                with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                    for i, table_info in enumerate(processed_tables):
                        df = table_info['dataframe']
                        tid = table_info['table_id']
                        sname = re.sub(r'[\\/*?:\[\]]', '_', tid)[:31]
                        with st.expander(f"Tabela {i+1}: {tid} (P{table_info['page']}) - {len(df)} linhas"):
                            st.dataframe(df)
                            out_single = io.BytesIO()
                            with pd.ExcelWriter(out_single, engine='openpyxl') as swriter:
                                df.to_excel(swriter, index=False, sheet_name=sname)
                            out_single.seek(0)
                            st.download_button(f"📥 Baixar Tabela {i+1}", out_single, f"{os.path.splitext(uploaded_file.name)[0]}_{tid}.xlsx",
                                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"dl_{i}")
                        df.to_excel(writer, index=False, sheet_name=sname)
                output_excel.seek(0)
                st.markdown("---")
                st.download_button(f"📥 Baixar TODAS as {len(processed_tables)} Tabelas (.xlsx)", output_excel,
                                   f"{os.path.splitext(uploaded_file.name)[0]}_todas_tabelas.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_all")
        progress_bar.progress(100)
        status_text.text("Processo concluído!")

    except Exception as e:
        st.error(f"Ocorreu um erro inesperado: {e}")
        import traceback
        st.error("Detalhes:")
        st.code(traceback.format_exc())
    finally:
        # Cleanup Temporary PDF
        if temp_pdf_path and os.path.exists(temp_pdf_path):
             try: os.unlink(temp_pdf_path)
             except Exception as e: st.warning(f"Não foi possível remover PDF temporário: {e}")

st.markdown("---")
st.markdown("*Desenvolvido com Streamlit, Camelot-py, PyMuPDF e Google Gemini.*")
