# -*- coding: utf-8 -*- # Add this line for encoding
import streamlit as st
import pandas as pd
import camelot # Use camelot-py, ensure it's installed
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
    import google.generativeai as genai
    # from google.generativeai import types # types is implicitly available via genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    # We will show a warning in the UI later if needed

# --- Configuration ---
# Increase rendering resolution for Gemini (higher value = better quality but slower)
GEMINI_IMAGE_RESOLUTION_FACTOR = 3 # Default was 2

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Conversor PDF para Excel",
    page_icon="📊",
    layout="wide"
)

# --- PDF Processing Functions ---

def extract_images_from_pdf_page(pdf_path, page_num):
    """Extracts a high-resolution image of a specific PDF page."""
    try:
        doc = fitz.open(pdf_path)
        if page_num >= len(doc):
            st.warning(f"Número de página {page_num + 1} está fora do intervalo (Total: {len(doc)}).")
            doc.close() # Close doc even if page num is invalid
            return None
        page = doc[page_num]

        # Render page as image with higher resolution
        matrix = fitz.Matrix(GEMINI_IMAGE_RESOLUTION_FACTOR, GEMINI_IMAGE_RESOLUTION_FACTOR)
        pix = page.get_pixmap(matrix=matrix)

        # Save image temporarily
        temp_img_path = f"temp_page_{page_num + 1}.png"
        pix.save(temp_img_path)
        doc.close() # Close the document after processing the page
        return temp_img_path
    except Exception as e:
        st.error(f"Erro ao extrair imagem da página {page_num + 1}: {e}")
        # Attempt to close doc in case of error during processing
        try:
            if 'doc' in locals() and doc.is_open:
                 doc.close()
        except:
            pass
        return None

def parse_page_input(page_input_str, max_pages):
    """Parses the user's page input string into a list of 0-based page indices."""
    if not page_input_str or page_input_str.strip().lower() == 'all':
        return list(range(max_pages))

    page_numbers = set() # Use set to avoid duplicates
    parts = page_input_str.split(',')

    for part in parts:
        part = part.strip()
        if not part: continue # Skip empty parts

        if '-' in part:
            # Page range (e.g., 5-9)
            try:
                start, end = map(str.strip, part.split('-'))
                start_idx = int(start) - 1  # Adjust to 0-based index
                end_idx = int(end)          # End is exclusive in range, so use as is

                if start_idx < 0 or end_idx > max_pages or start_idx >= end_idx:
                    st.warning(f"Intervalo de páginas inválido '{part}' (páginas de 1 a {max_pages}). Ignorando.")
                    continue
                page_numbers.update(range(start_idx, end_idx))
            except ValueError:
                st.warning(f"Formato de intervalo inválido '{part}'. Ignorando.")
        else:
            # Single page
            try:
                page_idx = int(part) - 1 # Adjust to 0-based index
                if 0 <= page_idx < max_pages:
                    page_numbers.add(page_idx)
                else:
                    st.warning(f"Número de página inválido '{part}' (páginas de 1 a {max_pages}). Ignorando.")
            except ValueError:
                st.warning(f"Formato de página inválido '{part}'. Ignorando.")

    return sorted(list(page_numbers)) # Return sorted list

# Modified: Now requires api_key to be passed
def detect_tables_with_gemini(pdf_path, page_input, api_key):
    """Detects and transcribes tables from specified PDF pages using Gemini VLM."""
    if not GEMINI_AVAILABLE:
        st.error("Gemini VLM não está disponível. Verifique a instalação do SDK: pip install google-generativeai")
        return []

    if not api_key:
         st.error("Chave API Gemini não fornecida na barra lateral.")
         return []

    try:
        genai.configure(api_key=api_key)
        # Check if configuration is successful by trying to load a model
        model = genai.GenerativeModel('gemini-1.5-flash') # Or 'gemini-pro-vision'
        # You could add a quick test call here if desired, like listing models
        # genai.list_models()
    except Exception as e:
         st.error(f"Erro ao configurar a API Gemini ou carregar modelo: {e}. Verifique sua chave API.")
         return []

    try:
        doc = fitz.open(pdf_path)
        max_pages = len(doc)
        doc.close() # Close after getting page count
    except Exception as e:
        st.error(f"Erro ao abrir o PDF para contagem de páginas: {e}")
        return []

    page_numbers_to_process = parse_page_input(page_input, max_pages)
    if not page_numbers_to_process:
         st.warning("Nenhuma página válida selecionada para processamento.")
         return []

    all_detected_tables = []
    temp_img_files = [] # Track temporary image files

    # --- Gemini Prompt ---
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
                continue # Skip if image extraction failed

            temp_img_files.append(temp_img_path)

            try:
                # Prepare image for Gemini
                page_image = Image.open(temp_img_path)

                # Send request to Gemini (Model already loaded)
                response = model.generate_content(
                    [prompt, page_image],
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1, # Lower temperature for consistency
                        max_output_tokens=8192, # Generous token limit
                    ),
                    request_options={'timeout': 600}, # Increase timeout to 10 minutes for complex pages
                    stream=False # Get the full response at once
                )

                # --- Robust JSON Parsing ---
                response_text = response.text
                match = re.search(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', response_text, re.DOTALL | re.IGNORECASE)

                if match:
                    json_str = match.group(1) or match.group(2)
                    try:
                        detected_data = json.loads(json_str)

                        if isinstance(detected_data, dict) and 'tables' in detected_data and isinstance(detected_data['tables'], list):
                            page_tables = detected_data['tables']
                            if not page_tables:
                                 st.info(f"Nenhuma tabela detectada pelo Gemini na página {page_num + 1}.")
                            else:
                                st.info(f"Gemini detectou {len(page_tables)} tabela(s) na página {page_num + 1}.")
                            for i, table_data in enumerate(page_tables):
                                if isinstance(table_data, dict) and 'headers' in table_data and 'data' in table_data:
                                     if isinstance(table_data['headers'], list) and isinstance(table_data['data'], list):
                                        df = pd.DataFrame(table_data['data'], columns=table_data['headers'])
                                        all_detected_tables.append({
                                            'page': page_num + 1,
                                            'table_index_on_page': i + 1,
                                            'dataframe': df,
                                            'source': 'Gemini'
                                        })
                                     else:
                                         st.warning(f"Formato de cabeçalhos/dados inválido na tabela {i+1} da página {page_num + 1} (resposta Gemini).")
                                else:
                                    st.warning(f"Estrutura JSON da tabela {i+1} na página {page_num + 1} está incorreta (resposta Gemini).")
                        else:
                            st.warning(f"Estrutura JSON principal ('tables' como lista) não encontrada na resposta do Gemini para a página {page_num + 1}.")

                    except json.JSONDecodeError as json_e:
                        st.error(f"Erro ao decodificar JSON da página {page_num + 1}: {json_e}")
                        st.text_area("Resposta Bruta do Gemini (Erro de JSON)", response_text, height=150)
                else:
                    st.warning(f"Nenhum bloco JSON válido encontrado na resposta do Gemini para a página {page_num + 1}.")
                    st.text_area("Resposta Bruta do Gemini (JSON não encontrado)", response_text, height=150)

            # Catch specific Gemini exceptions if possible (requires 'types')
            # except genai.types.BlockedPromptException as bpe:
            #      st.error(f"Processamento da página {page_num + 1} bloqueado pelo Gemini devido a políticas de segurança. {bpe}")
            # except genai.types.StopCandidateException as sce:
            #      st.error(f"Geração interrompida para a página {page_num + 1} pelo Gemini. {sce}")
            except Exception as e:
                st.error(f"Erro durante chamada à API Gemini para página {page_num + 1}: {type(e).__name__} - {e}")
            finally:
                 try:
                     if 'page_image' in locals() and hasattr(page_image, 'close'):
                         page_image.close()
                 except:
                     pass


    # --- Cleanup Temporary Image Files ---
    for img_path in temp_img_files:
        try:
            if os.path.exists(img_path):
                os.remove(img_path)
        except Exception as e:
            st.warning(f"Não foi possível remover o arquivo de imagem temporário {img_path}: {e}")

    return all_detected_tables


# --- Camelot and Data Processing Functions (unchanged from previous version) ---

def format_pages_for_camelot(page_input_str):
    """Formats page input for Camelot (e.g., '1,3,5-7' or 'all')."""
    if not page_input_str or page_input_str.strip().lower() == 'all':
        return 'all'
    return page_input_str.strip()

def extract_tables_with_camelot(pdf_path, page_input='all'):
    """Extracts tables using Camelot (lattice and stream)."""
    extracted_tables = []
    pages_formatted = format_pages_for_camelot(page_input)

    st.info(f"Tentando extrair com Camelot (páginas: {pages_formatted})...")

    # Try Lattice
    try:
        st.write("Executando Camelot - modo Lattice...")
        lattice_tables = camelot.read_pdf(pdf_path, pages=pages_formatted, flavor='lattice', suppress_warnings=True)
        if lattice_tables.n > 0:
             st.write(f"  -> {lattice_tables.n} tabelas encontradas com Lattice.")
             extracted_tables.extend(lattice_tables)
        else: st.write("  -> Nenhuma tabela encontrada com Lattice.")
    except Exception as e:
        st.warning(f"Erro/Aviso durante extração com Camelot Lattice: {e}")

    # Try Stream
    try:
        st.write("Executando Camelot - modo Stream...")
        stream_tables = camelot.read_pdf(pdf_path, pages=pages_formatted, flavor='stream', suppress_warnings=True)
        if stream_tables.n > 0:
            st.write(f"  -> {stream_tables.n} tabelas encontradas com Stream.")
            extracted_tables.extend(stream_tables)
        else: st.write("  -> Nenhuma tabela encontrada com Stream.")
    except Exception as e:
        st.warning(f"Erro/Aviso durante extração com Camelot Stream: {e}")

    camelot_results = []
    for i, table in enumerate(extracted_tables):
         camelot_results.append({
             'page': table.page,
             'table_index_on_page': i + 1, # Index might reset per flavor run
             'dataframe': table.df,
             'source': f"Camelot ({table.flavor})"
         })
    return camelot_results

def fix_duplicate_columns(df):
    """Renames duplicate column names by appending '_1', '_2', etc."""
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

def clean_dataframe(df):
    """Applies basic cleaning to an extracted DataFrame."""
    if df.empty: return df
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='all')
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.reset_index(drop=True)
    df = fix_duplicate_columns(df)
    return df

def process_extracted_tables(tables_list):
    """Cleans and processes a list of tables (dictionaries containing DataFrames)."""
    processed = []
    for i, table_info in enumerate(tables_list):
        try:
            df = table_info['dataframe'].copy()
            cleaned_df = clean_dataframe(df)
            if not cleaned_df.empty:
                 processed.append({
                     **table_info,
                     'dataframe': cleaned_df,
                     'table_id': f"{table_info['source'].replace(' ','')}_Pg{table_info['page']}_T{table_info['table_index_on_page']}"
                 })
            else:
                 st.write(f"Tabela {i+1} (Fonte: {table_info['source']}, Página: {table_info['page']}) ficou vazia após limpeza e foi descartada.")
        except Exception as e:
            st.error(f"Erro ao limpar tabela {i+1} (Fonte: {table_info['source']}, Página: {table_info['page']}): {e}")
    return processed


# --- Streamlit UI ---
st.title("📊 Conversor de PDF para Excel - Extrator de Tabelas")
st.markdown("""
Extraia tabelas de arquivos PDF e salve-as em formato Excel.
Use **Camelot** para extração padrão ou **Gemini VLM** (IA da Google) para tabelas complexas/imagens.
""")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configurações")
    extraction_method = st.radio(
        "Método de Extração:",
        ["Camelot", "Gemini VLM", "Ambos"],
        index=2, # Default to Both
        help="Camelot é rápido para tabelas estruturadas. Gemini VLM usa IA para tabelas complexas ou em imagens, mas requer API Key e é mais lento."
    )

    page_input = st.text_input(
        "Páginas a Processar:",
        placeholder="Ex: 1, 3, 5-7 (ou deixe em branco para todas)",
        help="Especifique páginas ou intervalos separados por vírgula. 'all' ou vazio processa o documento inteiro."
    )

    st.markdown("---")
    # API Key Input directly in sidebar
    st.subheader("🔑 Chave API Gemini")
    api_key_input = st.text_input(
        "Cole sua Chave API Gemini aqui:",
        type="password",
        key="api_key_input_sidebar",
        help="Necessária se 'Gemini VLM' ou 'Ambos' estiver selecionado. Sua chave não é armazenada."
    )
    st.markdown("[Obtenha uma Chave API Gemini](https://aistudio.google.com/app/apikey)")

    # Display warning if Gemini is needed but SDK is missing
    if extraction_method in ["Gemini VLM", "Ambos"] and not GEMINI_AVAILABLE:
         st.warning("O SDK do Google Gemini não foi encontrado. Instale com: `pip install google-generativeai`")

# --- Main Area ---
uploaded_file = st.file_uploader("1. Escolha um arquivo PDF", type=['pdf'])

if uploaded_file is not None:
    st.success(f"Arquivo '{uploaded_file.name}' carregado com sucesso!")
    st.write("Detalhes:", {"Nome": uploaded_file.name, "Tamanho": f"{uploaded_file.size / 1024:.2f} KB"})

    # Check if Gemini is required and if the key is provided
    gemini_needed = extraction_method in ["Gemini VLM", "Ambos"]
    if gemini_needed and not api_key_input:
        st.error("⚠️ Por favor, insira sua Chave API Gemini na barra lateral para usar o método Gemini VLM.")
        st.stop() # Stop execution if key is missing but needed

    if gemini_needed and not GEMINI_AVAILABLE:
        st.error("⚠️ O método Gemini VLM foi selecionado, mas o SDK necessário (`google-generativeai`) não está instalado.")
        st.stop()


    # Use a context manager for the temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.getvalue())
        temp_pdf_path = temp_pdf.name

    st.info("2. Iniciando extração de tabelas...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    all_results = []
    try:
        # --- Run Extraction Methods ---
        if extraction_method in ["Camelot", "Ambos"]:
            status_text.text("Executando Camelot...")
            camelot_raw_tables = extract_tables_with_camelot(temp_pdf_path, page_input)
            all_results.extend(camelot_raw_tables)
            progress_bar.progress(30 if extraction_method == "Camelot" else 15)

        if extraction_method in ["Gemini VLM", "Ambos"]:
            # Key and SDK availability already checked above
            status_text.text("Executando Gemini VLM...")
            gemini_raw_tables = detect_tables_with_gemini(temp_pdf_path, page_input, api_key_input)
            all_results.extend(gemini_raw_tables)
            progress_bar.progress(70 if extraction_method == "Gemini VLM" else (50 if all_results else 30))

        # --- Process and Display Results ---
        status_text.text("Limpando e processando tabelas encontradas...")
        if not all_results:
            st.warning("Nenhuma tabela foi extraída. Verifique as configurações ou o conteúdo do PDF.")
        else:
            st.success(f"Extração inicial concluída. {len(all_results)} tabelas brutas encontradas.")
            processed_tables = process_extracted_tables(all_results)
            progress_bar.progress(90)

            if not processed_tables:
                st.warning("Nenhuma tabela válida encontrada após o processamento/limpeza.")
            else:
                st.success(f"Processamento concluído! {len(processed_tables)} tabelas válidas prontas para visualização e download.")
                st.markdown("---")
                st.header("📊 Tabelas Extraídas")

                # Prepare Excel file in memory
                output_excel = io.BytesIO()
                with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                    for i, table_info in enumerate(processed_tables):
                        df_display = table_info['dataframe']
                        table_id = table_info['table_id']
                        sheet_name = re.sub(r'[\\/*?:\[\]]', '_', table_id) # Clean sheet name
                        sheet_name = sheet_name[:31] # Max sheet name length

                        with st.expander(f"Tabela {i+1}: {table_id} (Página {table_info['page']}) - {len(df_display)} linhas"):
                            st.dataframe(df_display)
                            # Individual download button
                            output_single = io.BytesIO()
                            # Use excel writer for single sheet download as well for consistency
                            with pd.ExcelWriter(output_single, engine='openpyxl') as single_writer:
                                df_display.to_excel(single_writer, index=False, sheet_name=sheet_name)
                            output_single.seek(0) # Reset buffer position
                            st.download_button(
                                label=f"📥 Baixar Tabela {i+1} (.xlsx)",
                                data=output_single,
                                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_{table_id}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f"dl_single_{i}"
                            )

                        # Write to the main Excel file
                        df_display.to_excel(writer, index=False, sheet_name=sheet_name)

                output_excel.seek(0) # Rewind the main buffer

                st.markdown("---")
                st.download_button(
                     label=f"📥 Baixar TODAS as {len(processed_tables)} Tabelas (.xlsx)",
                     data=output_excel,
                     file_name=f"{os.path.splitext(uploaded_file.name)[0]}_todas_tabelas.xlsx",
                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                     key="download_all"
                )

        progress_bar.progress(100)
        status_text.text("Processo concluído!")

    except Exception as e:
        st.error(f"Ocorreu um erro inesperado durante o processo: {e}")
        import traceback
        st.error("Detalhes do erro:")
        st.code(traceback.format_exc()) # Show full traceback
    finally:
        # --- Cleanup Temporary PDF File ---
        if 'temp_pdf_path' in locals() and os.path.exists(temp_pdf_path):
             try:
                 os.unlink(temp_pdf_path)
             except Exception as e:
                 st.warning(f"Não foi possível remover o arquivo PDF temporário {temp_pdf_path}: {e}")

st.markdown("---")
st.markdown("""
*Desenvolvido com Streamlit, Camelot-py, PyMuPDF e Google Gemini.*
""")
