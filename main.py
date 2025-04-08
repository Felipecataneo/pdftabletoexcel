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
    # Renomeado para genai para seguir a convenção comum
    import google.generativeai as genai
    # O import de types é geralmente feito através de genai
    # from google.genai import types # Não é necessário importar separadamente assim
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    # Adicionar um placeholder se genai não estiver disponível
    class MockGenai:
        class types:
            class Part:
                @staticmethod
                def from_bytes(data, mime_type):
                    return None # Placeholder
        class Client:
             def __init__(self, api_key): pass
             def get_generative_model(self, model_name): return MockModel()
        class GenerationConfig:
            def __init__(self, response_mime_type): pass


    class MockModel:
        def generate_content(self, contents, generation_config=None):
            raise ImportError("google.generativeai não instalado")

    genai = MockGenai() # Usar o mock se o real não for encontrado

# --- Configuration ---
GEMINI_IMAGE_RESOLUTION_FACTOR = 3
# Select the Gemini model to use (e.g., 'gemini-1.5-flash', 'gemini-pro-vision')
# Note: 'gemini-2.0-flash-exp' from the example might be experimental or specific.
# 'gemini-1.5-flash' is a good general-purpose choice.
GEMINI_MODEL_NAME = 'gemini-1.5-flash' # Usando o modelo flash que é mais rápido e recente

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
        # Aumentar a resolução para melhor análise do Gemini
        matrix = fitz.Matrix(GEMINI_IMAGE_RESOLUTION_FACTOR, GEMINI_IMAGE_RESOLUTION_FACTOR)
        pix = page.get_pixmap(matrix=matrix)
        # Criar um nome de arquivo temporário seguro
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img_file:
             temp_img_path = temp_img_file.name
             pix.save(temp_img_path) # Salva a imagem no caminho temporário
        return temp_img_path
    except Exception as e:
        st.error(f"Erro ao extrair imagem da página {page_num + 1}: {e}")
        # Limpar o arquivo temporário se ele foi criado mas houve erro depois
        if temp_img_path and os.path.exists(temp_img_path):
             try: os.remove(temp_img_path)
             except Exception: pass # Ignorar erros na limpeza durante o tratamento de outro erro
        return None
    finally:
        # Ensure the document is closed
        if doc:
            doc.close()

def parse_page_input(page_input_str, max_pages):
    """Parses the user's page input string into a list of 0-based page indices."""
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
                # Convertendo para índice 0-based
                start_idx = int(start) - 1
                # End é inclusivo na entrada do usuário, então o range vai até end_idx
                end_idx = int(end)
                if start_idx < 0 or end_idx > max_pages or start_idx >= end_idx:
                    st.warning(f"Intervalo de páginas inválido '{part}' (páginas de 1 a {max_pages}). Ignorando.")
                    continue
                # range(start, end) vai até end-1, então usamos end_idx diretamente
                page_numbers.update(range(start_idx, end_idx))
            except ValueError:
                st.warning(f"Formato de intervalo inválido '{part}'. Ignorando.")
        else:
            try:
                # Convertendo para índice 0-based
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
    """Detects and transcribes tables using Gemini VLM via genai API."""
    if not GEMINI_AVAILABLE:
        st.error("Gemini VLM não está disponível. Verifique a instalação: pip install google-generativeai")
        return []
    if not api_key:
         st.error("Chave API Gemini não fornecida na barra lateral.")
         return []

    # 1. Configure the Gemini Client (this automatically uses the API key from env or arguments)
    try:
        # A configuração da API Key é feita globalmente ou ao criar o cliente
        genai.configure(api_key=api_key)
        # Não há necessidade explícita de criar um 'Client' object como na versão anterior da API
    except Exception as e:
         st.error(f"Erro ao configurar a API Gemini: {e}. Verifique sua chave API.")
         return []

    # 2. Get the generative model
    try:
        # Obter o modelo generativo especificado
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    except Exception as e:
        st.error(f"Erro ao obter o modelo Gemini '{GEMINI_MODEL_NAME}': {e}")
        return []

    # Get page count and parse requested pages
    doc = None
    try:
        doc = fitz.open(pdf_path)
        max_pages = len(doc)
    except Exception as e:
        st.error(f"Erro ao abrir o PDF para contagem de páginas: {e}")
        if doc: doc.close() # Garante fechar se abriu mas falhou depois
        return []
    finally:
        if doc: doc.close()


    page_numbers_to_process = parse_page_input(page_input, max_pages)
    if not page_numbers_to_process:
         st.warning("Nenhuma página válida selecionada para processamento.")
         return []

    all_detected_tables = []
    temp_img_files = [] # Manter controle dos arquivos de imagem temporários criados

    # --- Gemini Prompt (mantido como antes, mas agora confiamos mais no response_mime_type) ---
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

    # --- Configuração para garantir saída JSON ---
    # CORREÇÃO: Usar a configuração para forçar a saída JSON
    generation_config = genai.types.GenerationConfig(
        response_mime_type="application/json"
    )

    with st.spinner(f"Processando {len(page_numbers_to_process)} página(s) com Gemini VLM..."):
        for page_num in page_numbers_to_process:
            st.info(f"Processando página {page_num + 1}/{max_pages} com Gemini VLM...")

            temp_img_path = extract_images_from_pdf_page(pdf_path, page_num)
            if not temp_img_path:
                st.warning(f"Pulando página {page_num + 1} devido a erro na extração da imagem.")
                continue
            temp_img_files.append(temp_img_path) # Adiciona à lista para limpeza posterior

            image_part = None # Inicializar
            try:
                # Ler bytes da imagem temporária
                with open(temp_img_path, "rb") as image_file:
                    image_bytes = image_file.read()

                # Criar a parte da imagem para a API
                # CORREÇÃO: Usar a forma correta de criar Part na API atual
                image_part = {
                    "mime_type": "image/png", # Como salvamos em PNG
                    "data": image_bytes
                }

                # Combinar prompt e imagem para a chamada
                contents = [prompt, image_part]

                # 3. Chamar generate_content no modelo, especificando a saída JSON
                # CORREÇÃO: Usar o objeto 'model' e passar 'generation_config'
                response = model.generate_content(
                    contents=contents,
                    generation_config=generation_config
                )

                # 4. Processar a resposta - agora esperamos JSON diretamente
                # CORREÇÃO: Remover regex e tentar decodificar response.text diretamente
                response_text = response.text
                try:
                    detected_data = json.loads(response_text)
                    if isinstance(detected_data, dict) and 'tables' in detected_data and isinstance(detected_data['tables'], list):
                        page_tables = detected_data['tables']
                        if not page_tables: st.info(f"Nenhuma tabela detectada pelo Gemini na página {page_num + 1}.")
                        else: st.info(f"Gemini detectou {len(page_tables)} tabela(s) na página {page_num + 1}.")

                        for i, table_data in enumerate(page_tables):
                            # Validação da estrutura interna da tabela
                            if isinstance(table_data, dict) and \
                               'headers' in table_data and isinstance(table_data['headers'], list) and \
                               'data' in table_data and isinstance(table_data['data'], list):
                                # Verificar se os dados são uma lista de listas (importante!)
                                if all(isinstance(row, list) for row in table_data['data']):
                                     # Verificar consistência no número de colunas (opcional, mas bom)
                                     num_headers = len(table_data['headers'])
                                     consistent_rows = [row for row in table_data['data'] if len(row) == num_headers]
                                     if len(consistent_rows) != len(table_data['data']):
                                         st.warning(f"Inconsistência no número de colunas nos dados da Tabela {i+1}/Página {page_num + 1}. Tentando usar linhas consistentes.")
                                         # Poderia tentar preencher ou usar apenas linhas consistentes
                                         if not consistent_rows:
                                             st.warning(f"Nenhuma linha consistente encontrada na Tabela {i+1}/Página {page_num + 1}. Pulando tabela.")
                                             continue # Pula esta tabela específica
                                         df = pd.DataFrame(consistent_rows, columns=table_data['headers'])
                                     else:
                                         df = pd.DataFrame(table_data['data'], columns=table_data['headers'])

                                     all_detected_tables.append({
                                         'page': page_num + 1, 'table_index_on_page': i + 1,
                                         'dataframe': df, 'source': 'Gemini'
                                     })
                                else:
                                    st.warning(f"Formato 'data' inválido (não é lista de listas) na Tabela {i+1}/Página {page_num + 1} (Gemini).")
                            else: st.warning(f"Estrutura JSON da Tabela {i+1}/Página {page_num + 1} incorreta (faltando 'headers' ou 'data', ou tipos errados) (Gemini).")
                    else:
                        st.warning(f"Estrutura JSON principal ('tables' não encontrada ou não é uma lista) na Página {page_num + 1} (Gemini). Resposta recebida:")
                        st.text_area("Resposta Bruta (Estrutura Inválida)", response_text, height=150)

                except json.JSONDecodeError as json_e:
                    # Erro ao decodificar JSON mesmo solicitando application/json
                    st.error(f"Erro ao decodificar JSON da Página {page_num + 1} (mesmo com response_mime_type): {json_e}")
                    st.text_area("Resposta Bruta (Erro JSON)", response_text, height=150)
                except AttributeError as ae:
                    # Pode acontecer se a resposta não tiver o atributo 'text' esperado
                     st.error(f"Erro ao acessar o texto da resposta do Gemini para página {page_num + 1}: {ae}")
                     st.json(response) # Mostrar a estrutura completa da resposta para depuração
                except Exception as e_resp:
                     # Capturar outros erros inesperados no processamento da resposta
                      st.error(f"Erro inesperado ao processar resposta do Gemini para página {page_num + 1}: {type(e_resp).__name__} - {e_resp}")
                      st.text_area("Resposta Bruta (Erro Genérico)", response.text if hasattr(response,'text') else str(response) , height=150)


            except Exception as e_api:
                 # Capturar potenciais erros da API (rede, autenticação, etc.)
                 st.error(f"Erro durante chamada à API Gemini para página {page_num + 1}: {type(e_api).__name__} - {e_api}")
                 # Adicionar mais detalhes se for um erro específico da API GenAI
                 if hasattr(e_api, 'response'):
                      st.error(f"Detalhes da resposta da API: {e_api.response}")
            finally:
                # Limpar a variável de bytes da imagem (embora o GC do Python cuide disso)
                image_bytes = None
                # Não remover o temp_img_path aqui, ele será removido no bloco finally externo

    # --- Limpeza dos Arquivos de Imagem Temporários ---
    # Mover este bloco para fora do loop e do spinner
    st.info("Limpando arquivos temporários...")
    for img_path in temp_img_files:
        try:
            if os.path.exists(img_path):
                os.remove(img_path)
                # st.write(f"Removido: {img_path}") # Descomentar para debug
        except Exception as e:
            st.warning(f"Não foi possível remover o arquivo de imagem temporário {img_path}: {e}")

    return all_detected_tables

# --- Camelot and Data Processing Functions ---

def format_pages_for_camelot(page_input_str):
    # Função auxiliar para garantir que 'all' ou a string formatada seja passada
    if not page_input_str or page_input_str.strip().lower() == 'all':
        return 'all'
    # Camelot espera uma string separada por vírgulas
    return page_input_str.strip()

def extract_tables_with_camelot(pdf_path, page_input='all'):
    extracted_tables = []
    pages_formatted = format_pages_for_camelot(page_input)
    st.info(f"Tentando extrair com Camelot (páginas: {pages_formatted})...")

    # CORREÇÃO: Remover suppress_warnings=True
    try: # Lattice
        st.write("Executando Camelot - modo Lattice...")
        # Removido suppress_warnings=True
        lattice_tables = camelot.read_pdf(pdf_path, pages=pages_formatted, flavor='lattice')
        if lattice_tables.n > 0:
             st.write(f"  -> {lattice_tables.n} tabelas encontradas com Lattice.")
             extracted_tables.extend(lattice_tables)
        else: st.write("  -> Nenhuma tabela encontrada com Lattice.")
    except Exception as e: st.warning(f"Erro/Aviso Camelot Lattice: {e}")

    # CORREÇÃO: Remover suppress_warnings=True
    try: # Stream
        st.write("Executando Camelot - modo Stream...")
        # Removido suppress_warnings=True
        stream_tables = camelot.read_pdf(pdf_path, pages=pages_formatted, flavor='stream')
        if stream_tables.n > 0:
            st.write(f"  -> {stream_tables.n} tabelas encontradas com Stream.")
            extracted_tables.extend(stream_tables)
        else: st.write("  -> Nenhuma tabela encontrada com Stream.")
    except Exception as e: st.warning(f"Erro/Aviso Camelot Stream: {e}")

    # Processar resultados do Camelot para formato consistente
    camelot_results = []
    for i, t in enumerate(extracted_tables):
        # Camelot pode retornar múltiplos dataframes por Table object às vezes,
        # mas geralmente t.df é o principal. Vamos usar o índice original do Camelot.
        camelot_results.append({
            'page': t.page,
            'table_index_on_page': t.order, # Usar a ordem que Camelot atribui
            'dataframe': t.df,
            'source': f"Camelot ({t.flavor})"
            })
    return camelot_results

def fix_duplicate_columns(df):
    """Renomeia colunas duplicadas adicionando _1, _2, etc."""
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        # Obter índices onde a coluna é duplicada
        dup_indices = cols[cols == dup].index.tolist()
        # Renomear a partir da segunda ocorrência
        for i, idx in enumerate(dup_indices):
            cols[idx] = f"{dup}_{i}" if i > 0 else dup # Mantém a primeira ocorrência com o nome original
    df.columns = cols
    return df

def clean_dataframe(df):
    """Limpa o DataFrame: remove espaços, linhas/colunas vazias, renomeia duplicatas."""
    if df.empty: return df
    # Remover colunas/linhas que são inteiramente NaN ou vazias
    df = df.replace(r'^\s*$', np.nan, regex=True) # Substitui strings vazias/espaços por NaN
    df = df.dropna(axis=1, how='all') # Remove colunas totalmente NaN
    df = df.dropna(axis=0, how='all') # Remove linhas totalmente NaN

    if df.empty: return df # Verificar novamente após dropna

    # Remover espaços em branco no início/fim das células de string
    # Usar applymap foi depreciado, usar dtypes para aplicar apenas em strings
    for col in df.select_dtypes(include=['object', 'string']).columns:
         df[col] = df[col].str.strip()

    # Renomear colunas duplicadas
    df = fix_duplicate_columns(df)

    # Resetar índice após remover linhas
    df = df.reset_index(drop=True)
    return df

def process_extracted_tables(tables_list):
    """Limpa e prepara as tabelas extraídas para exibição e download."""
    processed = []
    if not tables_list: return processed # Retorna lista vazia se não houver tabelas

    # Agrupar por página e fonte para tentar reindexar dentro da página se necessário
    # Mas por agora, vamos apenas usar os índices fornecidos pelas ferramentas
    for i, table_info in enumerate(tables_list):
        try:
            # Fazer uma cópia para não alterar o original na lista
            cleaned_df = clean_dataframe(table_info['dataframe'].copy())
            if not cleaned_df.empty:
                 # Criar um ID único e seguro para nome de aba/arquivo
                 source_simple = table_info['source'].replace(' ','').replace('(','').replace(')','')
                 table_id = f"{source_simple}_Pg{table_info['page']}_T{table_info['table_index_on_page']}"
                 # Adicionar a tabela processada à lista
                 processed.append({
                     **table_info, # Mantém informações originais como page, source
                     'dataframe': cleaned_df,
                     'table_id': table_id
                     })
            else:
                # Informar se uma tabela foi descartada após a limpeza
                st.write(f"Tabela {i+1} ({table_info.get('source','Desconhecido')}, P{table_info.get('page','?')}) descartada por estar vazia após limpeza.")
        except Exception as e:
             # Capturar erros durante a limpeza de uma tabela específica
            st.error(f"Erro ao limpar Tabela {i+1} ({table_info.get('source','Desconhecido')}, P{table_info.get('page','?')}), index {table_info.get('table_index_on_page','?')}: {e}")
    return processed


# --- Streamlit UI ---
st.title("📊 Conversor de PDF para Excel - Extrator de Tabelas")
st.markdown("Extraia tabelas de arquivos PDF e salve-as em formato Excel. Use **Camelot** (Lattice/Stream) ou **Gemini VLM**.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurações")
    extraction_method = st.radio(
        "Método de Extração:",
        ["Camelot", "Gemini VLM", "Ambos"],
        index=2, # Padrão para 'Ambos'
        help="Camelot é bom para tabelas com linhas claras. Gemini VLM usa IA para analisar a imagem da página e pode lidar com tabelas mais complexas ou sem linhas, mas requer uma Chave API."
    )
    page_input = st.text_input(
        "Páginas a Processar:",
        placeholder="Todas (Ex: 1, 3, 5-7)",
        help="Deixe em branco para processar todas as páginas. Formatos: '1', '1,3', '1-3', '1,3-5'."
    )
    st.markdown("---")
    st.subheader("🔑 Chave API Gemini")
    api_key_input = st.text_input(
        "Cole sua Chave API Gemini:",
        type="password",
        key="api_key_input_sidebar",
        help="Necessária se 'Gemini VLM' ou 'Ambos' for selecionado."
        )
    st.markdown("[Obtenha uma Chave API Gemini](https://aistudio.google.com/app/apikey)")
    if extraction_method in ["Gemini VLM", "Ambos"] and not GEMINI_AVAILABLE:
         st.warning("Biblioteca Google Gemini não encontrada. Instale com: `pip install google-generativeai`")
    elif extraction_method in ["Gemini VLM", "Ambos"] and not api_key_input:
        st.warning("Por favor, insira sua Chave API Gemini para usar este método.")

# Main Area
uploaded_file = st.file_uploader("1. Escolha um arquivo PDF", type=['pdf'])

if uploaded_file is not None:
    st.success(f"Arquivo '{uploaded_file.name}' carregado!")
    # Mostra informações básicas do arquivo
    file_details = {"Nome":uploaded_file.name,"Tipo":uploaded_file.type,"Tamanho (KB)":f"{uploaded_file.size/1024:.2f}"}
    st.write(file_details)

    # --- Validações antes de processar ---
    gemini_needed = extraction_method in ["Gemini VLM", "Ambos"]
    # Verifica API Key se Gemini for necessário
    if gemini_needed and not api_key_input:
        st.error("⚠️ Chave API Gemini é necessária na barra lateral para usar o método Gemini VLM.")
        st.stop() # Interrompe a execução se a chave for necessária mas não fornecida

    # Verifica se a biblioteca Gemini está instalada se for necessária
    if gemini_needed and not GEMINI_AVAILABLE:
        st.error("⚠️ A biblioteca `google-generativeai` não está instalada. Execute `pip install google-generativeai` e reinicie.")
        st.stop() # Interrompe se a biblioteca não estiver disponível

    # --- Processamento ---
    temp_pdf_path = None # Definir fora do try/finally para garantir o escopo
    processing_started = False # Flag para saber se o processamento começou

    try:
        # Salvar PDF carregado em um arquivo temporário seguro
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.getvalue())
            temp_pdf_path = temp_pdf.name # Guardar o caminho do arquivo temporário

        st.info("2. Iniciando extração...")
        progress_bar = st.progress(0, text="Iniciando...")
        status_text = st.empty() # Para mostrar mensagens de status
        all_results = []
        processing_started = True # Marcar que o processamento iniciou

        # --- Execução da Extração ---
        steps_total = 0
        if extraction_method in ["Camelot", "Ambos"]: steps_total += 1
        if extraction_method in ["Gemini VLM", "Ambos"]: steps_total += 1
        steps_done = 0
        base_progress = 10 # Progresso inicial

        if extraction_method in ["Camelot", "Ambos"]:
            status_text.info("Executando Camelot (Lattice e Stream)...")
            camelot_raw_tables = extract_tables_with_camelot(temp_pdf_path, page_input)
            all_results.extend(camelot_raw_tables)
            steps_done += 1
            progress = base_progress + int((steps_done / (steps_total + 1)) * (90 - base_progress)) # +1 para etapa de processamento
            progress_bar.progress(progress, text=f"Camelot concluído ({len(camelot_raw_tables)} tabelas encontradas).")


        if extraction_method in ["Gemini VLM", "Ambos"]:
            status_text.info("Executando Gemini VLM...")
            gemini_raw_tables = detect_tables_with_gemini(temp_pdf_path, page_input, api_key_input)
            all_results.extend(gemini_raw_tables)
            steps_done += 1
            progress = base_progress + int((steps_done / (steps_total + 1)) * (90 - base_progress)) # +1 para etapa de processamento
            progress_bar.progress(progress, text=f"Gemini VLM concluído ({len(gemini_raw_tables)} tabelas encontradas).")

        # --- Processamento e Exibição dos Resultados ---
        status_text.info("Limpando e processando tabelas extraídas...")
        if not all_results:
            st.warning("Nenhuma tabela bruta foi extraída pelos métodos selecionados.")
            progress_bar.progress(100, text="Concluído - Nenhuma tabela encontrada.")
        else:
            st.success(f"Extração inicial concluída: {len(all_results)} tabelas brutas encontradas no total.")
            processed_tables = process_extracted_tables(all_results)
            progress = base_progress + int(((steps_total + 1) / (steps_total + 1)) * (90 - base_progress)) # Final da etapa de processamento
            progress_bar.progress(progress, text="Processamento concluído.")


            if not processed_tables:
                st.warning("Nenhuma tabela permaneceu válida após o processo de limpeza.")
                progress_bar.progress(100, text="Concluído - Nenhuma tabela válida.")
            else:
                st.success(f"Processamento finalizado! {len(processed_tables)} tabelas válidas prontas.")
                st.markdown("---")
                st.header("📊 Tabelas Extraídas e Processadas")

                # Preparar arquivo Excel com todas as tabelas
                output_excel = io.BytesIO()
                excel_filename = f"{os.path.splitext(uploaded_file.name)[0]}_todas_tabelas.xlsx"
                try:
                    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                        sheet_names_used = set() # Para evitar nomes de aba duplicados
                        for i, table_info in enumerate(processed_tables):
                            df = table_info['dataframe']
                            tid = table_info['table_id'] # Usar o ID gerado

                            # Limpar nome da aba para ser compatível com Excel (max 31 chars, sem caracteres inválidos)
                            safe_sheet_name = re.sub(r'[\\/*?:\[\]]', '_', tid) # Substituir caracteres inválidos
                            safe_sheet_name = safe_sheet_name[:31] # Truncar para 31 caracteres

                            # Garantir nome único para a aba
                            original_name = safe_sheet_name
                            count = 1
                            while safe_sheet_name in sheet_names_used:
                                suffix = f"_{count}"
                                max_len = 31 - len(suffix)
                                safe_sheet_name = original_name[:max_len] + suffix
                                count += 1
                            sheet_names_used.add(safe_sheet_name)

                            # Escrever no arquivo Excel geral
                            df.to_excel(writer, index=False, sheet_name=safe_sheet_name)

                            # Exibir tabela individualmente com botão de download
                            expander_title = f"Tabela {i+1}: {tid} (Origem: {table_info['source']}, Pág: {table_info['page']}) - {len(df)} linhas x {len(df.columns)} colunas"
                            with st.expander(expander_title):
                                st.dataframe(df, height=min(300, (len(df) + 1) * 35 + 3)) # Altura dinâmica limitada

                                # Botão de download para tabela individual
                                out_single = io.BytesIO()
                                with pd.ExcelWriter(out_single, engine='openpyxl') as swriter:
                                    # Usar o nome da aba seguro também no nome do arquivo individual
                                    df.to_excel(swriter, index=False, sheet_name=safe_sheet_name)
                                out_single.seek(0)
                                single_filename = f"{os.path.splitext(uploaded_file.name)[0]}_{safe_sheet_name}.xlsx"
                                st.download_button(
                                    label=f"📥 Baixar Tabela Individual (.xlsx)",
                                    data=out_single,
                                    file_name=single_filename,
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key=f"dl_single_{i}" # Chave única para cada botão
                                    )

                    output_excel.seek(0)
                    st.markdown("---")
                    # Botão de download para o arquivo Excel com todas as tabelas
                    st.download_button(
                        label=f"📥 Baixar TODAS as {len(processed_tables)} Tabelas em um Arquivo (.xlsx)",
                        data=output_excel,
                        file_name=excel_filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dl_all"
                        )
                except Exception as e_excel:
                    st.error(f"Erro ao gerar o arquivo Excel: {e_excel}")

        progress_bar.progress(100, text="Processo concluído!")
        status_text.empty() # Limpar a mensagem de status final

    except fitz.fitz.FileNotFoundError:
         st.error(f"Erro: O arquivo PDF temporário não foi encontrado ou não pôde ser criado.")
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado durante o processamento:")
        st.exception(e) # Mostra o traceback formatado do Streamlit
        if 'progress_bar' in locals(): progress_bar.progress(100, text="Erro.")
        if 'status_text' in locals(): status_text.error("Erro durante a execução.")

    finally:
        # --- Limpeza do PDF Temporário ---
        # Garante que o arquivo PDF temporário seja removido, mesmo se ocorrerem erros
        if temp_pdf_path and os.path.exists(temp_pdf_path):
             try:
                 os.unlink(temp_pdf_path)
                 # st.write(f"Arquivo PDF temporário removido: {temp_pdf_path}") # Descomentar para debug
             except Exception as e_clean:
                 st.warning(f"Não foi possível remover o arquivo PDF temporário ({temp_pdf_path}): {e_clean}")
        elif processing_started and not temp_pdf_path:
             st.warning("Arquivo PDF temporário não foi definido, limpeza não realizada.")


st.markdown("---")
st.markdown("*Desenvolvido com Streamlit, Camelot-py, PyMuPDF e Google Gemini.*")
