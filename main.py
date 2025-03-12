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
from groq import Groq
import fitz  # PyMuPDF
import json

# Configuração da página Streamlit
st.set_page_config(
    page_title="Conversor de PDF para Excel",
    page_icon="📊",
    layout="wide"
)

# Inicializar o cliente Groq
@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets.get("GROQ_API_KEY", "seu_api_key_aqui"))

# Função para extrair imagens de uma página PDF
def extract_images_from_pdf_page(pdf_path, page_num):
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    # Renderizar a página como uma imagem com alta resolução
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_data = pix.tobytes("png")
    
    # Codificar a imagem em base64 para enviar ao Groq
    encoded_img = base64.b64encode(img_data).decode('utf-8')
    
    return encoded_img

# Função para analisar a entrada de páginas e convertê-la em uma lista de números de página
def parse_page_input(page_input, max_pages):
    if not page_input or page_input.lower() == 'all':
        return list(range(max_pages))
    
    page_numbers = []
    parts = page_input.split(',')
    
    for part in parts:
        if '-' in part:
            # Range de páginas (ex: 5-9)
            start, end = part.split('-')
            try:
                start = int(start.strip()) - 1  # Ajustar para indexação 0
                end = int(end.strip())  # Fim inclusivo
                page_numbers.extend(range(start, end))
            except ValueError:
                st.warning(f"Ignorando formato inválido de página: {part}")
        else:
            # Página única
            try:
                page_numbers.append(int(part.strip()) - 1)  # Ajustar para indexação 0
            except ValueError:
                st.warning(f"Ignorando formato inválido de página: {part}")
    
    # Filtrar páginas válidas
    return [p for p in page_numbers if 0 <= p < max_pages]

# Função para detectar e transcrever tabelas usando Groq VLM
def detect_tables_with_groq(pdf_path, page_input=None):
    client = get_groq_client()
    doc = fitz.open(pdf_path)
    max_pages = len(doc)
    
    # Determinar quais páginas processar
    if not page_input or page_input.lower() == 'all':
        page_numbers = list(range(max_pages))
    else:
        page_numbers = parse_page_input(page_input, max_pages)
    
    all_detected_tables = []
    
    for page_num in page_numbers:
        st.info(f"Processando página {page_num + 1} com Groq VLM...")
        
        # Extrair imagem da página
        encoded_img = extract_images_from_pdf_page(pdf_path, page_num)
        
        # Solicitar ao modelo Groq para detectar e transcrever tabelas
        try:
            completion = client.chat.completions.create(
                model="llama-3.2-90b-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Identifique e transcreva todas as tabelas nesta imagem. Formate a saída como um JSON com o seguinte formato: { 'tables': [ { 'headers': [coluna1, coluna2, ...], 'data': [ [valor1, valor2, ...], [valor1, valor2, ...], ... ] }, {...} ] }. O JSON deve conter apenas dados tabulares, sem descrições ou explicações adicionais."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encoded_img}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0,  # Usar temperatura baixa para resultados mais precisos
                max_completion_tokens=4000,
                top_p=1,
                stream=False,
            )
            
            response_text = completion.choices[0].message.content
            
            # Extrair a parte JSON da resposta
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                try:
                    detected_tables = json.loads(json_str)
                    
                    # Verificar se o formato é o esperado
                    if 'tables' in detected_tables:
                        # Converter cada tabela para DataFrame
                        for i, table in enumerate(detected_tables['tables']):
                            if 'headers' in table and 'data' in table:
                                df = pd.DataFrame(table['data'], columns=table['headers'])
                                all_detected_tables.append({
                                    'page': page_num + 1,
                                    'table_index': i + 1,
                                    'dataframe': df
                                })
                except json.JSONDecodeError:
                    st.warning(f"Não foi possível analisar o JSON na página {page_num + 1}.")
            else:
                st.warning(f"Nenhuma tabela encontrada na página {page_num + 1} ou formato de resposta inválido.")
                
        except Exception as e:
            st.error(f"Erro ao processar a página {page_num + 1} com Groq VLM: {str(e)}")
    
    return all_detected_tables

# Função para converter string de páginas para formato Camelot
def format_pages_for_camelot(page_input):
    if not page_input or page_input.lower() == 'all':
        return 'all'
    
    # Camelot aceita o formato "1,3,4-10"
    return page_input

# Função para extrair tabelas de um PDF usando Camelot
def extract_tables_from_pdf(pdf_path, page_input='all'):
    tables_list = []
    
    # Formatar a string de páginas para o Camelot
    pages = format_pages_for_camelot(page_input)
    
    try:
        lattice_tables = camelot.read_pdf(
            pdf_path, pages=pages, flavor='lattice', line_scale=40
        )
        if len(lattice_tables) > 0:
            tables_list.extend(lattice_tables)
    except Exception as e:
        st.warning(f"Aviso ao processar com modo lattice: {str(e)}")
    
    try:
        stream_tables = camelot.read_pdf(
            pdf_path, pages=pages, flavor='stream', edge_tol=150, row_tol=10
        )
        if len(stream_tables) > 0:
            tables_list.extend(stream_tables)
    except Exception as e:
        st.warning(f"Aviso ao processar com modo stream: {str(e)}")
    
    if len(tables_list) == 0:
        try:
            stream_tables_aggressive = camelot.read_pdf(
                pdf_path, pages=pages, flavor='stream', edge_tol=500, row_tol=30
            )
            if len(stream_tables_aggressive) > 0:
                tables_list.extend(stream_tables_aggressive)
        except Exception as e:
            st.warning(f"Aviso ao processar com configurações agressivas: {str(e)}")
    
    return tables_list

# Função para corrigir colunas duplicadas
def fix_duplicate_columns(df):
    columns = [str(col).replace('\n', ' ').strip() for col in df.columns]
    final_columns = []
    seen = set()
    
    for i, col in enumerate(columns):
        if col in seen:
            count = 1
            new_name = f"{col}_{count}"
            while new_name in seen:
                count += 1
                new_name = f"{col}_{count}"
            final_columns.append(new_name)
        else:
            final_columns.append(col)
        seen.add(final_columns[-1])
    
    df.columns = final_columns
    return df

# Função para limpar e processar os dados
def process_tables(tables):
    processed_dfs = []
    for i, table in enumerate(tables):
        try:
            df = table.df.copy()
            df = fix_duplicate_columns(df)
            df = df.replace(r'^\s*$', np.nan, regex=True).dropna(how='all').reset_index(drop=True)
            df = df.dropna(axis=1, how='all')
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            processed_dfs.append(df)
        except Exception as e:
            st.error(f"Erro ao processar tabela {i+1}: {str(e)}")
            processed_dfs.append(pd.DataFrame({'Erro': [f"Não foi possível processar tabela {i+1}: {str(e)}"]}))
    return processed_dfs

# Função para processar os DataFrames detectados pelo Groq
def process_groq_tables(detected_tables):
    processed_dfs = []
    for i, table_info in enumerate(detected_tables):
        try:
            df = table_info['dataframe'].copy()
            df = fix_duplicate_columns(df)
            df = df.replace(r'^\s*$', np.nan, regex=True).dropna(how='all').reset_index(drop=True)
            df = df.dropna(axis=1, how='all')
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            processed_dfs.append({
                'page': table_info['page'],
                'table_index': table_info['table_index'],
                'dataframe': df
            })
        except Exception as e:
            st.error(f"Erro ao processar tabela {i+1} do Groq: {str(e)}")
            processed_dfs.append({
                'page': table_info['page'],
                'table_index': table_info['table_index'],
                'dataframe': pd.DataFrame({'Erro': [f"Não foi possível processar tabela {i+1}: {str(e)}"]})
            })
    return processed_dfs

# Interface Streamlit
st.title("Conversor de PDF para Excel - Extração de Tabelas")
st.markdown("""
Este aplicativo extrai tabelas de arquivos PDF e as converte para o formato Excel.
Utiliza o modelo de visão Groq VLM para melhorar a detecção de tabelas complexas.
""")

with st.expander("Configurações Avançadas", expanded=True):
    extraction_method = st.radio(
        "Método de extração:",
        ["Camelot (Tradicional)", "Groq VLM (IA para tabelas complexas)", "Ambos (combinado)"]
    )
    
    page_input = st.text_input(
        "Páginas para processar (deixe em branco para todas):",
        help="Digite números de página separados por vírgula (ex: 1,3,5-7) ou deixe em branco para todas"
    )

uploaded_file = st.file_uploader("Escolha um arquivo PDF", type=['pdf'])
if uploaded_file is not None:
    progress_bar = st.progress(0)
    
    file_details = {"Nome do arquivo": uploaded_file.name, "Tipo": uploaded_file.type, "Tamanho": f"{uploaded_file.size/1024:.2f} KB"}
    st.write(file_details)
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name
        
        progress_bar.progress(20)
        st.info("PDF carregado com sucesso! Iniciando extração de tabelas...")
        
        # Configuração de páginas
        page_str = page_input if page_input else 'all'
        
        # Listas para armazenar os resultados
        all_tables = []
        all_groq_tables = []
        
        # Extração com Camelot (método tradicional)
        if extraction_method in ["Camelot (Tradicional)", "Ambos (combinado)"]:
            st.info("Extraindo tabelas com Camelot...")
            camelot_tables = extract_tables_from_pdf(temp_path, page_str)
            processed_camelot_tables = process_tables(camelot_tables)
            
            for i, df in enumerate(processed_camelot_tables):
                all_tables.append({
                    'source': 'Camelot',
                    'table_id': f"Camelot_{i+1}",
                    'dataframe': df
                })
            
            progress_bar.progress(50)
            st.success(f"Foram encontradas {len(camelot_tables)} tabelas com Camelot!")
        
        # Extração com Groq VLM
        if extraction_method in ["Groq VLM (IA para tabelas complexas)", "Ambos (combinado)"]:
            st.info("Extraindo tabelas com Groq VLM...")
            groq_detected_tables = detect_tables_with_groq(temp_path, page_str)
            processed_groq_tables = process_groq_tables(groq_detected_tables)
            
            for i, table_info in enumerate(processed_groq_tables):
                all_groq_tables.append({
                    'source': 'Groq',
                    'table_id': f"Groq_Pg{table_info['page']}_Tab{table_info['table_index']}",
                    'page': table_info['page'],
                    'dataframe': table_info['dataframe']
                })
            
            progress_bar.progress(80)
            st.success(f"Foram encontradas {len(groq_detected_tables)} tabelas com Groq VLM!")
        
        # Combinar os resultados
        combined_tables = all_tables + all_groq_tables
        
        if len(combined_tables) == 0:
            st.error("Não foi possível extrair tabelas deste PDF. Verifique se o arquivo contém tabelas visíveis.")
        else:
            st.success(f"Total de tabelas encontradas: {len(combined_tables)}")
            
            # Mostrar as tabelas extraídas
            for i, table_info in enumerate(combined_tables):
                with st.expander(f"Tabela {i+1} - {table_info['table_id']}"):
                    st.dataframe(table_info['dataframe'])
                    
                    output_single = io.BytesIO()
                    with pd.ExcelWriter(output_single, engine='openpyxl') as writer:
                        table_info['dataframe'].to_excel(writer, index=False, sheet_name=f'Tabela {i+1}')
                    
                    st.download_button(
                        label=f"📥 Baixar Tabela {i+1}",
                        data=output_single.getvalue(),
                        file_name=f"{uploaded_file.name.split('.')[0]}_{table_info['table_id']}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            # Criar arquivo Excel com todas as tabelas
            output_all = io.BytesIO()
            with pd.ExcelWriter(output_all, engine='openpyxl') as writer:
                for i, table_info in enumerate(combined_tables):
                    sheet_name = f"{table_info['table_id']}"[:31]  # Limitação do Excel para nomes de planilhas
                    table_info['dataframe'].to_excel(writer, index=False, sheet_name=sheet_name)
            
            st.download_button(
                label="📥 Baixar Todas as Tabelas",
                data=output_all.getvalue(),
                file_name=f"{uploaded_file.name.split('.')[0]}_todas_tabelas.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_all"
            )
        
        os.unlink(temp_path)
        progress_bar.progress(100)
        
    except Exception as e:
        st.error(f"Ocorreu um erro durante o processamento: {str(e)}")

st.markdown("---")
st.markdown("""
### Observações:
- Esta aplicação utiliza a biblioteca Camelot para extração tradicional de tabelas.
- O modelo Groq VLM é utilizado para detectar e transcrever tabelas complexas usando visão computacional.
- Para usar o Groq VLM, você precisa configurar uma chave de API no arquivo de segredos do Streamlit.
- Dependências adicionais: `groq`, `PyMuPDF (fitz)`, `PIL`
""")