import streamlit as st
import pandas as pd
import camelot
import tempfile
import os
import io
import numpy as np
import re

# Configuração da página Streamlit
st.set_page_config(
    page_title="Conversor de PDF para Excel",
    page_icon="📊",
    layout="wide"
)

# Função para extrair tabelas de um PDF usando Camelot
def extract_tables_from_pdf(pdf_path):
    tables_list = []
    try:
        lattice_tables = camelot.read_pdf(
            pdf_path, pages='all', flavor='lattice', line_scale=40
        )
        if len(lattice_tables) > 0:
            tables_list.extend(lattice_tables)
    except Exception as e:
        st.warning(f"Aviso ao processar com modo lattice: {str(e)}")
    
    try:
        stream_tables = camelot.read_pdf(
            pdf_path, pages='all', flavor='stream', edge_tol=150, row_tol=10
        )
        if len(stream_tables) > 0:
            tables_list.extend(stream_tables)
    except Exception as e:
        st.warning(f"Aviso ao processar com modo stream: {str(e)}")
    
    if len(tables_list) == 0:
        try:
            stream_tables_aggressive = camelot.read_pdf(
                pdf_path, pages='all', flavor='stream', edge_tol=500, row_tol=30
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

# Interface Streamlit
st.title("Conversor de PDF para Excel - Extração de Tabelas")
st.markdown("""
Este aplicativo extrai tabelas de arquivos PDF e as converte para o formato Excel.
Faça o upload de um arquivo PDF contendo tabelas e obtenha um Excel organizado.
""")

uploaded_file = st.file_uploader("Escolha um arquivo PDF", type=['pdf'])
if uploaded_file is not None:
    progress_bar = st.progress(0)
    
    file_details = {"Nome do arquivo": uploaded_file.name, "Tipo": uploaded_file.type, "Tamanho": f"{uploaded_file.size/1024:.2f} KB"}
    st.write(file_details)
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name
        
        progress_bar.progress(25)
        st.info("PDF carregado com sucesso! Iniciando extração de tabelas...")
        
        tables = extract_tables_from_pdf(temp_path)
        
        if len(tables) == 0:
            st.error("Não foi possível extrair tabelas deste PDF. Verifique se o arquivo contém tabelas visíveis.")
        else:
            st.success(f"Foram encontradas {len(tables)} tabelas no PDF!")
            
            progress_bar.progress(50)
            processed_tables = process_tables(tables)
            progress_bar.progress(75)
            
            for i, df in enumerate(processed_tables):
                with st.expander(f"Tabela {i+1}"):
                    st.dataframe(df)
                    
                    output_single = io.BytesIO()
                    with pd.ExcelWriter(output_single, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name=f'Tabela {i+1}')
                    
                    st.download_button(
                        label=f"📥 Baixar Tabela {i+1}",
                        data=output_single.getvalue(),
                        file_name=f"{uploaded_file.name.split('.')[0]}_tabela_{i+1}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            output_all = io.BytesIO()
            with pd.ExcelWriter(output_all, engine='openpyxl') as writer:
                for i, df in enumerate(processed_tables):
                    df.to_excel(writer, index=False, sheet_name=f'Tabela {i+1}')
            
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
        
with st.expander("Configurações Avançadas"):
    st.markdown("""
    ### Opções de Processamento
    - **Tabelas com linhas**: O modo 'lattice' funciona melhor para tabelas com linhas visíveis e bordas definidas.
    - **Tabelas sem linhas**: O modo 'stream' é mais adequado para tabelas baseadas em espaçamento.
    - **Problemas com colunas duplicadas**: A aplicação tenta resolver automaticamente.
    """)

st.markdown("---")
st.markdown("""
### Observações:
- Esta aplicação utiliza a biblioteca Camelot para extrair tabelas de PDFs.
- Dependências necessárias: Python 3.6+, Camelot, Ghostscript e OpenCV.
- Baseado no repositório: https://github.com/atlanhq/camelot
""")
