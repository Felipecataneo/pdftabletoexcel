# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import tempfile
import os
import io
import numpy as np
import re
import json

import pdfplumber

# PyMuPDF é usado apenas para renderizar a página em imagem para o Gemini.
# Mantido opcional para que o método principal (pdfplumber) nunca dependa dele.
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

# --- Gemini (novo SDK google-genai) ---
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# --- Configuration ---
GEMINI_IMAGE_RESOLUTION_FACTOR = 3
# Modelo atual e rápido. Pode ser trocado por 'gemini-2.5-pro' para casos mais difíceis.
GEMINI_MODEL_NAME = 'gemini-2.5-flash'

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Conversor PDF para Excel",
    page_icon="📊",
    layout="wide"
)

# ==========================================================================
# 1. EXTRAÇÃO PRINCIPAL (SEM IA) — pdfplumber
# ==========================================================================

# Estratégias de detecção de tabela do pdfplumber.
# "lines": usa as réguas/linhas desenhadas no PDF (ótimo para grades bem
#          delimitadas, como a tabela COMPONENT DATA).
# "text":  infere colunas/linhas pelo alinhamento do texto (ótimo para
#          tabelas sem bordas).
_PDFPLUMBER_STRATEGIES = [
    (
        "grade",
        {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 4,
            "join_tolerance": 4,
        },
    ),
    (
        "texto",
        {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "snap_tolerance": 4,
            "intersection_tolerance": 6,
        },
    ),
]


def _raw_table_to_df(raw_table):
    """Converte uma tabela bruta do pdfplumber (lista de listas) em DataFrame.

    - Usa a primeira linha não vazia como cabeçalho.
    - Preenche cabeçalhos vazios com nomes genéricos (Coluna_N).
    - Mantém células vazias como string vazia.
    """
    if not raw_table:
        return None

    # Normaliza células: None -> "", remove quebras de linha internas e espaços
    cleaned = []
    for row in raw_table:
        cleaned.append([
            (str(cell).replace("\n", " ").strip() if cell is not None else "")
            for cell in row
        ])

    # Descarta linhas totalmente vazias
    cleaned = [row for row in cleaned if any(c != "" for c in row)]
    if len(cleaned) < 1:
        return None

    header = cleaned[0]
    data = cleaned[1:]

    # Garante cabeçalhos não vazios e únicos
    seen = {}
    final_header = []
    for idx, h in enumerate(header):
        name = h if h else f"Coluna_{idx + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        final_header.append(name)

    if not data:
        # Tabela só com cabeçalho não tem utilidade
        return None

    # Ajusta larguras de linha divergentes ao número de colunas do cabeçalho
    n_cols = len(final_header)
    norm_data = []
    for row in data:
        if len(row) < n_cols:
            row = row + [""] * (n_cols - len(row))
        elif len(row) > n_cols:
            row = row[:n_cols]
        norm_data.append(row)

    return pd.DataFrame(norm_data, columns=final_header)


def extract_tables_with_pdfplumber(pdf_path, page_numbers):
    """Extrai tabelas usando pdfplumber (sem IA, sem chave de API).

    page_numbers: lista de índices 0-based de páginas a processar.
    Retorna lista de dicts no formato padrão do app.
    """
    results = []
    st.info("Executando extração rápida (pdfplumber)...")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            max_pages = len(pdf.pages)
            for page_idx in page_numbers:
                if page_idx >= max_pages:
                    continue
                page = pdf.pages[page_idx]

                page_had_table = False
                # Tenta cada estratégia; para na primeira que encontrar tabelas,
                # evitando duplicar a mesma tabela em 'grade' e 'texto'.
                for strat_name, settings in _PDFPLUMBER_STRATEGIES:
                    try:
                        raw_tables = page.extract_tables(table_settings=settings)
                    except Exception as e:
                        st.warning(
                            f"pdfplumber falhou na página {page_idx + 1} "
                            f"(estratégia {strat_name}): {e}"
                        )
                        continue

                    if not raw_tables:
                        continue

                    found_in_strat = 0
                    for t_idx, raw in enumerate(raw_tables):
                        df = _raw_table_to_df(raw)
                        if df is not None and not df.empty:
                            results.append({
                                'page': page_idx + 1,
                                'table_index_on_page': t_idx + 1,
                                'dataframe': df,
                                'source': f"pdfplumber ({strat_name})",
                            })
                            found_in_strat += 1

                    if found_in_strat > 0:
                        page_had_table = True
                        st.write(
                            f"  -> Página {page_idx + 1}: {found_in_strat} "
                            f"tabela(s) via estratégia '{strat_name}'."
                        )
                        break  # não tenta a próxima estratégia nesta página

                if not page_had_table:
                    st.write(f"  -> Página {page_idx + 1}: nenhuma tabela detectada.")
    except Exception as e:
        st.error(f"Erro ao abrir/processar o PDF com pdfplumber: {e}")

    return results


# ==========================================================================
# 2. EXTRAÇÃO POR IA (OPCIONAL) — Gemini VLM (novo SDK google-genai)
# ==========================================================================

def extract_image_from_pdf_page(pdf_path, page_num):
    """Renderiza uma imagem em alta resolução de uma página específica do PDF."""
    if not FITZ_AVAILABLE:
        st.error("PyMuPDF (fitz) não está instalado; necessário para o método Gemini.")
        return None
    doc = None
    try:
        doc = fitz.open(pdf_path)
        if page_num >= len(doc):
            st.warning(
                f"Número de página {page_num + 1} fora do intervalo "
                f"(Total: {len(doc)})."
            )
            return None
        page = doc[page_num]
        matrix = fitz.Matrix(GEMINI_IMAGE_RESOLUTION_FACTOR, GEMINI_IMAGE_RESOLUTION_FACTOR)
        pix = page.get_pixmap(matrix=matrix)
        return pix.tobytes("png")  # bytes PNG em memória, sem arquivo temporário
    except Exception as e:
        st.error(f"Erro ao renderizar imagem da página {page_num + 1}: {e}")
        return None
    finally:
        if doc:
            doc.close()


_GEMINI_PROMPT = """
Você é um especialista em extração de tabelas de documentos técnicos.
Analise a imagem fornecida (uma página de um documento) e identifique TODAS as
tabelas presentes.

Para CADA tabela, transcreva fielmente o conteúdo seguindo estas regras:
- Preserve a ordem das colunas e das linhas exatamente como na imagem.
- Mantenha números, unidades e símbolos exatamente como aparecem (ex: "16.000",
  "9 1/2", "P 7-7/8\\" REG", "588.64").
- Para CÉLULAS MESCLADAS (que ocupam várias linhas ou colunas), REPITA o valor
  em cada célula que a mesclagem cobre, para que cada linha fique completa.
- Para células vazias, use uma string vazia "".
- Inclua linhas de total/subtotal se existirem.
- Não invente dados que não estão visíveis.

Formate TODA a saída EXCLUSIVAMENTE como um único objeto JSON com a chave
principal 'tables'. O valor de 'tables' deve ser uma LISTA de objetos, cada um
representando uma tabela com DUAS chaves:
1. 'headers': lista de strings com os nomes das colunas.
2. 'data': lista de listas, onde cada lista interna é uma linha, com o mesmo
   número de elementos que 'headers'.

Exemplo:
{
  "tables": [
    {
      "headers": ["Item", "Descrição", "OD (in)"],
      "data": [
        ["1", "Broca PDC", "16.000"],
        ["2", "Estabilizador", "9.500"]
      ]
    }
  ]
}

Retorne APENAS o objeto JSON, sem texto adicional e sem marcadores como ```json.
Se nenhuma tabela for encontrada, retorne {"tables": []}.
"""


def detect_tables_with_gemini(pdf_path, page_numbers, api_key):
    """Detecta e transcreve tabelas usando o Gemini VLM (novo SDK google-genai)."""
    if not GEMINI_AVAILABLE:
        st.error("Biblioteca `google-genai` não está instalada. Execute `pip install google-genai`.")
        return []
    if not FITZ_AVAILABLE:
        st.error("PyMuPDF (fitz) não está instalado; necessário para renderizar páginas para o Gemini.")
        return []
    if not api_key:
        st.error("Chave API Gemini não fornecida na barra lateral.")
        return []

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Erro ao inicializar o cliente Gemini: {e}. Verifique sua chave API.")
        return []

    config = types.GenerateContentConfig(response_mime_type="application/json")

    all_detected_tables = []
    total = len(page_numbers)

    with st.spinner(f"Processando {total} página(s) com Gemini VLM ({GEMINI_MODEL_NAME})..."):
        for n, page_num in enumerate(page_numbers, start=1):
            st.info(f"Gemini VLM: processando página {page_num + 1} ({n}/{total})...")

            image_bytes = extract_image_from_pdf_page(pdf_path, page_num)
            if not image_bytes:
                st.warning(f"Pulando página {page_num + 1} (falha ao renderizar imagem).")
                continue

            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL_NAME,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                        _GEMINI_PROMPT,
                    ],
                    config=config,
                )
                response_text = response.text
            except Exception as e_api:
                st.error(
                    f"Erro na chamada à API Gemini (página {page_num + 1}): "
                    f"{type(e_api).__name__} - {e_api}"
                )
                continue

            try:
                detected = json.loads(response_text)
            except (json.JSONDecodeError, TypeError) as json_e:
                st.error(f"Erro ao decodificar JSON da página {page_num + 1}: {json_e}")
                st.text_area("Resposta bruta", response_text or "", height=150)
                continue

            page_tables = detected.get('tables', []) if isinstance(detected, dict) else []
            if not page_tables:
                st.info(f"Nenhuma tabela detectada pelo Gemini na página {page_num + 1}.")
                continue

            st.info(f"Gemini detectou {len(page_tables)} tabela(s) na página {page_num + 1}.")
            for i, table_data in enumerate(page_tables):
                if not (isinstance(table_data, dict) and
                        isinstance(table_data.get('headers'), list) and
                        isinstance(table_data.get('data'), list)):
                    st.warning(f"Estrutura inválida na Tabela {i + 1}/Página {page_num + 1}.")
                    continue

                headers = [str(h) for h in table_data['headers']]
                rows = table_data['data']
                if not all(isinstance(r, list) for r in rows):
                    st.warning(f"'data' não é lista de listas (Tabela {i + 1}/Pág {page_num + 1}).")
                    continue

                n_cols = len(headers)
                norm_rows = []
                for r in rows:
                    r = [str(c) for c in r]
                    if len(r) < n_cols:
                        r = r + [""] * (n_cols - len(r))
                    elif len(r) > n_cols:
                        r = r[:n_cols]
                    norm_rows.append(r)

                if not norm_rows:
                    continue

                df = pd.DataFrame(norm_rows, columns=headers)
                all_detected_tables.append({
                    'page': page_num + 1,
                    'table_index_on_page': i + 1,
                    'dataframe': df,
                    'source': 'Gemini',
                })

    return all_detected_tables


# ==========================================================================
# 3. LIMPEZA E PÓS-PROCESSAMENTO
# ==========================================================================

def parse_page_input(page_input_str, max_pages):
    """Converte a string de páginas do usuário em índices 0-based."""
    if not page_input_str or page_input_str.strip().lower() == 'all':
        return list(range(max_pages))
    page_numbers = set()
    for part in page_input_str.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            try:
                start, end = map(str.strip, part.split('-'))
                start_idx = int(start) - 1
                end_idx = int(end)
                if start_idx < 0 or end_idx > max_pages or start_idx >= end_idx:
                    st.warning(f"Intervalo inválido '{part}' (páginas de 1 a {max_pages}). Ignorando.")
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
                    st.warning(f"Página inválida '{part}' (de 1 a {max_pages}). Ignorando.")
            except ValueError:
                st.warning(f"Formato de página inválido '{part}'. Ignorando.")
    return sorted(page_numbers)


def get_pdf_page_count(pdf_path):
    """Conta páginas do PDF de forma leve."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception as e:
        st.error(f"Erro ao abrir o PDF: {e}")
        return 0


def fix_duplicate_columns(df):
    """Renomeia colunas duplicadas adicionando _1, _2, etc."""
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        dup_indices = cols[cols == dup].index.tolist()
        for i, idx in enumerate(dup_indices):
            cols[idx] = f"{dup}_{i}" if i > 0 else dup
    df.columns = cols
    return df


def clean_dataframe(df):
    """Limpa o DataFrame: remove espaços, linhas/colunas vazias, renomeia duplicatas."""
    if df.empty:
        return df
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='all')
    if df.empty:
        return df
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].str.strip()
    df = fix_duplicate_columns(df)
    df = df.reset_index(drop=True)
    return df


def process_extracted_tables(tables_list):
    """Limpa e prepara as tabelas extraídas para exibição e download."""
    processed = []
    if not tables_list:
        return processed
    for i, table_info in enumerate(tables_list):
        try:
            cleaned_df = clean_dataframe(table_info['dataframe'].copy())
            if not cleaned_df.empty:
                source_simple = re.sub(r'[\s()]', '', table_info['source'])
                table_id = f"{source_simple}_Pg{table_info['page']}_T{table_info['table_index_on_page']}"
                processed.append({
                    **table_info,
                    'dataframe': cleaned_df,
                    'table_id': table_id,
                })
            else:
                st.write(
                    f"Tabela {i + 1} ({table_info.get('source', '?')}, "
                    f"P{table_info.get('page', '?')}) descartada (vazia após limpeza)."
                )
        except Exception as e:
            st.error(f"Erro ao limpar Tabela {i + 1}: {e}")
    return processed


# ==========================================================================
# 4. INTERFACE STREAMLIT
# ==========================================================================

st.title("📊 Conversor de PDF para Excel - Extrator de Tabelas")
st.markdown(
    "Extraia tabelas de arquivos PDF e salve-as em Excel. O método principal "
    "(**pdfplumber**) é rápido, gratuito e **não precisa de chave de API**. "
    "Para tabelas complexas (células mescladas, sem bordas), use o **Gemini VLM** "
    "como reforço por IA."
)

with st.sidebar:
    st.header("⚙️ Configurações")
    extraction_method = st.radio(
        "Método de Extração:",
        ["Extração Rápida (pdfplumber)", "Gemini VLM (IA)", "Ambos"],
        index=0,  # padrão: método principal sem IA
        help=(
            "pdfplumber: rápido, sem API, ótimo para grades bem delimitadas. "
            "Gemini VLM: usa IA para ler a imagem da página, lida com tabelas "
            "complexas, mas requer Chave API. 'Ambos' executa os dois."
        ),
    )
    page_input = st.text_input(
        "Páginas a Processar:",
        placeholder="Todas (Ex: 1, 3, 5-7)",
        help="Deixe em branco para todas as páginas. Formatos: '1', '1,3', '1-3', '1,3-5'.",
    )
    st.markdown("---")
    st.subheader("🔑 Chave API Gemini")
    api_key_input = st.text_input(
        "Cole sua Chave API Gemini:",
        type="password",
        key="api_key_input_sidebar",
        help="Necessária apenas se 'Gemini VLM' ou 'Ambos' for selecionado.",
    )
    st.markdown("[Obtenha uma Chave API Gemini](https://aistudio.google.com/app/apikey)")

    needs_gemini = extraction_method in ["Gemini VLM (IA)", "Ambos"]
    if needs_gemini and not GEMINI_AVAILABLE:
        st.warning("Biblioteca `google-genai` não encontrada. Instale com `pip install google-genai`.")
    elif needs_gemini and not api_key_input:
        st.warning("Insira sua Chave API Gemini para usar este método.")

uploaded_file = st.file_uploader("1. Escolha um arquivo PDF", type=['pdf'])

if uploaded_file is not None:
    st.success(f"Arquivo '{uploaded_file.name}' carregado!")
    file_details = {
        "Nome": uploaded_file.name,
        "Tipo": uploaded_file.type,
        "Tamanho (KB)": f"{uploaded_file.size / 1024:.2f}",
    }
    st.write(file_details)

    use_pdfplumber = extraction_method in ["Extração Rápida (pdfplumber)", "Ambos"]
    use_gemini = extraction_method in ["Gemini VLM (IA)", "Ambos"]

    if use_gemini and not api_key_input:
        st.error("⚠️ Chave API Gemini é necessária na barra lateral para o método Gemini VLM.")
        st.stop()
    if use_gemini and not GEMINI_AVAILABLE:
        st.error("⚠️ A biblioteca `google-genai` não está instalada.")
        st.stop()

    temp_pdf_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.getvalue())
            temp_pdf_path = temp_pdf.name

        st.info("2. Iniciando extração...")
        progress_bar = st.progress(0, text="Iniciando...")
        status_text = st.empty()
        all_results = []

        max_pages = get_pdf_page_count(temp_pdf_path)
        if max_pages == 0:
            st.error("Não foi possível ler as páginas do PDF.")
            st.stop()

        page_numbers = parse_page_input(page_input, max_pages)
        if not page_numbers:
            st.warning("Nenhuma página válida selecionada para processamento.")
            st.stop()

        if use_pdfplumber:
            status_text.info("Executando extração rápida (pdfplumber)...")
            all_results.extend(extract_tables_with_pdfplumber(temp_pdf_path, page_numbers))
            progress_bar.progress(45, text="Extração rápida concluída.")

        if use_gemini:
            status_text.info("Executando Gemini VLM...")
            all_results.extend(detect_tables_with_gemini(temp_pdf_path, page_numbers, api_key_input))
            progress_bar.progress(80, text="Gemini VLM concluído.")

        status_text.info("Limpando e processando tabelas extraídas...")
        if not all_results:
            st.warning("Nenhuma tabela bruta foi extraída pelos métodos selecionados.")
            progress_bar.progress(100, text="Concluído - Nenhuma tabela encontrada.")
        else:
            st.success(f"Extração inicial concluída: {len(all_results)} tabela(s) bruta(s).")
            processed_tables = process_extracted_tables(all_results)
            progress_bar.progress(90, text="Processamento concluído.")

            if not processed_tables:
                st.warning("Nenhuma tabela permaneceu válida após a limpeza.")
                progress_bar.progress(100, text="Concluído - Nenhuma tabela válida.")
            else:
                st.success(f"Processamento finalizado! {len(processed_tables)} tabela(s) válida(s).")
                st.markdown("---")
                st.header("📊 Tabelas Extraídas e Processadas")

                output_excel = io.BytesIO()
                excel_filename = f"{os.path.splitext(uploaded_file.name)[0]}_todas_tabelas.xlsx"
                try:
                    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                        sheet_names_used = set()
                        for i, table_info in enumerate(processed_tables):
                            df = table_info['dataframe']
                            tid = table_info['table_id']

                            safe_sheet_name = re.sub(r'[\\/*?:\[\]]', '_', tid)[:31]
                            original_name = safe_sheet_name
                            count = 1
                            while safe_sheet_name in sheet_names_used:
                                suffix = f"_{count}"
                                safe_sheet_name = original_name[:31 - len(suffix)] + suffix
                                count += 1
                            sheet_names_used.add(safe_sheet_name)

                            df.to_excel(writer, index=False, sheet_name=safe_sheet_name)

                            expander_title = (
                                f"Tabela {i + 1}: {tid} (Origem: {table_info['source']}, "
                                f"Pág: {table_info['page']}) - {len(df)} linhas x {len(df.columns)} colunas"
                            )
                            with st.expander(expander_title):
                                st.dataframe(df, height=min(300, (len(df) + 1) * 35 + 3))

                                out_single = io.BytesIO()
                                with pd.ExcelWriter(out_single, engine='openpyxl') as swriter:
                                    df.to_excel(swriter, index=False, sheet_name=safe_sheet_name)
                                out_single.seek(0)
                                single_filename = (
                                    f"{os.path.splitext(uploaded_file.name)[0]}_{safe_sheet_name}.xlsx"
                                )
                                st.download_button(
                                    label="📥 Baixar Tabela Individual (.xlsx)",
                                    data=out_single,
                                    file_name=single_filename,
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key=f"dl_single_{i}",
                                )

                    output_excel.seek(0)
                    st.markdown("---")
                    st.download_button(
                        label=f"📥 Baixar TODAS as {len(processed_tables)} Tabelas (.xlsx)",
                        data=output_excel,
                        file_name=excel_filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dl_all",
                    )
                except Exception as e_excel:
                    st.error(f"Erro ao gerar o arquivo Excel: {e_excel}")

        progress_bar.progress(100, text="Processo concluído!")
        status_text.empty()

    except Exception as e:
        st.error("Ocorreu um erro inesperado durante o processamento:")
        st.exception(e)
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.unlink(temp_pdf_path)
            except Exception as e_clean:
                st.warning(f"Não foi possível remover o PDF temporário: {e_clean}")

st.markdown("---")
st.markdown("*Desenvolvido com Streamlit, pdfplumber, PyMuPDF e Google Gemini.*")
