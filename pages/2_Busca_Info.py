import io
import base64
from datetime import date, datetime
from pathlib import Path
import re
import streamlit as st
import pandas as pd
from pypdf import PdfReader

# ------------------------------
# ⚙️ CONFIG
# ------------------------------
st.set_page_config(page_title="Notas Fiscais (PDF)", page_icon="🧾", layout="wide")

st.title("🧾 Notas Fiscais – Busca Inteligente em PDFs")
st.caption("A página varre automaticamente os PDFs em **data/notas_fiscais** e permite filtrar por **UF** e **data de emissão**.")

# ===== Helpers
UF_LIST = [
    'AC','AL','AP','AM','BA','CE','DF','ES','GO','MA','MT','MS','MG','PA','PB','PR','PE','PI','RJ','RN','RS','RO','RR','SC','SP','SE','TO'
]

# Datas (várias variações encontradas nos modelos de NFS-e)
EMITIDA_PATTERNS = [
    re.compile(r"Emitid[ao]s?\s*em\s*:?\s*(\d{2}/\d{2}/\d{4})", re.IGNORECASE),
    re.compile(r"Emiss[aã]o\s*em\s*:?\s*(\d{2}/\d{2}/\d{4})", re.IGNORECASE),
    re.compile(r"Data\s*de\s*emiss[aã]o\s*:?\s*(\d{2}/\d{2}/\d{4})", re.IGNORECASE),
]

UF_TOKEN_PATTERN = re.compile(r"\b(" + "|".join(UF_LIST) + r")\b")
CITY_UF_PATTERN = re.compile(r"\b([A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ú'\-\.\s]+?)\s*[-,]\s*(" + "|".join(UF_LIST) + r")\b")
EMAIL_UF_PATTERN = re.compile(r"\.(AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO)\.gov\.br", re.IGNORECASE)


def embed_pdf(file_bytes: bytes, height: int = 600):
    """Exibe um PDF inline usando base64 em um <iframe>."""
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="{height}" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def extract_pdf_info(file_bytes: bytes, filename: str) -> dict:
    """Extrai metadados de NFS-e com heurísticas ("IA-like") para identificar UF de forma robusta.

    Sinais considerados para UF (com pontuações):
    - Cidade + UF ("Diamantina - MG" / "Belo Horizonte, MG"): +5
    - Domínios de e-mail *.xx.gov.br: +4
    - Ocorrências soltas do token de UF: +1
    - Sufixo no nome do arquivo (ex.: *_MG.pdf): +2
    """
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text_parts = []
        for page in reader.pages:
            try:
                text_parts.append(page.extract_text() or "")
            except Exception:
                pass
        full_text = "\n".join(text_parts)
    except Exception:
        return {"filename": filename, "emitida_em": None, "uf": None, "raw_text": ""}

    # ---- Data de emissão
    emitida_dt = None
    for pat in EMITIDA_PATTERNS:
        m = pat.search(full_text)
        if m:
            try:
                emitida_dt = datetime.strptime(m.group(1), "%d/%m/%Y").date()
                break
            except Exception:
                pass

    # ---- UF (heurísticas com pontuação)
    scores = {uf: 0 for uf in UF_LIST}

    # 1) Cidade - UF / Cidade, UF
    for _city, uf in CITY_UF_PATTERN.findall(full_text):
        scores[uf] += 5

    # 2) E-mails com .xx.gov.br
    for uf in EMAIL_UF_PATTERN.findall(full_text):
        uf = uf.upper()
        if uf in scores:
            scores[uf] += 4

    # 3) Tokens soltos de UF (mas com peso menor)
    for m in UF_TOKEN_PATTERN.findall(full_text):
        scores[m] += 1

    # 4) Sufixo no nome do arquivo
    mfile = UF_TOKEN_PATTERN.findall(Path(filename).stem.upper())
    for uf in mfile:
        scores[uf] += 2

    uf_value = None
    best = max(scores.values()) if scores else 0
    if best > 0:
        top = [k for k, v in scores.items() if v == best]
        uf_value = sorted(top)[0]  # desempate estável

    return {
        "filename": filename,
        "emitida_em": emitida_dt,
        "uf": uf_value,
        "raw_text": full_text
    }


# ------------------------------
# 📂 Varredura automática de PDFs
# ------------------------------
ROOT_DIR = Path("PDFS").absolute()
ROOT_DIR.mkdir(parents=True, exist_ok=True)

st.info(f"Lendo PDFs automaticamente de: **{ROOT_DIR}**. Coloque seus arquivos lá.")

pdf_items: list[tuple[str, bytes]] = []
for p in ROOT_DIR.rglob("*.pdf"):
    try:
        pdf_items.append((p.name, p.read_bytes()))
    except Exception:
        pass

if not pdf_items:
    st.warning("Nenhum PDF encontrado na pasta. Adicione arquivos em 'data/notas_fiscais' e recarregue a página.")
    st.stop()

# ------------------------------
# 🧪 Extração
# ------------------------------
records = [extract_pdf_info(b, name) for name, b in pdf_items]

df = pd.DataFrame(records)
if "emitida_em" in df.columns:
    df["emitida_em"] = pd.to_datetime(df["emitida_em"]).dt.date

# ------------------------------
# 🧰 Filtros (com todas as UFs, pesquisável)
# ------------------------------
with st.sidebar:
    st.header("Filtros")

    ufs_available = UF_LIST
    uf_default = sorted([u for u in df["uf"].dropna().unique()]) or UF_LIST
    uf_selected = st.multiselect(
        "UF (pesquisável)",
        options=ufs_available,
        default=uf_default,
        placeholder="Digite a UF (ex.: MG, SP, CE)",
        help="Você pode selecionar qualquer UF do Brasil, mesmo que ainda não apareça nos PDFs carregados."
    )

    if df["emitida_em"].notna().any():
        min_d = df["emitida_em"].dropna().min()
        max_d = df["emitida_em"].dropna().max()
    else:
        min_d = date.today()
        max_d = date.today()

    date_range = st.date_input(
        "Emissão – intervalo",
        value=(min_d, max_d) if min_d and max_d else None,
        min_value=min_d,
        max_value=max_d,
        help="Filtra pela data que aparece como ‘Emitida/Emissão em’."
    )

    page_size = st.number_input("Itens por página", 1, 50, 5)


mask = pd.Series(True, index=df.index)
if uf_selected:
    mask &= df["uf"].isin(uf_selected)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
    if start_d and end_d:
        mask &= df["emitida_em"].between(start_d, end_d, inclusive="both")

filtered = df[mask].sort_values(["emitida_em", "filename"], ascending=[False, True])

# ------------------------------
# 📊 Resultado + Export
# ------------------------------
st.subheader("Resultado filtrado")
st.dataframe(
    filtered[["filename", "uf", "emitida_em"]],
    use_container_width=True,
    hide_index=True
)

csv = filtered.to_csv(index=False).encode("utf-8-sig")
st.download_button("⬇️ Baixar CSV filtrado", data=csv, file_name="notas_filtradas.csv", mime="text/csv")

# ------------------------------
# 📄 Visualização dos PDFs
# ------------------------------
st.subheader("Visualizar PDFs")

total = len(filtered)
if total == 0:
    st.warning("Nenhum PDF corresponde aos filtros.")
    st.stop()

pages = (total - 1) // page_size + 1
page = st.number_input("Página", 1, pages, 1)
start = (page - 1) * page_size
end = min(start + page_size, total)

subset = filtered.iloc[start:end]

for _, row in subset.iterrows():
    fname = row["filename"]
    st.markdown(f"### {fname}")
    col1, col2, _ = st.columns([2, 2, 1])
    with col1:
        st.metric("UF (inferida)", row.get("uf") or "—")
    with col2:
        d = row.get("emitida_em")
        st.metric("Emitida em", d.strftime("%d/%m/%Y") if isinstance(d, date) else "—")

    for name, b in pdf_items:
        if name == fname:
            with st.expander("Abrir PDF", expanded=False):
                embed_pdf(b, height=650)
            break

st.caption("Se notar PDFs sem UF ou sem data, ajustamos as heurísticas. Modelos de NFS-e variam entre prefeituras.")
