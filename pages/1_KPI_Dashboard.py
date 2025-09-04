# pages/1_KPI_Dashboard.py

import json
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

# ---------- Config ----------
st.set_page_config(page_title="KPI Vendas • Análise Inteligente", layout="wide")

# OpenAI somente via st.secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else None
try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    openai_client = None

# ---------- Teste de conexão (diagnóstico opcional) ----------
if st.sidebar.checkbox("🔌 Testar conexão NeonDB", value=False):
    try:
        conn = st.connection("neondb", type="sql")  # usa [connections.neondb] do secrets.toml
        with conn.session as s:
            v = s.execute(text("select version()")).fetchone()[0]
        st.success("Conectou! 🎉")
        st.caption(v)
    except Exception as e:
        st.error(f"Falha ao conectar: {e}")

# ---------- Conexão com o Banco de Dados (APENAS secrets.toml) ----------
@st.cache_data(ttl=600)
def load_from_db():
    """Lê a tabela 'vendas' usando st.connection('neondb'). Requer em secrets.toml:

    [connections.neondb]
    url = "postgresql://usuario:senha@host/neondb?sslmode=require&channel_binding=require"

    OPENAI_API_KEY = "sk-..."
    """
    try:
        conn = st.connection("neondb", type="sql")
        # cache também no nível da query (ttl)
        df = conn.query("SELECT * FROM vendas", ttl=600)
        return df
    except Exception as e:
        st.error(f"Não foi possível conectar ou carregar os dados do banco de dados: {e}")
        return pd.DataFrame()

# ---------- Helpers ----------
def normalize_cols(df):
    colmap = {}
    for c in df.columns:
        norm = (
            c.strip()
            .lower()
            .replace("ã", "a").replace("õ", "o").replace("ç", "c")
            .replace("á", "a").replace("é", "e").replace("í", "i")
            .replace("ó", "o").replace("ú", "u").replace("ê", "e")
            .replace("â", "a").replace("ô", "o").replace(" ", "_").replace("-", "_")
        )
        colmap[norm] = c
    return colmap

def to_numeric_safe(x):
    if x is None or x == "":
        return np.nan
    try:
        return float(x)
    except (ValueError, TypeError):
        s = str(x).strip().replace("R$", "").replace(" ", "")
        if "," in s and "." in s:
            s = s.replace(".", "").replace(",", ".")
        elif "," in s:
            s = s.replace(",", ".")
        try:
            return float(s)
        except (ValueError, TypeError):
            return np.nan

def coerce_date(s):
    if pd.isna(s) or s is None:
        return None
    try:
        return pd.to_datetime(s, errors="coerce").date()
    except Exception:
        return None

def top_n(df, group_cols, value_col, n=10):
    grp = df.groupby(group_cols, dropna=False, as_index=False)[value_col].sum()
    grp = grp.sort_values(value_col, ascending=False).head(n)
    return grp

# ---------- Lógica de Negócio e IA ----------
def calculate_momentum(df_filtered):
    df_d = df_filtered.dropna(subset=["_data", "_valor"]).copy()
    if df_d.empty or df_d["_data"].isnull().all():
        return pd.DataFrame()

    df_d["yyyymm"] = df_d["_data"].apply(lambda d: d.year * 100 + d.month)
    last_ym = int(df_d["yyyymm"].max())
    recent_range = [last_ym - i for i in range(0, 3)]
    prev_range = [last_ym - i for i in range(3, 6)]

    recent_sales = (
        df_d[df_d["yyyymm"].isin(recent_range)]
        .groupby(["_cidade", "_estado"], as_index=False)["_valor"]
        .sum()
    )
    recent_sales.rename(columns={"_valor": "vendas_recentes"}, inplace=True)

    prev_sales = (
        df_d[df_d["yyyymm"].isin(prev_range)]
        .groupby(["_cidade", "_estado"], as_index=False)["_valor"]
        .sum()
    )
    prev_sales.rename(columns={"_valor": "vendas_anteriores"}, inplace=True)

    if recent_sales.empty:
        return pd.DataFrame()

    mom = pd.merge(recent_sales, prev_sales, on=["_cidade", "_estado"], how="left")
    mom["vendas_anteriores"] = mom["vendas_anteriores"].fillna(0.0)
    mom["crescimento_abs"] = mom["vendas_recentes"] - mom["vendas_anteriores"]

    mom["crescimento_pct"] = np.where(
        mom["vendas_anteriores"] > 0,
        mom["crescimento_abs"] / mom["vendas_anteriores"],
        np.where(mom["vendas_recentes"] > 0, 2.0, 0.0),
    )
    mom["crescimento_pct"] = np.minimum(mom["crescimento_pct"], 1.0)

    return mom.sort_values(["crescimento_pct", "vendas_recentes"], ascending=[False, False])

def openai_investment_analysis(df_momentum, total_vendas, qtd_cidades, df_top_estados, df_top_cidades):
    if not openai_client or df_momentum.empty:
        return "Análise da IA não disponível. Verifique a chave da API OpenAI."

    top_10_invest = df_momentum.head(10).to_dict(orient="records")
    top_5_estados = df_top_estados.head(5).to_dict(orient="records")
    top_5_cidades = df_top_cidades.head(5).to_dict(orient="records")

    momentum_data = []
    for i, city in enumerate(top_10_invest, 1):
        crescimento = city["crescimento_pct"] * 100
        momentum_data.append(
            f"{i}. {city['_cidade']}/{city['_estado']}: "
            f"{crescimento:.1f}% de crescimento, "
            f"R$ {city['vendas_recentes']:,.2f} recentes"
        )

    estados_data = [f"{i}. {e['_estado']}: R$ {e['_valor']:,.2f}" for i, e in enumerate(top_5_estados, 1)]
    cidades_data = [f"{i}. {c['_cidade']} ({c['_estado']}): R$ {c['_valor']:,.2f}" for i, c in enumerate(top_5_cidades, 1)]

    prompt = f"""Com base nos dados abaixo, gere recomendações acionáveis e um plano de 90 dias.

**DADOS FINANCEIROS GERAIS**
- Volume total analisado: R$ {total_vendas:,.2f}
- Cidades atendidas: {qtd_cidades}

**TOP 5 ESTADOS POR VOLUME**
{chr(10).join(estados_data)}

**TOP 5 CIDADES POR VOLUME**
{chr(10).join(cidades_data)}

**TOP 10 CIDADES POR MOMENTUM (CRESCIMENTO)**
{chr(10).join(momentum_data)}
"""

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um analista estratégico sênior."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1500,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ **Erro ao gerar análise da IA:** {str(e)}"

# ---------- Início da Interface ----------
st.sidebar.title("⚙️ Configurações")
st.sidebar.info("Os dados são carregados automaticamente do banco de dados NeonDB.")

df_raw = load_from_db()
if df_raw.empty:
    st.title("KPI de Vendas")
    st.warning("Não foi possível carregar os dados.")
    st.stop()

# Mapeamento de colunas
st.sidebar.subheader("🔧 Mapeamento de Colunas")
colmap = normalize_cols(df_raw)

def pick_col(label, candidates):
    options_list = [None] + list(df_raw.columns)
    best_match = None
    for col in df_raw.columns:
        col_lower = col.lower()
        for candidate in candidates:
            if candidate in col_lower:
                best_match = col
                break
        if best_match:
            break
    index = options_list.index(best_match) if best_match else 0
    return st.sidebar.selectbox(label, options=options_list, index=index)

# Seleção de colunas
col_estado = pick_col("Coluna de Estado (UF)", ["estado", "uf", "sigla"])
col_cidade = pick_col("Coluna de Cidade", ["cidade", "municipio", "município"])
col_data   = pick_col("Coluna de Data", ["data", "emissao", "emissão", "pedido", "date"])
col_valor  = pick_col("Coluna de Valor (R$)", ["valor", "total", "receita", "venda", "preço"])

if not all([col_estado, col_cidade, col_data, col_valor]):
    st.error("⚠️ Mapeie as quatro colunas principais na barra lateral.")
    st.stop()

# ---------- Preparação dos dados ----------
df = df_raw.copy()
df["_estado"] = (
    df[col_estado].astype(str).str.strip().str.upper()
    .replace({"NAN": None, "NONE": None, "": None, "nan": None, "None": None})
)
df["_cidade"] = (
    df[col_cidade].astype(str).str.strip().str.title()
    .replace({"NAN": None, "NONE": None, "": None, "nan": None, "None": None})
)
df["_data"]  = df[col_data].apply(coerce_date)
df["_valor"] = df[col_valor].apply(to_numeric_safe)
df = df.dropna(subset=["_estado", "_cidade", "_valor"])

# Filtros
st.sidebar.subheader("🔍 Filtros")
ufs = sorted([x for x in df["_estado"].dropna().unique().tolist() if x and len(x) <= 3])
default_ufs = ufs.copy()
if "MG" in ufs and "MG" not in default_ufs:
    default_ufs.append("MG")
uf_sel = st.sidebar.multiselect("Estados (UF)", options=ufs, default=default_ufs)

anos = sorted(list({d.year for d in df["_data"].dropna() if d}))
ano_sel = st.sidebar.selectbox("Ano (opcional)", options=[None] + anos, index=0)

df_f = df.copy()
if uf_sel:
    df_f = df_f[df_f["_estado"].isin(uf_sel)]
if ano_sel:
    df_f = df_f[df_f["_data"].apply(lambda d: d and d.year == ano_sel)]

cidades_opts = sorted([x for x in df_f["_cidade"].dropna().unique().tolist() if x])
cidades_sel = st.sidebar.multiselect("Cidades (opcional)", options=cidades_opts)
if cidades_sel:
    df_f = df_f[df_f["_cidade"].isin(cidades_sel)]

# ---------- Título e KPIs Globais ----------
st.title("📊 KPI de Vendas e Análise Inteligente")
if df_f.empty:
    st.warning("Nenhum dado encontrado com os filtros selecionados.")
    st.stop()

total_valor = float(np.nansum(df_f["_valor"]))
qtd_reg     = int(len(df_f))
qtd_cidades = df_f["_cidade"].nunique()
qtd_estados = df_f["_estado"].nunique()

c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("💰 Total (R$)", f"R$ {total_valor:,.2f}")
with c2: st.metric("🧾 Registros", f"{qtd_reg:,}")
with c3: st.metric("🏙️ Cidades", f"{qtd_cidades:,}")
with c4: st.metric("🗺️ Estados", f"{qtd_estados:,}")

if "MG" in df_f["_estado"].values:
    mg_valor   = df_f[df_f["_estado"] == "MG"]["_valor"].sum()
    mg_cidades = df_f[df_f["_estado"] == "MG"]["_cidade"].nunique()
    st.info(f"📍 **Minas Gerais (MG):** R$ {mg_valor:,.2f} em {mg_cidades} cidades")
if "RJ" in df_f["_estado"].values:
    rj_valor   = df_f[df_f["_estado"] == "RJ"]["_valor"].sum()
    rj_cidades = df_f[df_f["_estado"] == "RJ"]["_cidade"].nunique()
    st.info(f"📍 **Rio de Janeiro (RJ):** R$ {rj_valor:,.2f} em {rj_cidades} cidades")

st.divider()

# ---------- Gráfico de Evolução Mensal ----------
st.subheader("📈 Evolução Mensal das Vendas")
df_evo = df_f.dropna(subset=["_data", "_valor"]).copy()
if not df_evo.empty:
    df_evo["Ano-Mês"] = df_evo["_data"].apply(lambda d: f"{d.year}-{d.month:02d}")
    serie = df_evo.groupby("Ano-Mês", as_index=False)["_valor"].sum().sort_values("Ano-Mês")
    fig_evo = px.line(
        serie, x="Ano-Mês", y="_valor", title="Evolução Mensal do Valor de Vendas (R$)",
        markers=True, labels={"_valor": "Valor (R$)", "Ano-Mês": "Período"},
    )
    fig_evo.update_traces(line=dict(width=3, color="#1f77b4"))
    fig_evo.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(size=12), xaxis=dict(tickangle=45))
    fig_evo.update_xaxes(title_text="Mês/Ano")
    fig_evo.update_yaxes(title_text="Valor (R$)", tickprefix="R$ ")
    st.plotly_chart(fig_evo, use_container_width=True)
else:
    st.warning("Não há dados suficientes para mostrar a evolução mensal.")

st.divider()

# ---------- Gráficos Principais ----------
st.subheader("🏆 Top 10 por Valor de Vendas")
top_estados = top_n(df_f, ["_estado"], "_valor", n=10)
top_cidades = top_n(df_f, ["_cidade", "_estado"], "_valor", n=10)

col1, col2 = st.columns(2)
with col1:
    st.subheader("🏙️ Top 10 Cidades")
    if not top_cidades.empty:
        top_cidades_display = top_cidades.sort_values("_valor", ascending=True)
        top_cidades_display["label_completa"] = (
            top_cidades_display["_cidade"] + " (" + top_cidades_display["_estado"] + ")"
        )
        fig_cidades = go.Figure()
        colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
        for i, (_idx, row) in enumerate(top_cidades_display.iterrows()):
            fig_cidades.add_trace(go.Bar(
                y=[row["label_completa"]], x=[row["_valor"]], orientation="h",
                name=row["_cidade"], marker_color=colors[i % len(colors)],
                text=[f"R$ {row['_valor']:,.0f}"], textposition="auto",
                hovertemplate=f"<b>{row['_cidade']} ({row['_estado']})</b><br>Valor: R$ {row['_valor']:,.2f}<extra></extra>",
            ))
        fig_cidades.update_layout(height=500, showlegend=False,
                                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                  font=dict(size=12), margin=dict(l=20,r=20,t=50,b=20),
                                  xaxis=dict(title="Valor Total (R$)", tickprefix="R$ ", gridcolor="lightgray", zerolinecolor="lightgray"),
                                  yaxis=dict(title="", tickfont=dict(size=11)))
        st.plotly_chart(fig_cidades, use_container_width=True)
        st.markdown("**📋 Detalhes das Top 10 Cidades:**")
        resumo_cidades = top_cidades[["_cidade","_estado","_valor"]].copy()
        resumo_cidades.columns = ["Cidade","Estado","Valor Total (R$)"]
        resumo_cidades = resumo_cidades.sort_values("Valor Total (R$)", ascending=False)
        resumo_cidades["Valor Total (R$)"] = resumo_cidades["Valor Total (R$)"].apply(lambda x: f"R$ {x:,.2f}")
        st.dataframe(resumo_cidades, use_container_width=True, hide_index=True)

with col2:
    st.subheader("🌎 Top 10 Estados")
    if not top_estados.empty:
        top_estados_display = top_estados.sort_values("_valor", ascending=True)
        top_estados_display["percentual"] = (top_estados_display["_valor"] / total_valor * 100).round(1)
        fig_estados = go.Figure()
        colors = ["#ff9999","#66b3ff","#99ff99","#ffcc99","#c2c2f0","#ffb3e6","#c4e17f","#76d7ea","#ff9f80","#fec8c8"]
        for i, (_idx, row) in enumerate(top_estados_display.iterrows()):
            fig_estados.add_trace(go.Bar(
                y=[row["_estado"]], x=[row["_valor"]], orientation="h",
                name=row["_estado"], marker_color=colors[i % len(colors)],
                text=[f"R$ {row['_valor']:,.0f}"], textposition="auto",
                hovertemplate=f"<b>{row['_estado']}</b><br>Valor: R$ {row['_valor']:,.2f}<br>Participação: {row['percentual']}%<extra></extra>",
            ))
        fig_estados.update_layout(height=500, showlegend=False,
                                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                  font=dict(size=12), margin=dict(l=20,r=20,t=50,b=20),
                                  xaxis=dict(title="Valor Total (R$)", tickprefix="R$ ", gridcolor="lightgray", zerolinecolor="lightgray"),
                                  yaxis=dict(title="", tickfont=dict(size=11)))
        st.plotly_chart(fig_estados, use_container_width=True)
        resumo_estados = top_estados.copy()
        resumo_estados["percentual"] = (resumo_estados["_valor"] / total_valor * 100).round(1)
        resumo_estados = resumo_estados[["_estado","_valor","percentual"]]
        resumo_estados.columns = ["Estado","Valor Total (R$)","Participação (%)"]
        resumo_estados = resumo_estados.sort_values("Valor Total (R$)", ascending=False)
        resumo_estados["Valor Total (R$)"] = resumo_estados["Valor Total (R$)"].apply(lambda x: f"R$ {x:,.2f}")
        resumo_estados["Participação (%)"] = resumo_estados["Participação (%)"].apply(lambda x: f"{x}%")
        st.dataframe(resumo_estados, use_container_width=True, hide_index=True)

st.divider()

# ---------- Análise de Oportunidades com IA ----------
st.subheader("🚀 Análise de Oportunidades com IA")
with st.spinner("Calculando momentum e gerando análise da IA..."):
    df_momentum = calculate_momentum(df_f)
    if df_momentum.empty:
        st.warning("Não há dados suficientes para calcular o crescimento recente das cidades.")
    else:
        st.subheader("📊 Top 10 Cidades para Investir (Baseado em Crescimento)")
        df_plot = df_momentum.head(10).copy()
        df_plot["crescimento_pct_display"] = df_plot["crescimento_pct"] * 100
        df_plot["cidade_estado"] = df_plot["_cidade"] + "/" + df_plot["_estado"]
        df_plot = df_plot.sort_values("crescimento_pct_display", ascending=True)
        fig_crescimento = go.Figure()
        colors = ["#ff6b6b","#4ecdc4","#45b7d1","#f9ca24","#6c5ce7","#a29bfe","#fd79a8","#e17055","#00b894","#00cec9"]
        for i, (_idx, row) in enumerate(df_plot.iterrows()):
            fig_crescimento.add_trace(go.Bar(
                y=[row["cidade_estado"]], x=[row["crescimento_pct_display"]],
                orientation="h", name=row["_cidade"], marker_color=colors[i % len(colors)],
                text=[f"{row['crescimento_pct_display']:.1f}%"], textposition="auto",
                hovertemplate=f"<b>{row['_cidade']} ({row['_estado']})</b><br>Crescimento: {row['crescimento_pct_display']:.1f}%<br>Vendas Recentes: R$ {row['vendas_recentes']:,.2f}<extra></extra>",
            ))
        fig_crescimento.update_layout(height=500, showlegend=False,
                                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                      font=dict(size=12), margin=dict(l=20,r=20,t=50,b=20),
                                      xaxis=dict(title="Crescimento (%)", ticksuffix="%", gridcolor="lightgray",
                                                 zerolinecolor="lightgray", range=[0, 100]),
                                      yaxis=dict(title="", tickfont=dict(size=11)))
        st.plotly_chart(fig_crescimento, use_container_width=True)
        st.info("📌 Crescimento dos últimos 3 meses vs. 3 meses anteriores (limitado a 100%)")
        st.subheader("🤖 Análise Estratégica da IA")
        analysis = openai_investment_analysis(df_momentum, total_valor, qtd_cidades, top_estados, top_cidades)
        st.markdown(analysis)

st.divider()

# ---------- Tabela das 200 cidades de MG ----------
st.subheader("🏅 Top 200 Cidades de Minas Gerais (MG)")
df_mg = df_f[df_f["_estado"] == "MG"].copy()
if df_mg.empty:
    st.warning("Não há dados de Minas Gerais para o filtro selecionado.")
else:
    top_mg = (
        df_mg.groupby("_cidade", as_index=False)["_valor"].sum()
        .rename(columns={"_cidade": "Cidade", "_valor": "Valor (R$)"})
        .sort_values("Valor (R$)", ascending=False).head(200)
    )
    total_cidades_mg = df_mg["_cidade"].nunique()
    total_mg = top_mg["Valor (R$)"].sum()
    st.info(f"📍 Mostrando {len(top_mg)} de {total_cidades_mg} cidades de MG - Total: R$ {total_mg:,.2f}")
    top_mg_display = top_mg.copy()
    top_mg_display["Ranking"] = range(1, len(top_mg_display) + 1)
    top_mg_display["Participação (%)"] = (top_mg_display["Valor (R$)"] / total_mg * 100).round(2)
    st.dataframe(
        top_mg_display[["Ranking","Cidade","Valor (R$)","Participação (%)"]],
        use_container_width=True, height=600,
        column_config={
            "Ranking": st.column_config.NumberColumn(width="small"),
            "Cidade": st.column_config.TextColumn(width="medium"),
            "Valor (R$)": st.column_config.NumberColumn(format="R$ %.2f"),
            "Participação (%)": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )

st.divider()

# ---------- Tabela das vendas do RJ ----------
st.subheader("🌊 Vendas do Rio de Janeiro (RJ) - Registros Individuais")
df_rj = df_f[df_f["_estado"] == "RJ"].copy()
if df_rj.empty:
    st.warning("Não há dados do Rio de Janeiro para o filtro selecionado.")
else:
    total_rj = df_rj["_valor"].sum()
    total_registros_rj = len(df_rj)
    st.info(f"📍 {total_registros_rj} registros de vendas no RJ - Total: R$ {total_rj:,.2f}")
    cidades_rj = sorted(df_rj["_cidade"].unique().tolist())
    cidade_rj_sel = st.selectbox("Filtrar por cidade do RJ:", options=["Todas"] + cidades_rj, index=0)
    df_rj_filtrado = df_rj if cidade_rj_sel == "Todas" else df_rj[df_rj["_cidade"] == cidade_rj_sel].copy()
    rj_display = df_rj_filtrado[["_data","_cidade","_valor"]].copy()
    rj_display.columns = ["Data","Cidade","Valor (R$)"]
    rj_display = rj_display.sort_values(["Cidade","Data"], ascending=[True, False])
    rj_display["Data"] = rj_display["Data"].astype(str)
    rj_display["Valor (R$)"] = rj_display["Valor (R$)"].apply(lambda x: f"R$ {x:,.2f}")
    rj_display.reset_index(drop=True, inplace=True)
    rj_display.index = rj_display.index + 1
    rj_display.index.name = "Registro"
    st.dataframe(
        rj_display, use_container_width=True, height=400,
        column_config={
            "Registro": st.column_config.NumberColumn(width="small"),
            "Data": st.column_config.TextColumn(width="medium"),
            "Cidade": st.column_config.TextColumn(width="medium"),
            "Valor (R$)": st.column_config.TextColumn(width="medium"),
        },
    )
    if cidade_rj_sel != "Todas":
        st.write(f"**Filtrado por:** {cidade_rj_sel}")
        st.write(f"**Registros encontrados:** {len(df_rj_filtrado)}")
        st.write(f"**Total filtrado:** R$ {df_rj_filtrado['_valor'].sum():,.2f}")
