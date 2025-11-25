import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import opendssdirect as dss
import pandas as pd
import sqlite3
import os
import re

# =========================================================
# CONFIGURAÇÃO STREAMLIT
# =========================================================
st.set_page_config(page_title="Isolamento IEEE-123", layout="wide")
st.title("⚡ Plataforma Interativa – Isolamento Real IEEE-123")

st.sidebar.header("⚙️ Configuração / Seleção")

# =========================================================
# CAMINHOS (RELATIVOS AO REPO NO STREAMLIT)
# =========================================================
BASE_123BUS = "123Bus/"
MASTER = os.path.join(BASE_123BUS, "IEEE123Master.dss")
COORDS = os.path.join(BASE_123BUS, "BusCoords.dat")
DB_PATH = "ieee123_isolamento.db"   # banco gerado no Colab

TABELA_ISOLAMENTOS = "isolamentos"  # ajuste se o nome for diferente

# =========================================================
# FUNÇÕES DE SUPORTE – DSS / TOPOLOGIA
# =========================================================
def normalize(bus: str) -> str:
    return bus.split(".")[0] if bus else ""

@st.cache_resource
def init_dss():
    """
    Compila o modelo IEEE-123 no OpenDSSDirect apenas para:
    - construir o grafo
    - obter as barras de cada linha/chave
    """
    if not os.path.exists(MASTER):
        raise FileNotFoundError(f"Arquivo MASTER não encontrado: {MASTER}")
    dss.Text.Command(f'compile "{MASTER}"')
    dss.Solution.Solve()
    return True

def load_coordinates():
    coords = {}
    if not os.path.exists(COORDS):
        st.sidebar.error(f"❌ Arquivo de coordenadas não encontrado: {COORDS}")
        return coords
    with open(COORDS, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.split()
            if len(p) >= 3:
                try:
                    bus = p[0]
                    x = float(p[1])
                    y = float(p[2])
                    coords[bus] = (x, y)
                except Exception:
                    pass
    return coords

def build_graph_and_topology():
    """
    Monta:
      - grafo G com todas as linhas fechadas (estado nominal)
      - dict line2buses[name.lower()] = (bus1, bus2)
    """
    init_dss()

    G = nx.Graph()
    line2buses = {}

    for name in dss.Lines.AllNames():
        dss.Lines.Name(name)
        b1 = normalize(dss.Lines.Bus1())
        b2 = normalize(dss.Lines.Bus2())
        is_sw = name.lower().startswith("sw")

        # estado nominal: NFs fechadas, NAs abertas (sw7, sw8)
        closed = True
        if is_sw and name.lower() in ("sw7", "sw8"):
            closed = False

        line2buses[name.lower()] = (b1, b2)

        if closed:
            G.add_edge(b1, b2, element=name, is_switch=is_sw)

    return G, line2buses

# =========================================================
# FUNÇÕES DE SUPORTE – BANCO DE DADOS
# =========================================================
@st.cache_data
def load_db():
    """
    Carrega a tabela de isolamentos do banco sqlite.
    Espera colunas: linha, sw_nf, kw, n_barras, (opcional barras_isoladas)
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Banco de dados não encontrado: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {TABELA_ISOLAMENTOS}", conn)
    finally:
        conn.close()

    # Normaliza nomes de colunas em minúsculo
    df.columns = [c.lower() for c in df.columns]

    esperadas = {"linha", "sw_nf", "kw", "n_barras"}
    faltando = esperadas - set(df.columns)
    if faltando:
        raise RuntimeError(
            f"Tabela '{TABELA_ISOLAMENTOS}' não tem as colunas esperadas: "
            f"faltando: {faltando}. Colunas existentes: {list(df.columns)}"
        )

    return df

# =========================================================
# CARREGAMENTO DE DADOS
# =========================================================
# 1) Banco de dados
db_ok = True
try:
    df_iso = load_db()
except Exception as e:
    db_ok = False
    st.sidebar.error(f"Erro ao carregar banco: {e}")

# 2) Modelo DSS + coordenadas
topo_ok = True
try:
    G, line2buses = build_graph_and_topology()
    coords = load_coordinates()
    st.sidebar.success("Modelo IEEE-123 carregado para visualização ✔")
except Exception as e:
    topo_ok = False
    st.sidebar.error(f"Erro ao carregar modelo/coords: {e}")
    coords = {}

# Se qualquer uma das partes falhou, não continua
if not (db_ok and topo_ok):
    st.stop()

# =========================================================
# SIDEBAR – SELEÇÃO DO VÃO (LINHA) E VISUALIZAÇÃO
# =========================================================
st.sidebar.markdown("### 🔧 Selecione o vão (linha DSS)")

# lista de vãos (linhas) disponíveis no banco
linhas_disponiveis = sorted(df_iso["linha"].unique())
linha_sel = st.sidebar.selectbox(
    "Linha / Vão para manutenção:",
    options=linhas_disponiveis,
    index=0 if linhas_disponiveis else None,
)

# filtra todas as NFs testadas para esse vão
df_v = df_iso[df_iso["linha"] == linha_sel].copy()

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Info do banco")
st.sidebar.write(f"Vãos cadastrados: **{len(linhas_disponiveis)}**")
st.sidebar.write(f"Registros totais: **{len(df_iso)}**")

# =========================================================
# CÁLCULO DA NF ÓTIMA PARA O VÃO
# =========================================================
if df_v.empty:
    st.error("Não há registros no banco para este vão.")
    st.stop()

# ordena por menor kW e, em seguida, menor nº de barras isoladas
df_v_sorted = df_v.sort_values(["kw", "n_barras"])
melhor = df_v_sorted.iloc[0]

nf_otima = melhor["sw_nf"]
kw_otima = melhor["kw"]
nb_otima = melhor["n_barras"]

st.subheader(f"🔍 Vão selecionado: **{linha_sel}**")
st.markdown(
    f"""
**NF de manobra ótima (pelo banco):** `{nf_otima}`  
- ⚡ Potência interrompida: **{kw_otima:.1f} kW**  
- 🔻 Barras isoladas: **{int(nb_otima)}**  
"""
)

# mostra tabela completa de opções para o vão
st.markdown("### 📊 Opções de desligamento para este vão (do banco)")
st.dataframe(
    df_v_sorted[["sw_nf", "kw", "n_barras"]],
    use_container_width=True,
)

# =========================================================
# PLOT – TOPOLOGIA COLORIDA
# =========================================================
st.markdown("### 🗺️ Mapa da Rede com Vão e NF Destacados")

# Buses do vão (linha) selecionado (se existir na topologia)
linha_key = linha_sel.lower()
vao_bus1, vao_bus2 = line2buses.get(linha_key, (None, None))

# Buses da NF ótima
nf_key = str(nf_otima).lower()
nf_bus1, nf_bus2 = line2buses.get(nf_key, (None, None))

# separa arestas por tipo de destaque
edges_normal = []
edges_vao = []
edges_nf = []

for u, v, data in G.edges(data=True):
    elem = data.get("element", "")
    elem_lower = str(elem).lower()

    # categorização
    if elem_lower == linha_key:
        edges_vao.append((u, v))
    elif elem_lower == nf_key:
        edges_nf.append((u, v))
    else:
        edges_normal.append((u, v))

# monta vetores para plotly
def edge_segments(edge_list):
    xs, ys = [], []
    for u, v in edge_list:
        if u in coords and v in coords:
            x0, y0 = coords[u]
            x1, y1 = coords[v]
            xs += [x0, x1, None]
            ys += [y0, y1, None]
    return xs, ys

edge_x_norm, edge_y_norm = edge_segments(edges_normal)
edge_x_vao, edge_y_vao = edge_segments(edges_vao)
edge_x_nf, edge_y_nf = edge_segments(edges_nf)

# nós
node_x, node_y, node_text, node_color = [], [], [], []

for n in G.nodes():
    if n not in coords:
        continue
    x, y = coords[n]
    node_x.append(x)
    node_y.append(y)
    node_text.append(n)

    # cores dos nós
    if (n == vao_bus1) or (n == vao_bus2):
        node_color.append("#FFA500")  # laranja – vão
    elif (n == nf_bus1) or (n == nf_bus2):
        node_color.append("#FF4500")  # vermelho NF
    else:
        node_color.append("#1f77b4")  # azul padrão

fig = go.Figure()

# linhas normais
fig.add_trace(go.Scatter(
    x=edge_x_norm, y=edge_y_norm,
    mode="lines",
    line=dict(color="#B0B0B0", width=1),
    hoverinfo="none",
    name="Linhas ativas"
))

# vão selecionado
if edge_x_vao:
    fig.add_trace(go.Scatter(
        x=edge_x_vao, y=edge_y_vao,
        mode="lines",
        line=dict(color="#FFA500", width=3),
        hoverinfo="none",
        name=f"Vão {linha_sel}"
    ))

# NF ótima
if edge_x_nf:
    fig.add_trace(go.Scatter(
        x=edge_x_nf, y=edge_y_nf,
        mode="lines",
        line=dict(color="#FF4500", width=3, dash="dash"),
        hoverinfo="none",
        name=f"NF ótima {nf_otima}"
    ))

# nós
fig.add_trace(go.Scatter(
    x=node_x, y=node_y,
    mode="markers+text",
    text=node_text,
    textposition="top center",
    marker=dict(size=7, color=node_color, line=dict(width=0.5, color="black")),
    hovertemplate="<b>Barra:</b> %{text}<extra></extra>",
    name="Barras"
))

fig.update_layout(
    height=650,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=0, r=0, t=40, b=0),
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
)

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TIMELINE SIMPLES DA MANOBRA
# =========================================================
st.markdown("### ⏱️ Timeline da Manobra (conceitual)")

timeline = [
    "Estado inicial: todas as chaves NF fechadas, NAs abertas (sw7, sw8).",
    f"Identificado vão **{linha_sel}** no banco e calculadas as combinações de desligamento.",
    f"Selecionada NF ótima **{nf_otima}**, que isola as duas barras do vão com **mínimo kW interrompido ({kw_otima:.1f} kW)** "
    f"e **{int(nb_otima)} barras isoladas**.",
    f"Sequência sugerida:\n"
    f"  1️⃣ Confirmar condições de segurança no vão {linha_sel} (entre barras {vao_bus1} e {vao_bus2}).\n"
    f"  2️⃣ Abrir **{nf_otima}** (manobra de desligamento principal).\n"
    f"  3️⃣ Verificar tensões nas barras a jusante e validar ausência de energia no vão.\n"
    f"  4️⃣ Executar manutenção programada no vão {linha_sel}.\n"
    f"  5️⃣ Após manutenção, recompor a configuração original (fechar {nf_otima} novamente, se aplicável)."
]

for step in timeline:
    st.markdown(f"- {step}")
