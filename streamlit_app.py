import sqlite3
import ast
from typing import Dict, Tuple, List, Optional

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from streamlit_plotly_events import plotly_events

# =========================================================
# CONFIGURAÇÃO GERAL
# =========================================================
st.set_page_config(page_title="Isolamento Real IEEE 123 Bus", layout="wide")

st.sidebar.header("📂 Dados carregados")

DB_PATH = "ieee123_isolamento.db"   # banco na raiz do repo

# ---------------------------------------------------------
# FUNÇÕES DE ACESSO AO BANCO
# ---------------------------------------------------------
def get_connection():
    return sqlite3.connect(DB_PATH)


def table_exists(conn, name: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    )
    return cur.fetchone() is not None


def load_coords(conn) -> Dict[str, Tuple[float, float]]:
    cur = conn.cursor()
    cur.execute("SELECT bus, x, y FROM coords")
    return {row[0]: (row[1], row[2]) for row in cur.fetchall()}


def load_topology(conn) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT line, from_bus, to_bus, is_switch, norm FROM topology",
        conn,
    )


def load_vao_map(conn) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT u_bus, v_bus, nf, kw, n_barras FROM vao_map",
        conn,
    )


# =========================================================
# CARREGAMENTO DOS DADOS
# =========================================================
try:
    conn = get_connection()
except Exception as e:
    st.error(f"❌ Erro ao abrir o banco {DB_PATH}: {e}")
    st.stop()

with conn:
    has_coords = table_exists(conn, "coords")
    has_topology = table_exists(conn, "topology")
    has_vao_map = table_exists(conn, "vao_map")

    st.sidebar.write("**Banco:** ", f"`{DB_PATH}`")

    st.sidebar.write(
        "MASTER:",
        "✅" if has_topology else "❌",
    )
    st.sidebar.write(
        "COORDS:",
        "✅" if has_coords else "❌",
    )
    st.sidebar.write(
        "VAO_MAP:",
        "✅" if has_vao_map else "❌",
    )

    if not (has_coords and has_topology and has_vao_map):
        st.error(
            "Banco não possui alguma tabela necessária (`coords`, `topology`, `vao_map`)."
        )
        st.stop()

    coords = load_coords(conn)
    topo_df = load_topology(conn)
    vao_df = load_vao_map(conn)

# =========================================================
# EXPLICAÇÃO INICIAL
# =========================================================
st.title("⚡ Plataforma Interativa – Isolamento Real IEEE 123 Bus")

with st.expander("ℹ️ Sobre esta ferramenta", expanded=True):
    st.markdown(
        """
Ferramenta de apoio à manobra de **desligamento programado** em redes de distribuição,
baseada no alimentador teste **IEEE-123 Bus**.

A inteligência de isolamento (carga interrompida por NF e por vão U-V) foi calculada
anteriormente no **OpenDSS + Python (Colab)** e os resultados foram gravados no banco
SQLite `ieee123_isolamento.db`.

Este aplicativo usa **apenas** o banco + coordenadas de barras para exibir:

- ✅ Melhor chave **NF** de manobra para cada vão U-V  
- ⚡ Carga interrompida e número de barras isoladas  
- 🗺️ Mapa colorido da rede com destaque do vão e da NF  
- 🧾 “Linha do tempo” da manobra
"""
    )

# =========================================================
# CONSTRUÇÃO DO GRAFO
# =========================================================
G = nx.Graph()
for _, row in topo_df.iterrows():
    u = str(row["from_bus"])
    v = str(row["to_bus"])
    G.add_edge(
        u,
        v,
        line=str(row["line"]),
        is_switch=bool(row["is_switch"]),
        norm=str(row["norm"] or ""),
    )

all_buses = sorted(set(G.nodes()) & set(coords.keys()))

# =========================================================
# FUNÇÕES DE PLOT
# =========================================================
def build_figure(
    G: nx.Graph,
    coords: Dict[str, Tuple[float, float]],
    vao: Optional[Tuple[str, str]] = None,
    best_nf: Optional[str] = None,
) -> go.Figure:
    """
    Cria o gráfico do alimentador:
      - linhas normais: cinza
      - chaves NF: azul turquesa
      - NF de manobra escolhida: vermelho
      - vão U-V: laranja
    """
    # Categorias de arestas
    line_x, line_y = [], []
    nf_x, nf_y = [], []
    best_x, best_y = [], []
    vao_x, vao_y = [], []

    vao_set = set(vao) if vao and vao[0] and vao[1] else set()

    # descobrir a aresta do vão, se existir
    vao_edge = None
    if vao_set:
        u, v = vao
        for a, b, data in G.edges(data=True):
            if {a, b} == {u, v}:
                vao_edge = (a, b)
                break

    for u, v, data in G.edges(data=True):
        if u not in coords or v not in coords:
            continue
        x0, y0 = coords[u]
        x1, y1 = coords[v]

        if data.get("is_switch"):
            if best_nf and str(data.get("line")).lower() == best_nf.lower():
                best_x += [x0, x1, None]
                best_y += [y0, y1, None]
            else:
                nf_x += [x0, x1, None]
                nf_y += [y0, y1, None]
        else:
            line_x += [x0, x1, None]
            line_y += [y0, y1, None]

    # aresta do vão
    if vao_edge:
        u, v = vao_edge
        x0, y0 = coords[u]
        x1, y1 = coords[v]
        vao_x += [x0, x1, None]
        vao_y += [y0, y1, None]

    # nós
    node_x, node_y, node_text, node_color = [], [], [], []
    for n in G.nodes():
        if n not in coords:
            continue
        x, y = coords[n]
        node_x.append(x)
        node_y.append(y)
        node_text.append(n)

        if vao_set and n in vao_set:
            node_color.append("#FFA500")  # laranja – barras do vão
        else:
            node_color.append("#1f77b4")  # azul – barra normal

    fig = go.Figure()

    # Linhas normais
    if line_x:
        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                line=dict(color="#BBBBBB", width=1),
                hoverinfo="none",
                name="Linhas",
            )
        )

    # NF fechadas
    if nf_x:
        fig.add_trace(
            go.Scatter(
                x=nf_x,
                y=nf_y,
                mode="lines",
                line=dict(color="#00CED1", width=2),
                hoverinfo="none",
                name="Chaves NF",
            )
        )

    # NF de manobra
    if best_x:
        fig.add_trace(
            go.Scatter(
                x=best_x,
                y=best_y,
                mode="lines",
                line=dict(color="#FF4500", width=3),
                hoverinfo="none",
                name="NF de manobra",
            )
        )

    # Vão
    if vao_x:
        fig.add_trace(
            go.Scatter(
                x=vao_x,
                y=vao_y,
                mode="lines",
                line=dict(color="#FFA500", width=3, dash="dot"),
                hoverinfo="none",
                name="Vão U-V",
            )
        )

    # Nós
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            marker=dict(size=7, color=node_color),
            hovertemplate="<b>Barra:</b> %{text}<extra></extra>",
            name="Barras",
        )
    )

    fig.update_layout(
        height=650,
        showlegend=True,
        margin=dict(l=10, r=10, t=10, b=10),
        clickmode="event+select",
    )

    return fig


# =========================================================
# SELEÇÃO DO VÃO (CLIQUE + SIDEBAR)
# =========================================================
st.subheader("🗺️ Mapa Interativo da Rede")

if "bus_u" not in st.session_state:
    st.session_state.bus_u = ""
if "bus_v" not in st.session_state:
    st.session_state.bus_v = ""

# figura inicial (sem destaque)
base_fig = build_figure(
    G,
    coords,
    vao=(st.session_state.bus_u, st.session_state.bus_v)
    if st.session_state.bus_u and st.session_state.bus_v
    else None,
    best_nf=None,
)

# captura de cliques (usando streamlit-plotly-events)
events = plotly_events(
    base_fig,
    click_event=True,
    hover_event=False,
    select_event=False,
    key="graph",
)

# se o usuário clicou numa barra (scatter dos nós)
if events:
    ev = events[0]
    bus_clicked = ev.get("text")
    if bus_clicked:
        # vamos preenchendo U e V em sequência
        if not st.session_state.bus_u:
            st.session_state.bus_u = bus_clicked
        elif not st.session_state.bus_v:
            st.session_state.bus_v = bus_clicked
        else:
            # se já tem U e V, reinicia a seleção com o novo clique
            st.session_state.bus_u = bus_clicked
            st.session_state.bus_v = ""

st.sidebar.markdown("### 🔧 Selecione o vão")

bus_u = st.sidebar.selectbox(
    "Barra U",
    options=[""] + all_buses,
    index=([""] + all_buses).index(st.session_state.bus_u)
    if st.session_state.bus_u in all_buses
    else 0,
    key="bus_u_select",
)

bus_v = st.sidebar.selectbox(
    "Barra V",
    options=[""] + all_buses,
    index=([""] + all_buses).index(st.session_state.bus_v)
    if st.session_state.bus_v in all_buses
    else 0,
    key="bus_v_select",
)

# sincroniza estado
if bus_u:
    st.session_state.bus_u = bus_u
if bus_v:
    st.session_state.bus_v = bus_v

vao_ok = bool(st.session_state.bus_u and st.session_state.bus_v)

if vao_ok:
    u = st.session_state.bus_u
    v = st.session_state.bus_v

    st.markdown(
        f"#### 🔍 Vão selecionado: **{u} — {v}** (ordem não importa)"
    )

    # =====================================================
    # CONSULTA DAS OPÇÕES DE NF PARA ESSE VÃO
    # =====================================================
    with conn:
        df_vao = pd.read_sql_query(
            """
            SELECT u_bus, v_bus, nf, kw, n_barras
            FROM vao_map
            WHERE (u_bus = ? AND v_bus = ?)
               OR (u_bus = ? AND v_bus = ?)
            """,
            conn,
            params=(u, v, v, u),
        )

    if df_vao.empty:
        st.error("Não há registros de manobra para esse par de barras no banco.")
        best_nf = None
    else:
        # melhor NF: menor kW, depois menor nº de barras
        df_vao_sorted = df_vao.sort_values(["kw", "n_barras"])
        best_row = df_vao_sorted.iloc[0]
        best_nf = best_row["nf"]

        st.markdown("##### 🧮 Opções de NF para o vão")
        st.dataframe(
            df_vao_sorted.rename(
                columns={
                    "nf": "NF",
                    "kw": "kW interrompidos",
                    "n_barras": "Nº barras isoladas",
                }
            ),
            use_container_width=True,
        )

        # =================================================
        # MAPA COM DESTAQUE DO VÃO E DA NF
        # =================================================
        fig_vao = build_figure(
            G,
            coords,
            vao=(u, v),
            best_nf=best_nf,
        )
        st.plotly_chart(fig_vao, use_container_width=True)

        # =================================================
        # LINHA DO TEMPO DA MANOBRA
        # =================================================
        st.markdown("### 📜 Linha do tempo da manobra")

        st.markdown(
            f"""
1. **Identificação do vão de trabalho**  
   - Trecho entre as barras **{u}** e **{v}**.

2. **Análise prévia de desligamento (via banco de dados)**  
   - Para este vão, foram avaliadas todas as chaves **NF** disponíveis.  
   - A chave escolhida foi **{best_nf.upper()}**, por apresentar:  
     - Menor potência interrompida (**{best_row['kw']:.1f} kW**)  
     - Menor número de barras isoladas (**{int(best_row['n_barras'])} barras**).

3. **Sequência de manobra recomendada**  
   1. Confirmar condições de segurança e liberação do trecho {u}–{v}.  
   2. **Abrir a chave {best_nf.upper()}** (NF de manobra).  
   3. Verificar ausência de tensão no vão {u}–{v} e aplicar os procedimentos de bloqueio/etiquetagem.  
   4. Executar a **manutenção programada** no trecho.  
   5. Após conclusão, retirar bloqueios, inspecionar o trecho e **fechar novamente a chave {best_nf.upper()}**.  

4. **Restabelecimento**  
   - Normalização do esquema de manobra original do alimentador.  
   - Atualizar registros operacionais (ordem de serviço, diário de manobras, etc.).
"""
        )
else:
    st.info(
        "Selecione duas barras (U e V) pela barra lateral **ou clicando em duas barras no grafo** "
        "para analisar o melhor desligamento."
    )
