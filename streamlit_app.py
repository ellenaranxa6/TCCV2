import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sqlite3
from pathlib import Path

# =========================================================
#            CONFIGURAÇÃO BÁSICA DO APP
# =========================================================
st.set_page_config(page_title="Isolamento Real IEEE 123 Bus",
                   layout="wide")

st.title("⚡ Plataforma Interativa – Isolamento Real IEEE 123 Bus")

# =========================================================
#               CONEXÃO COM O BANCO SQLITE
# =========================================================
DB_NAME = "ieee123_isolamento.db"
DB_PATH = Path(__file__).parent / DB_NAME

if not DB_PATH.exists():
    st.error(f"❌ Banco de dados `{DB_NAME}` não encontrado no repositório.")
    st.stop()

conn = sqlite3.connect(DB_PATH)

# ---------------------------------------------------------
# Carrega tabelas importantes
# ---------------------------------------------------------
def load_table(name: str) -> pd.DataFrame:
    return pd.read_sql(f"SELECT * FROM {name}", conn)

try:
    df_coords  = load_table("coords")
    df_loads   = load_table("loads")
    df_topo    = load_table("topology")
    df_nf_map  = load_table("nf_map")
    df_vao_map = load_table("vao_map")
    data_ok = True
except Exception as e:
    st.error(f"Erro ao carregar tabelas do banco: {e}")
    data_ok = False

# =========================================================
#                SIDEBAR – STATUS E VÃO
# =========================================================
st.sidebar.header("📂 Dados carregados")

st.sidebar.markdown(f"**Banco:** `{DB_NAME}`")

if data_ok:
    st.sidebar.markdown("**MASTER:** ✅")
    st.sidebar.markdown("**COORDS:** ✅")
    st.sidebar.markdown("**VAO_MAP:** ✅")
else:
    st.sidebar.error("Problema ao carregar tabelas. Verifique o banco.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Selecione o vão")

# conjunto de barras disponíveis (usa coords)
buses = sorted(df_coords["bus"].unique(), key=str)

# usamos um formulário para só confirmar quando clicar no botão
with st.sidebar.form("form_vao"):
    u_bus = st.selectbox("Barra U", buses, key="sel_u_bus")
    v_bus = st.selectbox("Barra V", buses, key="sel_v_bus")
    btn_confirmar = st.form_submit_button("📌 Confirmar vão")

if btn_confirmar:
    st.session_state.vao_confirmado = True
    st.session_state.u_bus = str(u_bus)
    st.session_state.v_bus = str(v_bus)

# =========================================================
#                TEXTO EXPLICATIVO
# =========================================================
st.markdown(
    """
Ferramenta de apoio à manobra de **desligamento programado** em redes de distribuição,
baseada no alimentador teste **IEEE-123 Bus**.

A inteligência de isolamento (carga interrompida por NF e por vão U-V) foi calculada
anteriormente no **OpenDSS + Python (Colab)** e os resultados foram gravados em
um banco **SQLite** (`ieee123_isolamento.db`).

Este aplicativo usa apenas o banco + coordenadas de barras para exibir:

- ✅ Melhor chave **NF** de manobra para cada vão U-V  
- ⚡ Carga interrompida e número de barras isoladas  
- 🗺️ Mapa colorido da rede com destaque do vão e da NF  
- 📜 “Linha do tempo” da manobra
"""
)

# =========================================================
#            PREPARO DAS COORDENADAS / TOPOLOGIA
# =========================================================
# dicionário {bus: (x, y)}
coords = {str(row["bus"]): (row["x"], row["y"]) for _, row in df_coords.iterrows()}

# garantimos que tudo é string p/ evitar problemas de tipo
df_topo["from_bus"] = df_topo["from_bus"].astype(str)
df_topo["to_bus"]   = df_topo["to_bus"].astype(str)
df_topo["line"]     = df_topo["line"].astype(str)
df_topo["norm"]     = df_topo["norm"].astype(str)

# =========================================================
#                 MAPA BASE DA REDE
# =========================================================
st.subheader("🗺️ Mapa Interativo da Rede")

edge_traces = []

for _, row in df_topo.iterrows():
    a = row["from_bus"]
    b = row["to_bus"]

    if a not in coords or b not in coords:
        continue

    x0, y0 = coords[a]
    x1, y1 = coords[b]

    # cor base: linhas normais
    if row["norm"] == "NF":
        color = "#00bcd4"  # chaves NF fechadas
        width = 2
    else:
        color = "#bbbbbb"
        width = 1.5

    edge_traces.append(
        go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line=dict(color=color, width=width),
            hoverinfo="text",
            text=f"{row['line']} ({a}→{b})",
            showlegend=False,
        )
    )

# nós
node_x, node_y, node_text = [], [], []
for bus, (x, y) in coords.items():
    node_x.append(x)
    node_y.append(y)
    node_text.append(bus)

fig = go.Figure()

for tr in edge_traces:
    fig.add_trace(tr)

fig.add_trace(
    go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        marker=dict(size=7, color="#1f77b4"),
        hoverinfo="text",
        name="Barras",
    )
)

fig.update_layout(
    height=650,
    showlegend=False,
)

st.plotly_chart(fig, use_container_width=True)

# =========================================================
#      PROCESSAMENTO DO VÃO → NF ÓTIMA + CARGA + PLOT
# =========================================================

if st.session_state.get("vao_confirmado", False):

    u = st.session_state.u_bus
    v = st.session_state.v_bus

    st.subheader(f"🔍 Analisando Vão {u} — {v}")

    # --- 1) BUSCA NO BANCO: melhor NF para este vão --------
    df_vao = pd.read_sql(
        f"""
        SELECT *
        FROM vao_map
        WHERE 
            (u_bus = '{u}' AND v_bus = '{v}')
            OR
            (u_bus = '{v}' AND v_bus = '{u}')
        """,
        conn,
    )

    if df_vao.empty:
        st.error("❌ Nenhum resultado encontrado no banco para este vão.")
    else:
        nf = str(df_vao.iloc[0]["nf"])
        kw = float(df_vao.iloc[0]["kw"])
        n_barras = int(df_vao.iloc[0]["n_barras"])

        st.success(
            f"""
            ### ✅ Melhor chave NF: **{nf.upper()}**

            - 🔌 Carga interrompida: **{kw:.1f} kW**  
            - 🧱 Número de barras isoladas: **{n_barras}**
            """
        )

        # =====================================================
        #      2) BUSCA DAS BARRAS ISOLADAS PARA ESTA NF
        # =====================================================
        df_nf = pd.read_sql(f"SELECT * FROM nf_map WHERE nf = '{nf}'", conn)

        try:
            barras_isoladas = eval(df_nf.iloc[0]["barras_isoladas"])
            barras_isoladas = [str(b) for b in barras_isoladas]
        except Exception:
            barras_isoladas = []

        # =====================================================
        #      3) REdesenhar O GRAFO COM DESTAQUES
        # =====================================================

        st.subheader("🌐 Rede com destaque do vão e da NF")

        edge_traces2 = []

        for _, row in df_topo.iterrows():
            a = row["from_bus"]
            b = row["to_bus"]
            if a not in coords or b not in coords:
                continue

            x0, y0 = coords[a]
            x1, y1 = coords[b]

            # COR DA LINHA
            if row["line"] == nf:
                color = "#ff0000"  # NF aberta
                width = 4
            elif (a == u and b == v) or (a == v and b == u):
                color = "#ffa500"  # Vão selecionado
                width = 4
            elif row["norm"] == "NF":
                color = "#00bcd4"  # outras NFs fechadas
                width = 2
            else:
                color = "#cccccc"
                width = 1.5

            edge_traces2.append(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(color=color, width=width),
                    hoverinfo="text",
                    text=f"{row['line']} ({a}→{b})",
                    showlegend=False,
                )
            )

        # nós coloridos
        node_x2, node_y2, node_text2, node_color2 = [], [], [], []

        for bus, (x, y) in coords.items():
            node_x2.append(x)
            node_y2.append(y)
            node_text2.append(bus)

            if bus == u or bus == v:
                node_color2.append("#ffa500")  # barras do vão
            elif bus in barras_isoladas:
                node_color2.append("#ff4d4d")  # isoladas
            else:
                node_color2.append("#1f77b4")  # normal

        fig2 = go.Figure()

        for tr in edge_traces2:
            fig2.add_trace(tr)

        fig2.add_trace(
            go.Scatter(
                x=node_x2,
                y=node_y2,
                mode="markers+text",
                text=node_text2,
                textposition="top center",
                marker=dict(size=8, color=node_color2),
                hoverinfo="text",
                name="Barras",
            )
        )

        fig2.update_layout(height=650, showlegend=False)

        st.plotly_chart(fig2, use_container_width=True)

        # =====================================================
        #             4) LINHA DO TEMPO DA MANOBRA
        # =====================================================
        st.subheader("📜 Linha do tempo da manobra")

        st.markdown(
            f"""
            **1️⃣ - Identificar o vão:** {u} → {v}  

            **2️⃣ - Consultar banco de isolamento:**  
            &nbsp;&nbsp;&nbsp;&nbsp;• NF ótima encontrada: **{nf.upper()}**  
            &nbsp;&nbsp;&nbsp;&nbsp;• Barras desligadas: **{n_barras}**  
            &nbsp;&nbsp;&nbsp;&nbsp;• Carga interrompida: **{kw:.1f} kW**  

            **3️⃣ - Executar manobra de desligamento:**  
            &nbsp;&nbsp;&nbsp;&nbsp;• Abrir chave **{nf.upper()}**  

            **4️⃣ - Verificar rede isolada:**  
            &nbsp;&nbsp;&nbsp;&nbsp;• Conferir barras desenergizadas e cargas afetadas  

            **5️⃣ - Concluir manobra / iniciar trabalhos de manutenção.**
            """
        )

# fecha conexão ao final
conn.close()
