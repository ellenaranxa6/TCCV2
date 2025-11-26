import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sqlite3
from pathlib import Path

NF_COLORS = {
    "sw1": "#FF5733",
    "sw2": "#FFC300",
    "sw3": "#33FF57",
    "sw4": "#3380FF",
    "sw5": "#9D33FF",
    "sw6": "#FF33A8",
}

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
st.markdown("---")
st.subheader("🧩 Análise de Trecho com Múltiplos Vãos")

trecho_input = st.text_input(
    "Sequência de barras (ex: 62,63,64,65,66,60,67)",
    placeholder="62, 63, 64, 65, 66, 60, 67"
)

if st.button("📌 Processar Trecho (Multi-Vãos)"):

    # Parse
    try:
        barras_seq = [b.strip() for b in trecho_input.split(",")]
        barras_seq = [b for b in barras_seq if b != ""]
        if len(barras_seq) < 2:
            st.error("Informe pelo menos duas barras.")
            st.stop()
    except:
        st.error("Entrada inválida.")
        st.stop()

    # Criar vãos
    vaos = [(barras_seq[i], barras_seq[i+1]) for i in range(len(barras_seq)-1)]
    st.write("### 🔍 Vãos identificados:", vaos)

    # Consultar banco
    results = []
    for u, v in vaos:
        query = f"""
            SELECT * FROM vao_map
            WHERE (u_bus='{u}' AND v_bus='{v}')
               OR (u_bus='{v}' AND v_bus='{u}')
        """
        dfv = pd.read_sql(query, conn)
        if dfv.empty:
            results.append({"vao": f"{u}-{v}", "nf": None, "kw": None, "n_barras": None})
        else:
            r = dfv.iloc[0]
            results.append({"vao": f"{u}-{v}", "nf": r["nf"], "kw": r["kw"], "n_barras": r["n_barras"]})

    df_res = pd.DataFrame(results)
    st.write("### 📘 Resultado por vão", df_res)

    # Carga total
    total_kw = df_res["kw"].fillna(0).sum()
    st.success(f"### ⚡ Carga total interrompida: {total_kw:.1f} kW")

    # NFs necessárias
    nfs_necessarias = df_res["nf"].dropna().unique().tolist()
    st.info(f"### 🔑 Conjunto de chaves necessárias: {', '.join(nfs_necessarias)}")

    # ==========================
    # OBTER TOPOLOGIA E COORDS
    # ==========================
    df_topo = pd.read_sql("SELECT * FROM topology", conn)
    df_coords = pd.read_sql("SELECT * FROM coords", conn)

    # Criar dict coords
    coords = {str(row["bus"]): (row["x"], row["y"]) for _, row in df_coords.iterrows()}

    # ==========================
    # MAPA INTERATIVO COMPLETO
    # ==========================
    fig = go.Figure()

    # ---------- 1) Linhas cinza (mapa base)
    for _, row in df_topo.iterrows():
        u, v = str(row["from_bus"]), str(row["to_bus"])
        if u in coords and v in coords:
            fig.add_trace(go.Scatter(
                x=[coords[u][0], coords[v][0]],
                y=[coords[u][1], coords[v][1]],
                mode="lines",
                line=dict(color="#CCCCCC", width=1),
                hoverinfo="none",
                showlegend=False
            ))

    # ---------- 2) Trecho total (azul escuro)
    for u, v in vaos:
        if u in coords and v in coords:
            fig.add_trace(go.Scatter(
                x=[coords[u][0], coords[v][0]],
                y=[coords[u][1], coords[v][1]],
                mode="lines",
                line=dict(color="#0040FF", width=4),
                name="Trecho selecionado"
            ))

    # ---------- 3) Colorir por NF
    for u, v, nf in zip(df_res["vao"], df_res["nf"], df_res["vao"]):
        if nf is None:
            continue
        u_b, v_b = u.split("-")
        if u_b in coords and v_b in coords:
            cor = NF_COLORS.get(nf, "#000")
            fig.add_trace(go.Scatter(
                x=[coords[u_b][0], coords[v_b][0]],
                y=[coords[u_b][1], coords[v_b][1]],
                mode="lines",
                line=dict(color=cor, width=8),
                name=f"{nf} (NF)"
            ))

    # ---------- 4) Barras
    fig.add_trace(go.Scatter(
        x=[coords[b][0] for b in coords],
        y=[coords[b][1] for b in coords],
        mode="markers+text",
        text=[b for b in coords],
        textposition="top center",
        marker=dict(size=8, color="#1f77b4"),
        name="Barras"
    ))

    fig.update_layout(height=750, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

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
