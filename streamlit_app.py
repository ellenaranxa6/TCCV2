import streamlit as st
import plotly.graph_objects as go
import sqlite3
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import ast  # <-- para interpretar a lista de barras da tabela nf_map


# =========================================================
# CONFIGURAÇÃO INICIAL
# =========================================================
st.set_page_config(
    page_title="Isolamento Real IEEE-123 Bus",
    layout="wide"
)

st.title("⚡ Plataforma Interativa – Isolamento Real IEEE 123 Bus")

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "ieee123_isolamento.db"


# =========================================================
# FUNÇÕES AUXILIARES – BANCO
# =========================================================
def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


@st.cache_data(show_spinner=False)
def carregar_coords() -> Dict[str, Tuple[float, float]]:
    """Lê tabela coords(bus, x, y) do banco."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT bus, x, y FROM coords")
    rows = cur.fetchall()
    conn.close()
    return {str(b): (float(x), float(y)) for b, x, y in rows}


@st.cache_data(show_spinner=False)
def carregar_topologia():
    """
    Lê tabela topology(line, from_bus, to_bus, is_switch, norm)
    do banco.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT line, from_bus, to_bus, is_switch, norm "
        "FROM topology"
    )
    rows = cur.fetchall()
    conn.close()

    topo = []
    for line, f, t, is_sw, norm in rows:
        topo.append(
            dict(
                line=str(line),
                from_bus=str(f),
                to_bus=str(t),
                is_switch=bool(is_sw),
                norm=str(norm),
            )
        )
    return topo


@st.cache_data(show_spinner=False)
def carregar_vao_map():
    """
    Lê tabela vao_map(u_bus, v_bus, nf, kw, n_barras).
    Um registro por vão contendo a NF ótima.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT u_bus, v_bus, nf, kw, n_barras FROM vao_map"
    )
    rows = cur.fetchall()
    conn.close()

    vao_map = []
    for u, v, nf, kw, n in rows:
        vao_map.append(
            dict(
                u_bus=str(u),
                v_bus=str(v),
                nf=str(nf),
                kw=float(kw),
                n_barras=int(n),
            )
        )
    return vao_map


@st.cache_data(show_spinner=False)
def carregar_loads():
    """Tabela loads(bus, kw)."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT bus, kw FROM loads")
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    return {str(b): float(kw) for b, kw in rows}


@st.cache_data(show_spinner=False)
def carregar_nf_map():
    """
    Tabela nf_map(nf, barras_isoladas TEXT, kw REAL, n_barras INTEGER).

    barras_isoladas está salva como uma string de lista Python,
    ex.: '["33", "61", "18", ...]'.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT nf, barras_isoladas, kw, n_barras FROM nf_map"
        )
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()

    nf_dict: Dict[str, Dict] = {}
    for nf, barras_str, kw, n in rows:
        barras_set = set()
        if barras_str:
            try:
                lista = ast.literal_eval(barras_str)
                for b in lista:
                    barras_set.add(str(b).strip())
            except Exception:
                # fallback burro: separa por vírgula
                for b in str(barras_str).replace("[", "").replace("]", "").replace('"', "").split(","):
                    b = b.strip()
                    if b:
                        barras_set.add(b)

        nf_dict[str(nf)] = {
            "barras": barras_set,
            "kw": float(kw),
            "n_barras": int(n),
        }

    return nf_dict


def listar_tabelas() -> List[str]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    return rows


# =========================================================
# FUNÇÕES DE PROCESSAMENTO DE VÃOS
# =========================================================
def identificar_vaos_blocos(lista_barras: List[str]) -> List[Tuple[str, str]]:
    """
    Converte lista de barras em pares disjuntos:
    [60,62,63,64,65,66,60,67] ->
    [(60,62), (63,64), (65,66), (60,67)]
    """
    vaos = []
    for i in range(0, len(lista_barras), 2):
        if i + 1 < len(lista_barras):
            u = lista_barras[i].strip()
            v = lista_barras[i + 1].strip()
            if u and v:
                vaos.append((u, v))
    return vaos


def buscar_nf_para_vao(
    u: str,
    v: str,
    vao_map: List[Dict]
) -> Optional[Dict]:
    """
    Procura no vao_map a NF ótima para o vão (u, v),
    considerando que o usuário pode informar em qualquer ordem.
    """
    candidatos = [
        registro for registro in vao_map
        if (registro["u_bus"] == u and registro["v_bus"] == v)
        or (registro["u_bus"] == v and registro["v_bus"] == u)
    ]
    if not candidatos:
        return None

    # Se houver mais de um registro: escolhe o de menor kW, depois menos barras
    candidatos.sort(key=lambda r: (r["kw"], r["n_barras"]))
    return candidatos[0]


def obter_barras_unicas(vaos: List[Tuple[str, str]]) -> List[str]:
    """Retorna a lista de barras únicas presentes em uma lista de vãos."""
    s = set()
    for u, v in vaos:
        s.add(u)
        s.add(v)
    return sorted(s, key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x))


def impacto_consolidado(lista_nf: List[str],
                        loads: Dict[str, float],
                        nf_map_data: Dict[str, Dict]) -> Tuple[float, int, List[str]]:
    """
    Calcula o impacto consolidado de uma manobra que envolve
    várias NFs, **sem dupla contagem** de carga:

      - união das barras isoladas por todas as NFs;
      - soma dos kW por barra (usando tabela loads).

    Retorna: (kW_total, n_barras_unicas, lista_barras_ordenada)
    """
    barras_afetadas = set()
    for nf in lista_nf:
        reg = nf_map_data.get(nf)
        if not reg:
            continue
        barras_afetadas |= reg["barras"]

    kw_total = sum(loads.get(b, 0.0) for b in barras_afetadas)
    barras_ordenadas = sorted(
        barras_afetadas,
        key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x)
    )
    return kw_total, len(barras_afetadas), barras_ordenadas


# =========================================================
# FUNÇÕES DE PLOT
# =========================================================
def construir_mapa_base(coords, topo):
    """
    Retorna uma figura Plotly com a topologia base (sem destaques).
    """
    edge_x = []
    edge_y = []

    for el in topo:
        u = el["from_bus"]
        v = el["to_bus"]
        if u in coords and v in coords:
            x0, y0 = coords[u]
            x1, y1 = coords[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

    node_x = [coords[b][0] for b in coords]
    node_y = [coords[b][1] for b in coords]
    node_text = list(coords.keys())

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="#D3D3D3", width=1),
            hoverinfo="none",
            name="Linhas",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            marker=dict(size=6, color="#1f77b4"),
            name="Barras",
            hovertemplate="Barra %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        height=600,
        showlegend=True,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )

    return fig


def plotar_mapa_com_trecho(
    coords,
    topo,
    vaos: List[Tuple[str, str]],
    info_vaos: List[Dict],
):
    """
    Plota o mapa base + destaques:
      - vãos selecionados (linhas pretas grossas)
      - NFs associadas em vermelho
    """
    fig = construir_mapa_base(coords, topo)

    # --- Destaque dos vãos selecionados ---
    destaque_edge_x = []
    destaque_edge_y = []

    for u, v in vaos:
        if u in coords and v in coords:
            x0, y0 = coords[u]
            x1, y1 = coords[v]
            destaque_edge_x += [x0, x1, None]
            destaque_edge_y += [y0, y1, None]

    if destaque_edge_x:
        fig.add_trace(
            go.Scatter(
                x=destaque_edge_x,
                y=destaque_edge_y,
                mode="lines",
                line=dict(color="black", width=4),
                name="Trecho selecionado (vãos)",
                hoverinfo="none",
            )
        )

    # --- Destaque das NF associadas ---
    nf_edges_x = []
    nf_edges_y = []
    nf_labels_x = []
    nf_labels_y = []
    nf_labels_text = []

    topo_por_line = {el["line"]: el for el in topo}

    for info in info_vaos:
        nf = info["nf"]
        if nf in topo_por_line:
            el = topo_por_line[nf]
            u = el["from_bus"]
            v = el["to_bus"]
            if u in coords and v in coords:
                x0, y0 = coords[u]
                x1, y1 = coords[v]
                nf_edges_x += [x0, x1, None]
                nf_edges_y += [y0, y1, None]
                nf_labels_x.append((x0 + x1) / 2)
                nf_labels_y.append((y0 + y1) / 2)
                nf_labels_text.append(nf)

    if nf_edges_x:
        fig.add_trace(
            go.Scatter(
                x=nf_edges_x,
                y=nf_edges_y,
                mode="lines",
                line=dict(color="red", width=3, dash="dash"),
                name="Chaves NF de manobra",
                hoverinfo="none",
            )
        )

    if nf_labels_x:
        fig.add_trace(
            go.Scatter(
                x=nf_labels_x,
                y=nf_labels_y,
                mode="text",
                text=nf_labels_text,
                textposition="middle center",
                textfont=dict(color="red", size=10),
                showlegend=False,
            )
        )

    return fig


# =========================================================
# CARREGAMENTO DO BANCO E STATUS
# =========================================================
st.sidebar.header("📂 Dados carregados")

if not DB_PATH.exists():
    st.sidebar.error("Banco ieee123_isolamento.db não encontrado na pasta do app.")
    st.stop()

tabelas = listar_tabelas()
st.sidebar.write("Banco:", f"`{DB_PATH.name}`")
st.sidebar.write("TOPOLOGY:", "✅" if "topology" in tabelas else "❌")
st.sidebar.write("COORDS:", "✅" if "coords" in tabelas else "❌")
st.sidebar.write("LOADS:", "✅" if "loads" in tabelas else "❌")
st.sidebar.write("VAO_MAP:", "✅" if "vao_map" in tabelas else "❌")
st.sidebar.write("NF_MAP:", "✅" if "nf_map" in tabelas else "❌")

coords = carregar_coords()
topo = carregar_topologia()
vao_map = carregar_vao_map()
loads = carregar_loads()
nf_map_data = carregar_nf_map()

if not coords or not topo or not vao_map:
    st.error("Banco encontrado, mas alguma tabela essencial está vazia.")
    st.stop()


# =========================================================
# DESCRIÇÃO RÁPIDA
# =========================================================
st.markdown(
    """
Ferramenta de apoio à manobra de **desligamento programado** em redes de distribuição,
baseada no alimentador teste **IEEE-123 Bus**.

A inteligência de isolamento (carga interrompida por NF e por vão U-V) foi calculada
anteriormente no **OpenDSS + Python (Colab)** e os resultados foram gravados
em um banco **SQLite** (`ieee123_isolamento.db`).

Este aplicativo usa **apenas** o banco + coordenadas de barras para exibir:

- ✅ Melhor chave **NF** de manobra para cada vão U-V  
- ⚡ Carga interrompida e número de barras isoladas (por NF e consolidado)  
- 🗺️ Mapa colorido da rede com destaque do vão e da NF  
- 📜 “Linha do tempo” simplificada da manobra  
"""
)

st.markdown("---")


# =========================================================
# MAPA BASE ESTÁTICO
# =========================================================
st.subheader("🗺️ Mapa Base da Rede (IEEE-123 Bus)")
fig_base = construir_mapa_base(coords, topo)
st.plotly_chart(fig_base, use_container_width=True)

st.markdown("---")


# =========================================================
# INTERFACE – VÃO SIMPLES
# =========================================================
st.sidebar.subheader("🔧 Vão simples (U-V)")

lista_barras = sorted(
    coords.keys(),
    key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x),
)

u_simples = st.sidebar.selectbox("Barra U", lista_barras, key="u_simples")
v_simples = st.sidebar.selectbox("Barra V", lista_barras, key="v_simples")

if st.sidebar.button("Confirmar vão simples"):
    vao_simples = (u_simples, v_simples)
    info = buscar_nf_para_vao(u_simples, v_simples, vao_map)

    st.subheader("🔎 Resultado – Vão simples")

    if info is None:
        st.error(f"Não há NF cadastrada no banco para o vão {u_simples} – {v_simples}.")
    else:
        st.success(
            f"**Melhor NF:** `{info['nf']}`  |  "
            f"**Carga interrompida (NF isolada):** {info['kw']:.1f} kW  |  "
            f"**Barras isoladas:** {info['n_barras']}"
        )

        fig_vao = plotar_mapa_com_trecho(
            coords,
            topo,
            vaos=[vao_simples],
            info_vaos=[info],
        )
        st.plotly_chart(fig_vao, use_container_width=True)

        st.markdown("**Sequência de manobra sugerida:**")
        st.markdown(
            f"1️⃣ Abrir NF **{info['nf']}** para isolar o vão **{u_simples} – {v_simples}**.  \n"
            f"✅ Demais chaves permanecem no estado nominal."
        )

    st.markdown("---")


# =========================================================
# INTERFACE – TRECHO MULTI-VÃOS
# =========================================================
st.subheader("🧩 Trecho com múltiplos vãos (entrada em blocos de 2 barras)")

entrada_seq = st.text_input(
    "Sequência de barras (ex: 60,62,63,64,65,66,60,67)",
    value="60,62,63,64,65,66,60,67",
)

if st.button("Processar Trecho (Multi-Vãos)"):
    barras_raw = [b.strip() for b in entrada_seq.split(",") if b.strip()]
    if len(barras_raw) < 2:
        st.error("Informe pelo menos duas barras.")
    else:
        vaos = identificar_vaos_blocos(barras_raw)

        if not vaos:
            st.error("Nenhum vão pôde ser formado com a sequência informada.")
        else:
            st.markdown("### 🔍 Vãos identificados (blocos de 2 barras):")
            st.write(vaos)

            info_vaos = []
            nao_encontrados = []

            for u, v in vaos:
                info = buscar_nf_para_vao(u, v, vao_map)
                if info is None:
                    nao_encontrados.append((u, v))
                else:
                    info_vaos.append(
                        dict(
                            u=u,
                            v=v,
                            nf=info["nf"],
                            kw=info["kw"],
                            n_barras=info["n_barras"],
                        )
                    )

            if nao_encontrados:
                st.warning(
                    "Não foram encontrados registros para os seguintes vãos: "
                    + ", ".join([f"{u}-{v}" for u, v in nao_encontrados])
                )

            if info_vaos:
                st.markdown("### ✅ NF de manobra por vão (impacto individual)")

                df_data = [
                    {
                        "Vão (U-V)": f"{d['u']} - {d['v']}",
                        "NF ótima": d["nf"],
                        "kW interrompidos (NF isolada)": d["kw"],
                        "Barras isoladas (NF isolada)": d["n_barras"],
                    }
                    for d in info_vaos
                ]
                st.table(df_data)

                # gráfico com todos os vãos e NFs
                fig_multi = plotar_mapa_com_trecho(
                    coords,
                    topo,
                    vaos=[(d["u"], d["v"]) for d in info_vaos],
                    info_vaos=info_vaos,
                )
                st.markdown("### 🗺️ Mapa com trecho e NFs destacadas")
                st.plotly_chart(fig_multi, use_container_width=True)

                # ------- IMPACTO CONSOLIDADO (SEM DUPLA CONTAGEM) -------
                st.markdown("### ⚡ Impacto consolidado da manobra (sem dupla contagem)")

                # NFs na ordem de abertura (sem repetir)
                lista_nf_ordenada: List[str] = []
                for d in info_vaos:
                    if d["nf"] not in lista_nf_ordenada:
                        lista_nf_ordenada.append(d["nf"])

                kw_total, n_barras_unicas, barras_ordenadas = impacto_consolidado(
                    lista_nf_ordenada, loads, nf_map_data
                )

                if not nf_map_data:
                    st.warning(
                        "Tabela `nf_map` não está disponível no banco. "
                        "Não foi possível calcular o impacto consolidado."
                    )
                else:
                    st.success(
                        f"**Carga total interrompida:** {kw_total:.1f} kW  \n"
                        f"**Barras desenergizadas únicas:** {n_barras_unicas}"
                    )

                    with st.expander("Ver barras desenergizadas únicas"):
                        st.write(barras_ordenadas)

                # ------- Linha de tempo da manobra -------
                st.markdown("### 📜 Linha de tempo de manobra (sequência sugerida)")

                for i, nf in enumerate(lista_nf_ordenada, start=1):
                    vaos_nf = [
                        f"{d['u']}-{d['v']}"
                        for d in info_vaos
                        if d["nf"] == nf
                    ]
                    st.markdown(
                        f"{i}️⃣ Abrir NF **{nf}** para isolar os vãos: "
                        + ", ".join(vaos_nf)
                    )

                st.markdown(
                    "✅ Após conclusão da manutenção, **fechar as NFs na ordem inversa**, "
                    "conforme os procedimentos operacionais da distribuidora."
                )
