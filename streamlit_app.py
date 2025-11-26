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
