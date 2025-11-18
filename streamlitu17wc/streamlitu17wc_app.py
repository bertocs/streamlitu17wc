# ============================================================
# U17 WORLD CUP DASHBOARD · RCL SCOUT GROUP
# ============================================================
# - Sin login (acceso directo)
# - Filtros globales: Selección, Posición, Minutos
# - Pestaña 1: Producción ofensiva (Top + scatter profesional)
# - Pestaña 2: Radar comparativo multi-jugador / multi-selección,
#              normalizado 0–100 vs grupo filtrado
# - Pestaña 3: Tabla + descarga
# - Pestaña 4: Glosario métricas
# - Footer: marca personal y área de datos RCL
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# =========================
# CONFIGURACIÓN GLOBAL
# =========================
st.set_page_config(
    page_title="U17 World Cup Dashboard - RCL",
    layout="wide",
)

# Rutas
DATA_PATH = Path(__file__).parent / "players_stats_u17_v2.xlsx"
LOGO_PATH = Path(__file__).parent / "rclscoutinggroup.png"



# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    """
    Carga el Excel y crea columnas derivadas útiles:
    - GolesTotal (si hay desglose por tipo de gol)
    - Min_por_partido
    - Goles90, Tiros90, xG90
    - Goles_por_min, Tiros_por_min
    """
    df = pd.read_excel(path)

    # Goles totales a partir del desglose, si existen esas columnas
    goles_cols = {"GolesArea", "GolesLej", "GolesCabeza", "GolesIzq", "GolesDer"}
    if goles_cols.issubset(df.columns):
        df["GolesTotal"] = (
            df["GolesArea"].fillna(0)
            + df["GolesLej"].fillna(0)
            + df["GolesCabeza"].fillna(0)
            + df["GolesIzq"].fillna(0)
            + df["GolesDer"].fillna(0)
        )

    # Minutos por partido
    if {"Min", "Apar"}.issubset(df.columns):
        df["Min_por_partido"] = df["Min"] / df["Apar"].replace(0, pd.NA)

    # Métricas por 90' y por minuto
    if "Min" in df.columns:
        minutos = df["Min"].replace(0, pd.NA)

        if "GolesTotal" in df.columns:
            df["Goles90"] = (df["GolesTotal"] / minutos) * 90
            df["Goles_por_min"] = df["GolesTotal"] / minutos

        if "Tiros" in df.columns:
            df["Tiros90"] = (df["Tiros"] / minutos) * 90
            df["Tiros_por_min"] = df["Tiros"] / minutos

        if "xG" in df.columns:
            df["xG90"] = (df["xG"] / minutos) * 90

    return df


def normalize_metrics(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """
    Normaliza columnas numéricas seleccionadas a escala 0–100
    respecto al rango observado en df (min–max).

    Si una métrica tiene min == max, se fija en 50.
    """
    norm_df = df.copy()
    for col in metrics:
        if col not in norm_df.columns:
            continue
        col_min = norm_df[col].min()
        col_max = norm_df[col].max()
        if pd.isna(col_min) or pd.isna(col_max):
            norm_df[col + "_norm"] = pd.NA
        elif col_max == col_min:
            norm_df[col + "_norm"] = 50
        else:
            norm_df[col + "_norm"] = (norm_df[col] - col_min) / (col_max - col_min) * 100
    return norm_df


# ============================================================
# DASHBOARD PRINCIPAL
# ============================================================
def main() -> None:
    # ------------------ CABECERA CON LOGO --------------------
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        try:
            st.image(str(LOGO_PATH), use_container_width=True)
        except Exception:
            st.write("")

    with col_title:
        st.markdown(
            """
            # 🟦 RCL Scout Group  
            ### U17 World Cup Player Dashboard
            """
        )
        st.caption("Plataforma interna de exploración de rendimiento individual (U17).")

    # Carga de datos
    df = load_data(DATA_PATH)

    # ========================================================
    # SIDEBAR: FILTROS GLOBALES
    # ========================================================
    st.sidebar.header("Filtros globales")

    # Selección (País)
    selecciones = sorted(df["País"].dropna().unique().tolist()) if "País" in df.columns else []
    selected_selecciones = st.sidebar.multiselect(
        "Selección",
        options=selecciones,
        default=selecciones,
        help="Filtra por selección nacional.",
    )

    # Posición
    posiciones = sorted(df["Pos"].dropna().unique().tolist()) if "Pos" in df.columns else []
    selected_pos = st.sidebar.multiselect(
        "Posición",
        options=posiciones,
        default=posiciones,
        help="Filtra por rol nominal del jugador.",
    )

    # Minutos globales
    if "Min" in df.columns and not df["Min"].isna().all():
        min_minutos = int(df["Min"].min())
        max_minutos = int(df["Min"].max())
    else:
        min_minutos, max_minutos = 0, 0

    min_jugados = st.sidebar.slider(
        "Minutos mínimos jugados (global)",
        min_value=min_minutos,
        max_value=max_minutos,
        value=min_minutos,
        step=10,
    )

    # Aplicar filtros
    filtered_df = df.copy()

    if "País" in df.columns and selected_selecciones:
        filtered_df = filtered_df[filtered_df["País"].isin(selected_selecciones)]

    if "Pos" in df.columns and selected_pos:
        filtered_df = filtered_df[filtered_df["Pos"].isin(selected_pos)]

    if "Min" in df.columns:
        filtered_df = filtered_df[filtered_df["Min"] >= min_jugados]

    st.sidebar.markdown("---")
    st.sidebar.write(f"Jugadores filtrados: **{len(filtered_df)}**")

    # ========================================================
    # TABS
    # ========================================================
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "⚽ Producción ofensiva",
            "📊 Radar comparativo",
            "📋 Tabla & descarga",
            "ℹ️ Glosario métricas",
        ]
    )

    # --------------------------------------------------------
    # TAB 1: PRODUCCIÓN OFENSIVA
    # --------------------------------------------------------
    with tab1:
        st.markdown("## Producción ofensiva")
        st.caption("Volumen, eficiencia y perfiles atacantes en el grupo filtrado.")

        col_left, col_right = st.columns(2)

        # ---------- Top jugadores ofensivos ----------
        with col_left:
            st.markdown("### Top por métrica ofensiva")

            metric_candidates = [
                m for m in [
                    "Goles90",
                    "GolesTotal",
                    "Goles_por_min",
                    "Tiros90",
                    "Tiros",
                    "Tiros_por_min",
                    "xG90",
                    "xG",
                ] if m in filtered_df.columns
            ]

            if not metric_candidates:
                st.info("No hay métricas ofensivas suficientes para este gráfico.")
            else:
                metric_top = st.selectbox(
                    "Métrica para ranking",
                    options=metric_candidates,
                    format_func=lambda x: x.replace("_", " "),
                    key="metric_top_offensive",
                )

                top_n = st.slider("N jugadores en el ranking", 5, 40, 10, key="top_n_offensive")

                cols_needed = ["NombreJugador", "Equipo", "País", "Pos", "Min", metric_top]
                cols_present = [c for c in cols_needed if c in filtered_df.columns]

                df_top = (
                    filtered_df[cols_present]
                    .dropna(subset=[metric_top])
                    .sort_values(metric_top, ascending=False)
                    .head(top_n)
                )

                if len(df_top) == 0:
                    st.info("No hay datos para la métrica seleccionada.")
                else:
                    fig_top = px.bar(
                        df_top.sort_values(metric_top),
                        x=metric_top,
                        y="NombreJugador",
                        color="Equipo" if "Equipo" in df_top.columns else None,
                        orientation="h",
                        hover_data=[c for c in ["Equipo", "País", "Pos", "Min"] if c in df_top.columns],
                        title=f"Top {top_n} por {metric_top}",
                    )
                    fig_top.update_layout(
                        xaxis_title=metric_top,
                        yaxis_title="",
                        margin=dict(l=10, r=10, t=60, b=10),
                    )
                    st.plotly_chart(fig_top, use_container_width=True)

        # ---------- Scatter ofensivo con filtro de minutos propio ----------
        with col_right:
            st.markdown("### Diagrama de puntos: volumen vs eficiencia")
            st.caption(
                "Combina una métrica de volumen (eje X) con una de eficiencia o resultado (eje Y)."
            )

            if filtered_df.empty:
                st.info("No hay datos tras los filtros globales.")
            else:
                # Filtro de minutos local para este gráfico
                if "Min" in filtered_df.columns and not filtered_df["Min"].isna().all():
                    min_m = int(filtered_df["Min"].min())
                    max_m = int(filtered_df["Min"].max())
                    min_scatter = st.slider(
                        "Minutos mínimos para este gráfico",
                        min_value=min_m,
                        max_value=max_m,
                        value=min_m,
                        step=10,
                        key="min_scatter_off",
                    )
                    df_scatter_base = filtered_df[filtered_df["Min"] >= min_scatter]
                else:
                    df_scatter_base = filtered_df.copy()

                numeric_cols = df_scatter_base.select_dtypes(include="number").columns.tolist()

                if len(numeric_cols) < 2:
                    st.info("No hay suficientes columnas numéricas para este gráfico.")
                else:
                    # Recomendaciones por defecto
                    vol_candidates = [m for m in ["Tiros90", "Tiros_por_min", "Tiros", "xG90"] if m in numeric_cols]
                    eff_candidates = [m for m in ["Goles90", "Goles_por_min", "GolesTotal"] if m in numeric_cols]

                    x_default = vol_candidates[0] if vol_candidates else numeric_cols[0]
                    y_default = eff_candidates[0] if eff_candidates else numeric_cols[1]

                    x_metric = st.selectbox(
                        "Eje X (volumen)",
                        options=numeric_cols,
                        index=numeric_cols.index(x_default),
                        key="off_x",
                    )
                    y_metric = st.selectbox(
                        "Eje Y (eficiencia / resultado)",
                        options=numeric_cols,
                        index=numeric_cols.index(y_default),
                        key="off_y",
                    )

                    df_scatter = df_scatter_base.dropna(subset=[x_metric, y_metric])

                    if len(df_scatter) == 0:
                        st.info("No hay datos suficientes para las métricas seleccionadas.")
                    else:
                        fig_scatter = px.scatter(
                            df_scatter,
                            x=x_metric,
                            y=y_metric,
                            color="Pos" if "Pos" in df_scatter.columns else None,
                            size="Min" if "Min" in df_scatter.columns else None,
                            hover_name="NombreJugador" if "NombreJugador" in df_scatter.columns else None,
                            hover_data=[c for c in ["Equipo", "País", "GolesTotal", "Tiros", "Min"] if c in df_scatter.columns],
                            title=f"{y_metric} vs {x_metric}",
                        )

                        # Líneas de referencia (medianas) para cuadrantes
                        x_ref = df_scatter[x_metric].median()
                        y_ref = df_scatter[y_metric].median()
                        fig_scatter.add_vline(x=x_ref, line_dash="dash")
                        fig_scatter.add_hline(y=y_ref, line_dash="dash")

                        fig_scatter.update_layout(
                            xaxis_title=x_metric,
                            yaxis_title=y_metric,
                            margin=dict(l=10, r=10, t=60, b=10),
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)

                        st.markdown(
                            """
                            **Interpretación típica de cuadrantes:**  
                            - Arriba derecha: alto volumen y alta eficiencia → perfiles dominantes.  
                            - Arriba izquierda: baja carga pero alta eficiencia → perfiles muy productivos por toque.  
                            - Abajo derecha: mucho volumen con poca conversión → generadores con margen de mejora.  
                            - Abajo izquierda: poco volumen y baja eficiencia → perfiles poco influyentes en finalización.
                            """
                        )

    # --------------------------------------------------------
    # TAB 2: RADAR COMPARATIVO (MÉTRICAS LIBRES)
    # --------------------------------------------------------
    with tab2:
        st.markdown("## Radar comparativo (multi-jugador / multi-selección)")
        st.caption(
            "Valores normalizados 0–100 sobre el grupo filtrado. "
            "Sirve para comparar perfiles, no para leer cifras absolutas."
        )

        if filtered_df.empty:
            st.info("No hay datos tras aplicar los filtros globales.")
        else:
            # Todas las columnas numéricas disponibles en el dataset filtrado
            numeric_cols = filtered_df.select_dtypes(include="number").columns.tolist()

            if not numeric_cols:
                st.info("No hay columnas numéricas disponibles para construir el radar.")
            else:
                st.markdown("### Selección de métricas para el radar")

                metrics_selected = st.multiselect(
                    "Métricas numéricas a incluir (se normalizan 0–100 sobre el grupo filtrado):",
                    options=sorted(numeric_cols),
                    default=sorted(numeric_cols)[:5],  # primeras 5 por defecto para no ensuciar el radar
                )

                if not metrics_selected:
                    st.info("Selecciona al menos una métrica para el radar.")
                else:
                    # Normalizamos SOLO sobre el grupo filtrado actual
                    norm_df = normalize_metrics(filtered_df, metrics_selected)
                    norm_cols = [m + "_norm" for m in metrics_selected]

                    compare_mode = st.radio(
                        "Comparar por",
                        options=["Jugador", "Selección"],
                        horizontal=True,
                    )

                    # ========================
                    # MODO JUGADOR
                    # ========================
                    if compare_mode == "Jugador":
                        if "NombreJugador" not in norm_df.columns:
                            st.info("No hay columna 'NombreJugador' para comparar jugadores.")
                        else:
                            players = sorted(norm_df["NombreJugador"].dropna().unique().tolist())
                            selected_players = st.multiselect(
                                "Jugadores a comparar",
                                options=players,
                                default=players[:3] if len(players) >= 3 else players,
                                help="Elige pocos jugadores (2–5) para que el radar siga siendo legible.",
                            )

                            if not selected_players:
                                st.info("Selecciona al menos un jugador.")
                            else:
                                radar_rows = []
                                for p in selected_players:
                                    df_p = norm_df[norm_df["NombreJugador"] == p]
                                    if df_p.empty:
                                        continue
                                    vals = df_p[norm_cols].mean()
                                    for met, val in zip(metrics_selected, vals.values):
                                        radar_rows.append({
                                            "Métrica": met,
                                            "Valor": val,
                                            "Entidad": p,
                                        })

                                if not radar_rows:
                                    st.info("No hay datos para los jugadores seleccionados.")
                                else:
                                    radar_all = pd.DataFrame(radar_rows)

                                    fig_radar = px.line_polar(
                                        radar_all,
                                        r="Valor",
                                        theta="Métrica",
                                        color="Entidad",
                                        line_close=True,
                                    )
                                    fig_radar.update_traces(fill="toself", opacity=0.4)
                                    fig_radar.update_layout(
                                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                                        title="Radar comparativo entre jugadores seleccionados (0–100 respecto al grupo filtrado)",
                                        margin=dict(l=10, r=10, t=80, b=10),
                                    )
                                    st.plotly_chart(fig_radar, use_container_width=True)

                    # ========================
                    # MODO SELECCIÓN
                    # ========================
                    else:  # compare_mode == "Selección"
                        if "País" not in norm_df.columns:
                            st.info("No hay columna 'País' para comparar selecciones.")
                        else:
                            selecciones_tab = sorted(norm_df["País"].dropna().unique().tolist())
                            selected_countries = st.multiselect(
                                "Selecciones a comparar",
                                options=selecciones_tab,
                                default=selecciones_tab[:3] if len(selecciones_tab) >= 3 else selecciones_tab,
                                help="Elige unas pocas selecciones para evitar ruido visual.",
                            )

                            if not selected_countries:
                                st.info("Selecciona al menos una selección.")
                            else:
                                radar_rows = []
                                for c in selected_countries:
                                    df_c = norm_df[norm_df["País"] == c]
                                    if df_c.empty:
                                        continue
                                    vals = df_c[norm_cols].mean()
                                    for met, val in zip(metrics_selected, vals.values):
                                        radar_rows.append({
                                            "Métrica": met,
                                            "Valor": val,
                                            "Entidad": c,
                                        })

                                if not radar_rows:
                                    st.info("No hay datos para las selecciones seleccionadas.")
                                else:
                                    radar_all = pd.DataFrame(radar_rows)

                                    fig_radar = px.line_polar(
                                        radar_all,
                                        r="Valor",
                                        theta="Métrica",
                                        color="Entidad",
                                        line_close=True,
                                    )
                                    fig_radar.update_traces(fill="toself", opacity=0.4)
                                    fig_radar.update_layout(
                                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                                        title="Radar comparativo entre selecciones seleccionadas (0–100 respecto al grupo filtrado)",
                                        margin=dict(l=10, r=10, t=80, b=10),
                                    )
                                    st.plotly_chart(fig_radar, use_container_width=True)

                st.markdown(
                    """
                    **Criterio de lectura del radar:**  
                    - Los valores se expresan en una escala 0–100 sobre el grupo filtrado.  
                    - 100 ≈ jugador/equipo en el máximo del grupo en esa métrica.  
                    - 50 ≈ valor medio aproximado del grupo.  
                    - Útil para comparar perfiles (qué tipo de jugador/selección es), no para leer números absolutos.
                    """
                )

    # --------------------------------------------------------
    # TAB 3: TABLA + DESCARGA
    # --------------------------------------------------------
    with tab3:
        st.markdown("## Tabla detallada de jugadores (grupo filtrado)")

        df_table = filtered_df.copy()

        if "NombreJugador" in df_table.columns:
            search_name = st.text_input("Filtrar por nombre (contiene):")
            if search_name:
                df_table = df_table[
                    df_table["NombreJugador"]
                    .astype(str)
                    .str.contains(search_name, case=False, na=False)
                ]

        if "Pos" in df_table.columns:
            pos_table_options = sorted(df_table["Pos"].dropna().unique().tolist())
            selected_pos_table = st.multiselect(
                "Posiciones en tabla",
                options=pos_table_options,
                default=pos_table_options,
            )
            df_table = df_table[df_table["Pos"].isin(selected_pos_table)]

        all_columns = df_table.columns.tolist()
        default_cols = [
            col for col in [
                "NombreJugador",
                "Equipo",
                "País",
                "Pos",
                "PosDet",
                "Min",
                "Rating",
                "GolesTotal",
                "Goles90",
                "Tiros",
                "Tiros90",
            ] if col in all_columns
        ]

        selected_cols = st.multiselect(
            "Columnas a mostrar / exportar",
            options=all_columns,
            default=default_cols if default_cols else all_columns,
        )

        sort_metric = st.selectbox(
            "Ordenar por",
            options=selected_cols if selected_cols else all_columns,
        )

        sort_ascending = st.checkbox("Orden ascendente", value=False)

        if len(df_table) == 0 or len(selected_cols) == 0:
            st.info("No hay datos para mostrar con los filtros actuales.")
        else:
            df_show = df_table[selected_cols]

            try:
                df_show_sorted = df_show.sort_values(
                    by=sort_metric,
                    ascending=sort_ascending,
                )
            except Exception:
                df_show_sorted = df_show

            st.dataframe(
                df_show_sorted,
                use_container_width=True,
                height=500,
            )

            csv = df_show_sorted.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Descargar tabla filtrada (CSV)",
                data=csv,
                file_name="u17_filtrado_rcl.csv",
                mime="text/csv",
            )

    # --------------------------------------------------------
    # TAB 4: GLOSARIO
    # --------------------------------------------------------
    with tab4:
        st.markdown("## ℹ️ Glosario de métricas (Sofascore / partido-equipo)")

        st.markdown(
            """
**Identificación y contexto**  
• **IDPartido**: Identificador único del partido.  
• **IDTorneo**: Identificador único del torneo.  
• **Torneo**: Nombre del torneo.  
• **IDTemporada**: Identificador de la temporada.  
• **Ronda**: Jornada o fase del torneo.  

**Equipos**  
• **IDLocal / IDVisita**: Identificadores de los equipos.  
• **Local / Visita**: Nombre de los equipos.  
• **PosLocal / PosVisita**: Posición antes del partido.  
• **OCLocal / OCVisita**: Ocasiones claras creadas.  

**Goles y resultado**  
• **GolesLocal / GolesVisita**: Marcador final.  
• **InicioUTC**: Hora de inicio del partido.  

**Tiros**  
• **TirosLocal / TirosVisita**: Tiros totales.  
• **TirosAPLocal / TirosAPVisita**: Tiros al palo.  
• **PalosLocal / PalosVisita**: Tiros que dan en el poste.  
• **TirosFueraLocal / TirosFueraVisita**: Tiros fuera.  
• **BloqLocal / BloqVisita**: Tiros bloqueados.  
• **TirosAreaLocal / TirosAreaVisita**: Tiros dentro del área.  
• **TirosLejLocal / TirosLejVisita**: Tiros desde lejos.  
• **ErrTiroLocal / ErrTiroVisita**: Errores en tiros.  

**Ataque**  
• **AtaLocal / AtaVisita**: Ataques totales.  
• **AtaTotalLocal / AtaTotalVisita**: Acciones ofensivas totales.  
• **AtasGrandesLocal / AtasGrandesVisita**: Ataques peligrosos.  
• **OCGolLocal / OCGolVisita**: Ocasiones claras convertidas.  
• **OCFallLocal / OCFallVisita**: Ocasiones claras falladas.  
• **ToquesAreaLocal / ToquesAreaVisita**: Toques en el área rival.  
• **PasesProfLocal / PasesProfVisita**: Pases profundos.  

**Balón parado**  
• **EsqLocal / EsqVisita**: Saques de esquina.  
• **TirosLibresLocal / TirosLibresVisita**: Tiros libres.  
• **SaquesLocal / SaquesVisita**: Saques de banda.  
• **SaquesMetaLocal / SaquesMetaVisita**: Saques de portería.  

**Faltas y disciplina**  
• **FaltasLocal / FaltasVisita**: Faltas cometidas.  
• **Faltas3TLocal / Faltas3TVisita**: Faltas en zona crítica.  
• **TAmlocal / TAmvisita**: Amarillas.  
• **TRlocal / TRVisita**: Rojas.  

**Defensa**  
• **EntradasLocal / EntradasVisita**: Entradas intentadas.  
• **EntrGanLocal / EntrGanVisita**: Entradas ganadas.  
• **EntrTotLocal / EntrTotVisita**: Entradas totales.  
• **Entradas3TLocal / Entradas3TVisita**: Entradas en último tercio.  
• **InterLocal / InterVisita**: Intercepciones.  
• **RecupsLocal / RecupsVisita**: Recuperaciones.  
• **DespejesLocal / DespejesVisita**: Despejes.  
• **ErrGolLocal / ErrGolVisita**: Errores que terminan en gol.  

**Juego con balón**  
• **PasesLocal / PasesVisita**: Pases totales.  
• **PasesCompLocal / PasesCompVisita**: Pases completados.  
• **PelotasLargasLocal / PelotasLargasVisita**: Balones largos.  
• **CentrosLocal / CentrosVisita**: Centros.  
• **RegatesLocal / RegatesVisita**: Regates.  

**Duelos**  
• **DuelosLocal / DuelosVisita**: Duelos totales.  
• **DuelosSueloLocal / DuelosSueloVisita**: Duelos en el suelo.  
• **DuelosAereosLocal / DuelosAereosVisita**: Duelos aéreos.  
• **PerdidasLocal / PerdidasVisita**: Pérdidas.  

**Offside**  
• **OffLocal / OffVisita**: Fueras de juego.  

**Portería**  
• **PenAtajLocal / PenAtajVisita**: Penaltis atajados.  
• **SalidasAltasLocal / SalidasAltasVisita**: Salidas por alto.  
• **PunosLocal / PunosVisita**: Puños.
            """
        )

    # ============================================================
    # FOOTER · MARCA PROFESIONAL
    # ============================================================
    st.markdown("""
    <hr style='border:0.5px solid #DDD;'>

    <div style='text-align: center; font-size: 13px; color:#666;'>
        Plataforma desarrollada para el <strong>Área de Datos de RCL Scout Group</strong>.<br>
        Implementación y diseño por <strong>José Alberto Cruz</strong>  
        <a href="https://www.linkedin.com/in/josealbertocs" target="_blank">(LinkedIn)</a>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# ENTRYPOINT
# ============================================================
if __name__ == "__main__":
    main()
