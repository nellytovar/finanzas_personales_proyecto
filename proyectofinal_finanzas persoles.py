# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px


st.set_page_config(layout="wide", page_title="Proyecto: Finanzas Personales", page_icon="💸")

# ...existing code...
import io
@st.cache_data
def load_data(path="finanzas_personales_3000.csv"):
    encodings = ["utf-8", "latin1", "cp1252"]
    # leer el archivo en binario y probar decodificar con distintos encodings
    with open(path, "rb") as f:
        raw = f.read()
    for enc in encodings:
        try:
            text = raw.decode(enc)
            df = pd.read_csv(io.StringIO(text), parse_dates=["fecha_generacion"])
            st.sidebar.write(f"Archivo cargado con encoding: {enc}")
            return df
        except Exception:
            continue
        
    # fallback: decodificar reemplazando bytes inválidos
    text = raw.decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(text), parse_dates=["fecha_generacion"])
    st.sidebar.write("Archivo cargado con fallback (bytes inválidos reemplazados).")
    return df
# ...existing code...

def prepare_features(df):
    features = df[[
        "ingreso_mensual_mxn",
        "gasto_fijo_mxn",
        "gasto_variable_mxn",
        "ahorro_mensual_mxn",
        "deuda_total_mxn",
        "uso_tarjeta_credito_pct",
        "score_financiero"
    ]].copy()
    features = features.fillna(0)
    return features

def compute_clusters(df, n_clusters=3, random_state=42):
    X = prepare_features(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    df_out = df.copy()
    df_out["cluster_financiero"] = labels
    return df_out, kmeans, scaler

def describe_cluster(df, cluster_label):
    sub = df[df["cluster_financiero"] == cluster_label]
    desc = {
        "personas": len(sub),
        "ingreso_promedio": float(sub["ingreso_mensual_mxn"].mean().round(2)) if len(sub)>0 else 0,
        "ahorro_promedio": float(sub["ahorro_mensual_mxn"].mean().round(2)) if len(sub)>0 else 0,
        "deuda_promedio": float(sub["deuda_total_mxn"].mean().round(2)) if len(sub)>0 else 0,
        "score_promedio": float(sub["score_financiero"].mean().round(2)) if len(sub)>0 else 0,
    }
    return desc

# -----------------------
# Main UI
# -----------------------
st.title("💸 Proyecto: Segmentación de Finanzas Personales")
st.markdown("""
Este dashboard segmenta 3,000 perfiles financieros simulados para identificar patrones y entregar recomendaciones prácticas.
Usa el panel lateral para filtrar y ajustar el número de clusters.
""")

# Sidebar - controles
st.sidebar.header("Filtros y parámetros")
data_path = st.sidebar.text_input("Ruta CSV", value="finanzas_personales_3000.csv")
n_clusters = st.sidebar.slider("Número de clusters (KMeans)", min_value=2, max_value=6, value=3, step=1)
edad_min, edad_max = st.sidebar.slider("Rango de edad", 18, 70, (18, 70))
ingreso_min, ingreso_max = st.sidebar.slider("Rango ingreso (MXN)", int(0), int(200000), (0, 80000), step=1000)
empleo_opts = st.sidebar.multiselect("Tipo de empleo", options=["Empleado formal", "Freelance/Independiente", "Desempleado", "Pensionado"], default=["Empleado formal", "Freelance/Independiente"])

# Cargar datos
df = load_data(data_path)

# Aplicar filtros
mask = (
    (df["edad"] >= edad_min) &
    (df["edad"] <= edad_max) &
    (df["ingreso_mensual_mxn"] >= ingreso_min) &
    (df["ingreso_mensual_mxn"] <= ingreso_max) &
    (df["tipo_empleo"].isin(empleo_opts))
)
df_f = df[mask].reset_index(drop=True)

st.sidebar.markdown(f"**Registros después de filtros:** {len(df_f)}")

# Calcular clusters
df_clusters, kmeans_model, scaler = compute_clusters(df_f, n_clusters=n_clusters)

# Top KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Registros", len(df_f))
col2.metric("Ingreso promedio (MXN)", f"${df_f['ingreso_mensual_mxn'].mean():,.0f}")
col3.metric("Ahorro promedio (MXN)", f"${df_f['ahorro_mensual_mxn'].mean():,.0f}")
col4.metric("Deuda promedio (MXN)", f"${df_f['deuda_total_mxn'].mean():,.0f}")

st.markdown("---")

# Visualizaciones en dos columnas
left, right = st.columns((2,1))

with left:
    st.subheader("Distribución: Ingreso vs Ahorro (coloreado por cluster)")
    fig = px.scatter(
        df_clusters,
        x="ingreso_mensual_mxn",
        y="ahorro_mensual_mxn",
        color="cluster_financiero",
        hover_data=["id","edad","tipo_empleo","score_financiero"],
        labels={"ingreso_mensual_mxn":"Ingreso (MXN)", "ahorro_mensual_mxn":"Ahorro (MXN)"},
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Histograma: Score financiero")
    fig2 = px.histogram(df_clusters, x="score_financiero", nbins=30, color="cluster_financiero")
    st.plotly_chart(fig2, use_container_width=True)

with right:
    st.subheader("KPIs por cluster")
    cluster_summary = []
    for c in sorted(df_clusters["cluster_financiero"].unique()):
        desc = describe_cluster(df_clusters, c)
        cluster_summary.append({
            "cluster": int(c),
            "personas": desc["personas"],
            "ingreso_prom": desc["ingreso_promedio"],
            "ahorro_prom": desc["ahorro_promedio"],
            "deuda_prom": desc["deuda_promedio"],
            "score_prom": desc["score_promedio"]
        })
    kpi_df = pd.DataFrame(cluster_summary).set_index("cluster")
    st.table(kpi_df.style.format({
        "personas":"{:,}",
        "ingreso_prom":"${:,.0f}",
        "ahorro_prom":"${:,.0f}",
        "deuda_prom":"${:,.0f}",
        "score_prom":"{:.1f}"
    }))

st.markdown("---")

# Mostrar caracterización del cluster seleccionado
st.subheader("Descripción y recomendaciones por cluster")
cluster_sel = st.selectbox("Selecciona cluster", options=sorted(df_clusters["cluster_financiero"].unique()))
desc = describe_cluster(df_clusters, cluster_sel)

st.markdown(f"**Cluster {cluster_sel} — Resumen rápido**")
st.write(f"- Personas: **{desc['personas']}**")
st.write(f"- Ingreso promedio: **${desc['ingreso_promedio']:,.0f}**")
st.write(f"- Ahorro promedio: **${desc['ahorro_promedio']:,.0f}**")
st.write(f"- Deuda promedio: **${desc['deuda_promedio']:,.0f}**")
st.write(f"- Score promedio: **{desc['score_promedio']:.1f}/100**")

# Recomendaciones simples basadas en cluster
st.markdown("**Recomendaciones sugeridas:**")
if desc['score_promedio'] >= 60:
    st.success("Perfil con buena salud financiera. Mantener ahorro sistemático y explorar inversión conservadora.")
elif desc['score_promedio'] >= 40:
    st.info("Perfil moderado. Revisar gastos variables y priorizar creación de fondo de emergencia.")
else:
    st.warning("Perfil vulnerable. Priorizar reducción de deuda y control de gasto. Buscar asesoría financiera básica.")

st.markdown("---")

# Búsqueda por usuario y detalle
st.subheader("Buscar perfil por ID")
user_id = st.text_input("Ingresa id (ej. user_1)", value="")
if user_id:
    matched = df_clusters[df_clusters["id"] == user_id]
    if matched.empty:
        st.error("No se encontró ese ID con los filtros activos.")
    else:
        row = matched.iloc[0]
        st.write(row.to_frame().T)
        st.write("Recomendación personalizada:")
        score = row["score_financiero"]
        if score >= 60:
            st.write("- Mantener hábito de ahorro. Revisar inversión a mediano plazo.")
        elif score >= 40:
            st.write("- Disminuir gastos no esenciales. Priorizar ahorro mensual automático.")
        else:
            st.write("- Priorizar pago de deuda con mayor interés. Presupuesto estricto.")

st.markdown("---")
st.write("📁 Para reproducir: coloca `finanzas_personales_3000.csv` en la misma carpeta que esta app.")
st.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
