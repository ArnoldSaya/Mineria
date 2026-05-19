import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title="Dashboard Dota 2",
    layout="wide"
)

plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# =========================================================
# TÍTULO
# =========================================================

st.title("🎮 Dashboard de Minería de Datos — Dota 2")

# =========================================================
# CARGA DE DATOS
# =========================================================

matches = pd.read_csv('data/matches.csv')
players = pd.read_csv('data/players.csv')
heroes = pd.read_csv('data/hero_stats.csv')

# =========================================================
# UNIÓN DE TABLAS
# =========================================================

df = players.merge(
    matches,
    on='match_id',
    how='left'
)

df = df.merge(
    heroes,
    on='hero_id',
    how='left'
)

# =========================================================
# LIMPIEZA DE NULOS
# =========================================================

df = df.dropna(
    subset=[
        'match_id',
        'hero_id',
        'hero_name'
    ]
)

numeric_cols = [
    'hero_damage',
    'tower_damage',
    'hero_healing',
    'gold_per_min',
    'xp_per_min',
    'net_worth',
    'kills',
    'deaths',
    'assists'
]

for col in numeric_cols:

    if col in df.columns:

        df[col] = df[col].fillna(0)

# =========================================================
# LIMPIEZA GENERAL
# =========================================================

df['duration_min'] = df['duration'] / 60

df = df[
    (df['duration_min'] > 5) &
    (df['duration_min'] < 120)
]

# =========================================================
# TEAM
# =========================================================

df['team'] = df['player_slot'].apply(
    lambda x: 'Radiant' if x < 128 else 'Dire'
)

# =========================================================
# RESULTADO
# =========================================================

df['win'] = df.apply(
    lambda x:
        1 if (
            (x['team'] == 'Radiant' and x['radiant_win']) or
            (x['team'] == 'Dire' and not x['radiant_win'])
        )
        else 0,
    axis=1
)

df['resultado'] = df['win'].map({
    1: 'Victoria',
    0: 'Derrota'
})

# =========================================================
# KDA
# =========================================================

df['KDA'] = (
    (df['kills'] + df['assists']) /
    df['deaths'].replace(0, 1)
)

# =========================================================
# CLASIFICACIÓN DE ROLES SEGÚN GPM
# =========================================================

def classify_role(gpm):

    if gpm >= 550:
        return 'Core'

    elif gpm >= 400:
        return 'Semi-Core'

    else:
        return 'Support'

df['role'] = df['gold_per_min'].apply(
    classify_role
)

# =========================================================
# EXTRAER ROL PRINCIPAL DEL HÉROE
# =========================================================

def extract_main_role(role_text):

    if pd.isna(role_text):
        return 'Desconocido'

    role_text = str(role_text)

    if 'Carry' in role_text:
        return 'Carry'

    elif 'Support' in role_text:
        return 'Support'

    elif 'Nuker' in role_text:
        return 'Nuker'

    elif 'Initiator' in role_text:
        return 'Initiator'

    elif 'Durable' in role_text:
        return 'Durable'

    else:
        return 'Otros'

df['hero_role'] = df['roles'].apply(
    extract_main_role
)

# =========================================================
# ATRIBUTOS EN ESPAÑOL
# =========================================================

attr_map = {
    'agi': 'Agilidad',
    'str': 'Fuerza',
    'int': 'Inteligencia',
    'all': 'Universal'
}

df['primary_attr_es'] = df['primary_attr'].map(
    attr_map
)

# =========================================================
# NOMBRE COMPLETO HÉROE
# =========================================================

df['hero_label'] = (
    df['hero_name']
    + ' ('
    + df['primary_attr_es']
    + ')'
)

# =========================================================
# KPIs
# =========================================================

st.subheader("📊 Resumen General")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Partidas",
        len(matches)
    )

with col2:

    st.metric(
        "Jugadores",
        len(players)
    )

with col3:

    st.metric(
        "Winrate Radiant",
        f"{matches['radiant_win'].mean()*100:.1f}%"
    )

with col4:

    st.metric(
        "Duración Promedio",
        f"{df['duration_min'].mean():.1f} min"
    )

# =========================================================
# HIPÓTESIS 1
# =========================================================

st.header("📌 Hipótesis 1")

st.markdown("""
### ¿Cómo influye la composición del draft de héroes en el resultado final?
""")

col1, col2 = st.columns([2,1])

# =========================================================
# TODOS LOS HÉROES
# =========================================================

with col1:

    fig1, ax1 = plt.subplots(
        figsize=(14, 30)
    )

    hero_counts = (
        df['hero_label']
        .value_counts()
        .sort_values()
    )

    sns.barplot(
        x=hero_counts.values,
        y=hero_counts.index,
        palette='viridis',
        ax=ax1
    )

    ax1.set_title(
        'Frecuencia de Uso de Todos los Héroes',
        fontsize=18,
        fontweight='bold'
    )

    ax1.set_xlabel(
        'Cantidad de Partidas'
    )

    ax1.set_ylabel(
        'Héroes'
    )

    ax1.tick_params(
        axis='y',
        labelsize=8,
        pad=10
    )

    st.pyplot(fig1)

# =========================================================
# WINRATE POR ATRIBUTO
# =========================================================

with col2:

    fig2, ax2 = plt.subplots(
        figsize=(6,6)
    )

    attr_win = (
        df.groupby('primary_attr_es')['win']
        .mean() * 100
    )

    ax2.pie(
        attr_win.values,
        labels=attr_win.index,
        autopct='%1.1f%%',
        startangle=90
    )

    ax2.set_title(
        'Winrate según Atributo'
    )

    st.pyplot(fig2)

# =========================================================
# HIPÓTESIS 2
# =========================================================

st.header("📌 Hipótesis 2")

st.markdown("""
### ¿Cómo influye el rendimiento individual en la victoria?
""")

col3, col4 = st.columns(2)

# =========================================================
# KDA
# =========================================================

with col3:

    fig3, ax3 = plt.subplots(
        figsize=(8,5)
    )

    sns.violinplot(
        data=df,
        x='resultado',
        y='KDA',
        palette={
            'Victoria': '#4ecdc4',
            'Derrota': '#ff6b6b'
        },
        ax=ax3
    )

    ax3.set_title(
        'Distribución del KDA según Resultado'
    )

    ax3.set_xlabel(
        'Resultado'
    )

    ax3.set_ylabel(
        'KDA'
    )

    st.pyplot(fig3)

# =========================================================
# XP VS HERO DAMAGE
# =========================================================

with col4:

    fig4, ax4 = plt.subplots(
        figsize=(8,5)
    )

    scatter_df = df.sample(
        min(800, len(df))
    )

    sns.scatterplot(
        data=scatter_df,
        x='xp_per_min',
        y='hero_damage',
        hue='resultado',
        alpha=0.7,
        ax=ax4
    )

    ax4.set_title(
        'XP por Minuto vs Hero Damage'
    )

    ax4.set_xlabel(
        'XP por minuto'
    )

    ax4.set_ylabel(
        'Hero Damage'
    )

    st.pyplot(fig4)

# =========================================================
# HIPÓTESIS 3
# =========================================================

st.header("📌 Hipótesis 3")

st.markdown("""
### ¿Cómo influye la economía y distribución de recursos en la victoria?
""")

col5, col6 = st.columns(2)

# =========================================================
# GPM POR ROL DEL HÉROE
# =========================================================

with col5:

    fig5, ax5 = plt.subplots(
        figsize=(9,5)
    )

    hero_role_gpm = (
        df.groupby('hero_role')['gold_per_min']
        .mean()
        .sort_values(ascending=False)
    )

    sns.barplot(
        x=hero_role_gpm.index,
        y=hero_role_gpm.values,
        palette='magma',
        ax=ax5
    )

    ax5.set_title(
        'GPM Promedio según Rol del Héroe',
        fontsize=16,
        fontweight='bold'
    )

    ax5.set_xlabel(
        'Rol del Héroe'
    )

    ax5.set_ylabel(
        'Gold Per Minute'
    )

    ax5.tick_params(
        axis='x',
        rotation=20
    )

    st.pyplot(fig5)

# =========================================================
# NET WORTH
# =========================================================

with col6:

    fig6, ax6 = plt.subplots(
        figsize=(8,5)
    )

    sns.kdeplot(
        data=df,
        x='net_worth',
        hue='resultado',
        fill=True,
        common_norm=False,
        ax=ax6
    )

    ax6.set_title(
        'Distribución de Net Worth'
    )

    ax6.set_xlabel(
        'Net Worth'
    )

    ax6.set_ylabel(
        'Densidad'
    )

    st.pyplot(fig6)

# =========================================================
# TABLA FINAL
# =========================================================

st.header("📋 Vista previa del Dataset")

st.dataframe(
    df.head(20)
)