import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# CONFIG
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

st.markdown("""
### Hipótesis Analizadas
1. Influencia del draft de héroes en el resultado final.
2. Influencia del rendimiento individual en la victoria.
3. Influencia de la economía y recursos en el resultado.
""")

# =========================================================
# CARGA DE DATOS
# =========================================================

matches = pd.read_csv('data/matches.csv')
players = pd.read_csv('data/players.csv')
heroes = pd.read_csv('data/hero_stats.csv')

# =========================================================
# UNIÓN
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
# LIMPIEZA
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
# WIN
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
# ROLE
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
# KPIs
# =========================================================

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

st.header("📌 Hipótesis 1 — Influencia del Draft")

col1, col2 = st.columns(2)

# ---------------------------------------------------------
# TOP HÉROES
# ---------------------------------------------------------

with col1:

    fig1, ax1 = plt.subplots(figsize=(8,4))

    top_heroes = (
        df['hero_name']
        .value_counts()
        .head(8)
    )

    sns.barplot(
        x=top_heroes.index,
        y=top_heroes.values,
        palette='Blues_d',
        ax=ax1
    )

    ax1.set_title(
        'Héroes Más Utilizados'
    )

    ax1.set_xlabel('')

    ax1.set_ylabel(
        'Cantidad'
    )

    ax1.tick_params(
        axis='x',
        rotation=35
    )

    st.pyplot(fig1)

# ---------------------------------------------------------
# WINRATE ATRIBUTO
# ---------------------------------------------------------

with col2:

    fig2, ax2 = plt.subplots(figsize=(5,4))

    attr_win = (
        df.groupby('primary_attr')['win']
        .mean() * 100
    )

    labels_map = {
        'agi': 'Agilidad',
        'str': 'Fuerza',
        'int': 'Inteligencia',
        'all': 'Universal'
    }

    attr_win.index = [
        labels_map.get(x, x)
        for x in attr_win.index
    ]

    ax2.pie(
        attr_win.values,
        labels=attr_win.index,
        autopct='%1.1f%%',
        startangle=90
    )

    ax2.set_title(
        'Winrate según atributo'
    )

    st.pyplot(fig2)

# =========================================================
# HIPÓTESIS 2
# =========================================================

st.header("📌 Hipótesis 2 — Rendimiento Individual")

col3, col4 = st.columns(2)

# ---------------------------------------------------------
# VIOLIN KDA
# ---------------------------------------------------------

with col3:

    fig3, ax3 = plt.subplots(figsize=(7,4))

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
        'Distribución del KDA'
    )

    st.pyplot(fig3)

# ---------------------------------------------------------
# SCATTER XP DAMAGE
# ---------------------------------------------------------

with col4:

    fig4, ax4 = plt.subplots(figsize=(7,4))

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
        'XP vs Hero Damage'
    )

    st.pyplot(fig4)

# =========================================================
# HIPÓTESIS 3
# =========================================================

st.header("📌 Hipótesis 3 — Economía y Recursos")

col5, col6 = st.columns(2)

# ---------------------------------------------------------
# GPM ROL
# ---------------------------------------------------------

with col5:

    fig5, ax5 = plt.subplots(figsize=(7,4))

    role_gpm = (
        df.groupby('role')['gold_per_min']
        .mean()
    )

    sns.barplot(
        x=role_gpm.index,
        y=role_gpm.values,
        palette=['limegreen', 'orange', 'deepskyblue'],
        ax=ax5
    )

    ax5.set_title(
        'GPM Promedio por Rol'
    )

    st.pyplot(fig5)

# ---------------------------------------------------------
# NET WORTH
# ---------------------------------------------------------

with col6:

    fig6, ax6 = plt.subplots(figsize=(7,4))

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

    st.pyplot(fig6)

# =========================================================
# TABLA FINAL
# =========================================================

st.subheader("Vista previa del Dataset")

st.dataframe(
    df.head(20)
)