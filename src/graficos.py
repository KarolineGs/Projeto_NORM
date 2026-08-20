import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from scipy.spatial import ConvexHull

from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.patches import (
    Polygon,
    FancyBboxPatch,
)

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# src/visualization/graficos.py

import logging

logger =  logging.getLogger()

RED = "#C62828"
GREEN = "#2E7D32"
NAVY = "#0D3B66"
ORANGE = "#EF6C00"
GRAY = "#616161"
LIGHT_GRAY = "#E0E0E0"
LIGHT_RED = "#EF9A9A"
WHITE = "#FFFFFF"

def grafico_scatter_matrix(
    df_analise: pd.DataFrame,
    salvar_em: str | None = None,
    exibir: bool = True,
):
    """
    Gera matriz de dispersão entre Bário,
    Estrôncio e Salinidade.

    Vermelho: plataformas com NORM.
    Verde: plataformas sem NORM.
    """

    variaveis = {
        "Bário mediano (mg/L)": "MEDIANA_BARIO_PLAT",
        "Estrôncio mediano (mg/L)": "MEDIANA_ESTRONCIO_PLAT",
        "Salinidade mediana (mg/L)": "MEDIANA_SALINIDADE_PLAT",
        "Relação Bário/Estrôncio (Ba/Sr)": "RELACAO_BARIO_ESTRONCIO",
    }

    # Mantém somente as variáveis que existem no dataframe
    variaveis = {
        nome: coluna
        for nome, coluna in variaveis.items()
        if coluna in df_analise.columns
    }

    colunas = list(variaveis.values())

    dados = (
        df_analise[
            [
                "LOCAL DA GERAÇÃO",
                "TEM_NORM",
                *colunas,
            ]
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=[
                "TEM_NORM",
                *colunas,
            ]
        )
        .copy()
    )

    dados["Classificação"] = (
        dados["TEM_NORM"]
        .astype(int)
        .map({
            1: "Com NORM",
            0: "Sem NORM",
        })
    )

    logger.info(
        "Gerando scatter matrix química | "
        "%d plataformas com dados completos",
        len(dados),
    )

    fig = px.scatter_matrix(
        dados,

        dimensions=colunas,

        labels={
            coluna: nome
            for nome, coluna in variaveis.items()
        },

        color="Classificação",
        symbol="Classificação",

        hover_name="LOCAL DA GERAÇÃO",

        hover_data={
            "TEM_NORM": False,
            "Classificação": False,
        },

        color_discrete_map={
            "Com NORM": "#d62728",
            "Sem NORM": "#2ca02c",
        },

        symbol_map={
            "Com NORM": "circle",
            "Sem NORM": "diamond",
        },

        category_orders={
            "Classificação": [
                "Com NORM",
                "Sem NORM",
            ]
        },

        title=(
            "Matriz de dispersão — "
            "Bário, Estrôncio, Salinidade e relação Ba/Sr"
        ),

        height=900,
        width=1000,
    )

    fig.update_traces(
        diagonal_visible=False,

        marker={
            "size": 7,
            "opacity": 0.8,
            "line": {
                "width": 0.7,
                "color": "black",
            },
        },
    )

    fig.update_layout(
        template="plotly_white",

        legend={
            "title": {
                "text": "Classificação"
            }
        },

        margin={
            "l": 80,
            "r": 50,
            "t": 90,
            "b": 80,
        },
    )

    # ======================================================
    # SALVAR
    # ======================================================

    if salvar_em is not None:

        fig.write_html(
            str(salvar_em)
        )

        logger.info(
            "Scatter matrix salvo em %s",
            salvar_em,
        )

    # ======================================================
    # EXIBIR
    # ======================================================

    if exibir:
        fig.show()

    return fig
def calcular_correlacao_quimica(
    df_analise: pd.DataFrame,
) -> pd.DataFrame:

    variaveis = {
        "Estrôncio": "MEDIANA_ESTRONCIO_PLAT",
        "Salinidade": "MEDIANA_SALINIDADE_PLAT",
        "Bário": "MEDIANA_BARIO_PLAT",

    }

    colunas = [
        coluna
        for coluna in variaveis.values()
        if coluna in df_analise.columns
    ]

    corr = (
        df_analise[colunas]
        .rename(
            columns={
                valor: nome
                for nome, valor
                in variaveis.items()
            }
        )
        .corr(method="spearman")
    )

    return corr

def grafico_quimica_3d(
    df_analise: pd.DataFrame,
    salvar_em: str | None = None,
    exibir: bool = True,
):
    """
    Gera gráfico 3D das variáveis químicas padronizadas.

    Parameters
    ----------
    df_analise : pd.DataFrame
        DataFrame de análise.

    salvar_em : str | None, padrão=None
        Caminho para salvar o gráfico em HTML.

    exibir : bool, padrão=True
        Se True, abre o gráfico.
        Se False, apenas gera/salva.

    Returns
    -------
    plotly.graph_objects.Figure
        Figura Plotly criada.
    """

    variaveis = [
        "MEDIANA_SALINIDADE_PLAT",
        "RELACAO_BARIO_ESTRONCIO",
        "MEDIA_BSW_PLAT",
    ]

    dados = (
        df_analise[
            [
                "LOCAL DA GERAÇÃO",
                "TEM_NORM",
                "PROB_NORM",
                "MASSA_CLASSIFICADA_KG",
                "TIPO_MASSA",
                *variaveis,
            ]
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .copy()
    )

    dados["NORM"] = (
        dados["TEM_NORM"]
        .astype(int)
        .map({
            1: "Com NORM",
            0: "Sem NORM",
        })
    )
    dados["MASSA_CLASSIFICADA_KG"] = dados["MASSA_CLASSIFICADA_KG"].clip(lower=0)
    dados["MASSA_CLASSIFICADA_T"] = dados["MASSA_CLASSIFICADA_KG"] / 1000
    log_massa = np.log1p(dados["MASSA_CLASSIFICADA_KG"])
    dados["TAMANHO_MARCADOR"] = (
        6 + 22 * log_massa / max(log_massa.max(), 1)
    )

    # ======================================================
    # PADRONIZAÇÃO
    # ======================================================

    scaler = StandardScaler()

    colunas_z = [
        "SALINIDADE_Z",
        "RELACAO_BA_SR_Z",
        "BSW_Z",
    ]

    dados[colunas_z] = scaler.fit_transform(
        dados[variaveis]
    )

    # ======================================================
    # GRÁFICO
    # ======================================================

    fig = px.scatter_3d(
        dados,
        x="SALINIDADE_Z",
        y="RELACAO_BA_SR_Z",
        z="BSW_Z",
        color="PROB_NORM",
        symbol="NORM",
        size="TAMANHO_MARCADOR",
        size_max=28,
        hover_name="LOCAL DA GERAÇÃO",

        hover_data={
            "MEDIANA_SALINIDADE_PLAT": ":.2f",
            "RELACAO_BARIO_ESTRONCIO": ":.4f",
            "MEDIA_BSW_PLAT": ":.2f",
            "TIPO_MASSA": True,
            "MASSA_CLASSIFICADA_T": ":,.2f",
            "PROB_NORM": ":.1%",
            "TAMANHO_MARCADOR": False,
            "SALINIDADE_Z": False,
            "RELACAO_BA_SR_Z": False,
            "BSW_Z": False,
        },

        color_continuous_scale="RdYlBu_r",
        range_color=[0, 1],

        title=(
            "Salinidade × Relação Ba/Sr × BSW — "
            "variáveis padronizadas"
        ),

        opacity=0.85,
    )

    fig.update_layout(
        width=1000,
        height=750,

        scene=dict(
            xaxis_title=(
                "Salinidade padronizada (z-score)"
            ),
            yaxis_title=(
                "Relação Ba/Sr padronizada (z-score)"
            ),
            zaxis_title=(
                "BSW padronizado (z-score)"
            ),
            aspectmode="cube",
        ),

        legend={
            "title": {"text": "Classificação"},
            "x": 0.02,
            "y": 0.98,
            "xanchor": "left",
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.8)",
        },
        coloraxis_colorbar={
            "title": "Probabilidade de NORM",
            "tickformat": ".0%",
            "x": 1.05,
            "y": 0.45,
            "len": 0.65,
        },
        margin={"l": 20, "r": 190, "b": 20, "t": 70},
    )

    # ======================================================
    # SALVAR
    # ======================================================

    if salvar_em is not None:

        fig.write_html(
            str(salvar_em)
        )

        logger.info(
            "Gráfico químico 3D salvo em %s",
            salvar_em,
        )

    # ======================================================
    # EXIBIR
    # ======================================================

    if exibir:

        fig.show()

    return fig
def grafico_boxplots_quimicos(
    df_analise: pd.DataFrame,
    salvar_em: str | None = None,
    exibir: bool = True,
):

    mapa = {
        "MEDIANA_SALINIDADE_PLAT": "SALINIDADE",
        "MEDIANA_BARIO_PLAT": "BARIO",
        "MEDIANA_ESTRONCIO_PLAT": "ESTRONCIO",
        "RELACAO_BARIO_ESTRONCIO": "RELACAO_BA_SR",
    }

    mapa = {
        coluna: nome
        for coluna, nome in mapa.items()
        if coluna in df_analise.columns
    }

    if not mapa:
        raise ValueError(
            "Nenhuma variavel quimica disponivel para gerar os boxplots."
        )

    dados = (
        df_analise[
            [
                "TEM_NORM",
                *mapa.keys(),
            ]
        ]
        .rename(columns=mapa)
        .copy()
    )

    numericas = ["TEM_NORM", *mapa.values()]

    for coluna in numericas:

        dados[coluna] = pd.to_numeric(
            dados[coluna],
            errors="coerce",
        )

    dados = (
        dados
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(subset=numericas)
    )

    colunas_quimicas = list(mapa.values())
    dados = dados[
        dados[colunas_quimicas].gt(0).all(axis=1)
    ].copy()

    dados["GRUPO"] = (
        dados["TEM_NORM"]
        .astype(int)
        .map({
            1: "Com NORM",
            0: "Sem NORM",
        })
    )

    ordem = [
        "Com NORM",
        "Sem NORM",
    ]

    cores = {
        "Com NORM": "#E74C3C",
        "Sem NORM": "#3498DB",
    }

    variaveis = [
        (
            "SALINIDADE",
            "Salinidade (mg/L)",
        ),
        (
            "BARIO",
            "Bário (mg/L)",
        ),
        (
            "ESTRONCIO",
            "Estrôncio (mg/L)",
        ),
        (
            "RELACAO_BA_SR",
            "Relação Bário/Estrôncio (Ba/Sr)",
        ),
    ]

    variaveis = [
        (coluna, titulo)
        for coluna, titulo in variaveis
        if coluna in colunas_quimicas
    ]

    fig, axes = plt.subplots(
        int(np.ceil(len(variaveis) / 4)),
        min(4, len(variaveis)),
        figsize=(
            3.5 * min(4, len(variaveis)),
            5.2 * int(np.ceil(len(variaveis) / 4)),
        ),
        squeeze=False,
    )

    axes = axes.ravel()

    for ax, (coluna, titulo) in zip(
        axes,
        variaveis,
    ):

        sns.boxplot(
            data=dados,
            x="GRUPO",
            y=coluna,
            order=ordem,
            hue="GRUPO",
            hue_order=ordem,
            palette=cores,
            legend=False,
            width=0.48,
            showfliers=False,
            saturation=1,
            linewidth=1,
            boxprops={
                "alpha": 0.35,
            },
            whiskerprops={
                "linewidth": 1,
            },
            capprops={
                "linewidth": 1,
            },
            medianprops={
                "color": "#8B1A1A",
                "linewidth": 1.3,
            },
            ax=ax,
        )

        sns.stripplot(
            data=dados,
            x="GRUPO",
            y=coluna,
            order=ordem,
            hue="GRUPO",
            hue_order=ordem,
            palette=cores,
            legend=False,
            jitter=0.15,
            size=4,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.4,
            ax=ax,
        )

        ax.set_yscale(
            "log"
        )

        ax.set_title(
            titulo,
            fontsize=11,
            fontweight="bold",
            color=NAVY,
            pad=12,
        )

        ax.set_xlabel("")

        ax.set_ylabel(
            "Escala logarítmica",
            fontsize=8,
        )

        ax.grid(
            axis="y",
            which="major",
            linestyle=":",
            linewidth=0.7,
            alpha=0.3,
        )

        ax.grid(
            axis="y",
            which="minor",
            linestyle=":",
            linewidth=0.4,
            alpha=0.12,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[len(variaveis):]:
        ax.set_visible(False)

    legenda = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=cores["Com NORM"],
            markeredgecolor=WHITE,
            markersize=7,
            label="Com NORM",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=cores["Sem NORM"],
            markeredgecolor=WHITE,
            markersize=7,
            label="Sem NORM",
        ),
    ]

    fig.legend(
        handles=legenda,
        loc="lower center",
        ncol=2,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(
        rect=[0, 0.08, 1, 1],
        w_pad=2.5,
    )

    if salvar_em:

        fig.savefig(
            salvar_em,
            dpi=300,
            bbox_inches="tight",
        )

    if exibir:
        plt.show()
    else:
        plt.close(fig)

    return fig
def tamanho_bolha(
    bsw
):
    bsw = np.asarray(
        bsw,
        dtype=float,
    )

    return (
        35
        + np.sqrt(
            np.clip(
                bsw,
                0,
                None,
            )
        )
        * 35
    )
def preparar_dados_similaridade(
    df_analise: pd.DataFrame,
    coluna_x: str,
    coluna_y: str,
) -> pd.DataFrame:

    colunas = [
        "LOCAL DA GERAÇÃO",
        "TEM_NORM",
        "MEDIA_BSW_PLAT",
        coluna_x,
        coluna_y,
    ]

    dados = (
        df_analise[colunas]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .copy()
    )

    dados = dados[
        (dados[coluna_x] > 0)
        & (dados[coluna_y] > 0)
    ]

    return dados
def calcular_envoltoria_log(
    dados: pd.DataFrame,
    coluna_x: str,
    coluna_y: str,
):

    pontos = (
        dados[
            [
                coluna_x,
                coluna_y,
            ]
        ]
        .to_numpy(dtype=float)
    )

    if len(pontos) < 3:
        return None, None

    pontos_log = np.log10(
        pontos
    )

    hull = ConvexHull(
        pontos_log
    )

    vertices_log = (
        pontos_log[
            hull.vertices
        ]
    )

    vertices_originais = (
        10 ** vertices_log
    )

    return (
        vertices_log,
        vertices_originais,
    )
def verificar_pontos_dentro(
    dados: pd.DataFrame,
    vertices_log,
    coluna_x: str,
    coluna_y: str,
) -> pd.Series:

    if vertices_log is None:

        return pd.Series(
            False,
            index=dados.index,
        )

    poligono = Path(
        vertices_log
    )

    pontos = (
        dados[
            [
                coluna_x,
                coluna_y,
            ]
        ]
        .to_numpy(dtype=float)
    )

    pontos_log = np.log10(
        pontos
    )

    dentro = poligono.contains_points(
        pontos_log,
        radius=1e-9,
    )

    return pd.Series(
        dentro,
        index=dados.index,
        dtype=bool,
    )
def desenhar_similaridade(
    ax,
    df_analise: pd.DataFrame,
    coluna_x: str,
    coluna_y: str,
    titulo: str,
    xlabel: str,
    ylabel: str,
):

    dados = preparar_dados_similaridade(
        df_analise,
        coluna_x,
        coluna_y,
    )

    com_norm = dados[
        dados["TEM_NORM"] == 1
    ].copy()

    sem_norm = dados[
        dados["TEM_NORM"] == 0
    ].copy()

    vertices_log, vertices = (
        calcular_envoltoria_log(
            com_norm,
            coluna_x,
            coluna_y,
        )
    )

    sem_norm["PERFIL_SEMELHANTE"] = (
        verificar_pontos_dentro(
            sem_norm,
            vertices_log,
            coluna_x,
            coluna_y,
        )
    )

    semelhantes = sem_norm[
        sem_norm["PERFIL_SEMELHANTE"]
    ].copy()

    if vertices is not None:

        ax.add_patch(
            Polygon(
                vertices,
                closed=True,
                facecolor=LIGHT_RED,
                edgecolor=RED,
                alpha=0.22,
                linewidth=2.2,
                zorder=1,
            )
        )

    ax.scatter(
        com_norm[coluna_x],
        com_norm[coluna_y],
        s=tamanho_bolha(
            com_norm["MEDIA_BSW_PLAT"]
        ),
        color=RED,
        edgecolor=WHITE,
        linewidth=1,
        alpha=0.85,
        zorder=4,
    )

    ax.scatter(
        sem_norm[coluna_x],
        sem_norm[coluna_y],
        s=tamanho_bolha(
            sem_norm["MEDIA_BSW_PLAT"]
        ),
        color=GREEN,
        edgecolor=WHITE,
        linewidth=1,
        alpha=0.90,
        zorder=5,
    )

    ax.scatter(
        semelhantes[coluna_x],
        semelhantes[coluna_y],
        s=(
            tamanho_bolha(
                semelhantes["MEDIA_BSW_PLAT"]
            )
            + 180
        ),
        facecolor="none",
        edgecolor=NAVY,
        linewidth=2.2,
        zorder=6,
    )

    for _, linha in dados.iterrows():

        ax.text(
            linha[coluna_x],
            linha[coluna_y],
            f'{linha["MEDIA_BSW_PLAT"]:.0f}',
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color=WHITE,
            zorder=9,
        )

    for _, linha in sem_norm.iterrows():

        ax.annotate(
            linha["LOCAL DA GERAÇÃO"],
            xy=(
                linha[coluna_x],
                linha[coluna_y],
            ),
            xytext=(8, 7),
            textcoords="offset points",
            fontsize=7.8,
            fontweight="bold",
            color=NAVY,
            bbox=dict(
                boxstyle="round,pad=0.16",
                facecolor=WHITE,
                edgecolor="none",
                alpha=0.75,
            ),
            zorder=10,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_title(
        titulo,
        fontsize=13,
        fontweight="bold",
        color=NAVY,
        loc="left",
        pad=12,
    )

    ax.set_xlabel(
        xlabel,
        fontsize=11,
        fontweight="bold",
        color=NAVY,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=11,
        fontweight="bold",
        color=NAVY,
    )

    ax.grid(
        which="major",
        linestyle=":",
        linewidth=0.8,
        alpha=0.30,
    )

    ax.grid(
        which="minor",
        linestyle=":",
        linewidth=0.4,
        alpha=0.12,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return semelhantes

def grafico_regioes_similaridade(
    df_analise: pd.DataFrame,
    salvar_em: str | None = None,
    exibir: bool = True,
):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 13),
    )

    axes = axes.ravel()

    fig.patch.set_facecolor(
        WHITE
    )

    # ==============================================
    # BÁRIO × ESTRÔNCIO
    # ==============================================

    semelhantes_ba_sr = desenhar_similaridade(
        ax=axes[0],
        df_analise=df_analise,
        coluna_x="MEDIANA_BARIO_PLAT",
        coluna_y="MEDIANA_ESTRONCIO_PLAT",
        titulo="Bário × Estrôncio",
        xlabel="Bário (mg/L)",
        ylabel="Estrôncio (mg/L)",
    )

    # ==============================================
    # SALINIDADE × BÁRIO
    # ==============================================

    semelhantes_sal_ba = desenhar_similaridade(
        ax=axes[1],
        df_analise=df_analise,
        coluna_x="MEDIANA_SALINIDADE_PLAT",
        coluna_y="MEDIANA_BARIO_PLAT",
        titulo="Salinidade × Bário",
        xlabel="Salinidade (mg/L)",
        ylabel="Bário (mg/L)",
    )

    # ==============================================
    # SALINIDADE × ESTRÔNCIO
    # ==============================================

    semelhantes_sal_sr = desenhar_similaridade(
        ax=axes[2],
        df_analise=df_analise,
        coluna_x="MEDIANA_SALINIDADE_PLAT",
        coluna_y="MEDIANA_ESTRONCIO_PLAT",
        titulo="Salinidade × Estrôncio",
        xlabel="Salinidade (mg/L)",
        ylabel="Estrôncio (mg/L)",
    )

    # ==============================================
    # SALINIDADE × RELAÇÃO ESTRÔNCIO/BÁRIO
    # ==============================================

    semelhantes_sal_relacao = desenhar_similaridade(
        ax=axes[3],
        df_analise=df_analise,
        coluna_x="MEDIANA_SALINIDADE_PLAT",
        coluna_y="RELACAO_BARIO_ESTRONCIO",
        titulo="Salinidade × Relação Ba/Sr",
        xlabel="Salinidade (mg/L)",
        ylabel="Relação Bário/Estrôncio (Ba/Sr)",
    )



    legenda = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=RED,
            markeredgecolor=WHITE,
            markersize=9,
            label="Plataformas com NORM",
        ),

        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=GREEN,
            markeredgecolor=WHITE,
            markersize=9,
            label="Plataformas sem NORM",
        ),

        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=NAVY,
            markeredgewidth=2,
            markersize=13,
            label=(
                "Sem NORM dentro da "
                "região de similaridade"
            ),
        ),

        Line2D(
            [0],
            [0],
            color=RED,
            linewidth=2.2,
            label="Limite da região NORM",
        ),
    ]

    fig.legend(
        handles=legenda,
        title="Classificação",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
        fontsize=9,
        frameon=True,
    )

    fig.text(
        0.5,
        0.97,
        (
            "O tamanho das bolhas e os valores "
            "internos representam o BSW (%)"
        ),
        ha="center",
        va="top",
        fontsize=10,
        color=GRAY,
    )

    plt.tight_layout(
        rect=[0, 0.13, 1, 0.94],
        w_pad=2.5,
    )

    if salvar_em:

        fig.savefig(
            salvar_em,
            dpi=300,
            bbox_inches="tight",
            facecolor=WHITE,
        )

    if exibir:
        plt.show()
    else:
        plt.close(fig)

    return fig, {
        "BARIO_ESTRONCIO": semelhantes_ba_sr,
        "SALINIDADE_BARIO": semelhantes_sal_ba,
        "SALINIDADE_ESTRONCIO": semelhantes_sal_sr,
        "SALINIDADE_RELACAO_BA_SR": semelhantes_sal_relacao,
    }
def desenhar_matriz_cartoes(
    ax,
    y_real,
    y_previsto,
    probabilidades,
    resumo_folds: dict,
    limite_youden: float,
    titulo: str,
    letra: str,
):

    tn, fp, fn, tp = confusion_matrix(
        y_real,
        y_previsto,
        labels=[0, 1],
    ).ravel()

    total = (
        tn + fp + fn + tp
    )

    acertos = (
        tn + tp
    )

    sensibilidade = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )

    especificidade = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    acuracia = (
        acertos / total
        if total > 0
        else np.nan
    )

    auc = roc_auc_score(
        y_real,
        probabilidades,
    )

    cartoes = [
        (
            0, 1, tn,
            "Verdadeiro\nnegativo",
            "#B9BEC7",
        ),
        (
            1, 1, fp,
            "Falso\npositivo",
            "#F4D6D4",
        ),
        (
            0, 0, fn,
            "Falso\nnegativo",
            "#F4D6D4",
        ),
        (
            1, 0, tp,
            "Verdadeiro\npositivo",
            "#9DCCAE",
        ),
    ]

    for (
        x,
        y_pos,
        valor,
        descricao,
        cor_fundo,
    ) in cartoes:

        caixa = FancyBboxPatch(
            (x, y_pos),
            width=0.88,
            height=0.88,
            boxstyle=(
                "round,pad=0.03,"
                "rounding_size=0.06"
            ),
            facecolor=cor_fundo,
            edgecolor="#C7C7C7",
            linewidth=1.1,
        )

        ax.add_patch(caixa)

        ax.text(
            x + 0.44,
            y_pos + 0.56,
            str(valor),
            ha="center",
            va="center",
            fontsize=22,
            fontweight="bold",
            color="#263445",
        )

        ax.text(
            x + 0.44,
            y_pos + 0.17,
            descricao,
            ha="center",
            va="center",
            fontsize=6.5,
            color="#626A73",
        )

    ax.text(
        -0.13,
        1.44,
        "Real\nsem NORM",
        ha="right",
        va="center",
        fontsize=8,
        fontweight="bold",
    )

    ax.text(
        -0.13,
        0.44,
        "Real\ncom NORM",
        ha="right",
        va="center",
        fontsize=8,
        fontweight="bold",
    )

    ax.text(
        0.44,
        -0.12,
        "Previsto\nsem NORM",
        ha="center",
        va="top",
        fontsize=8,
        fontweight="bold",
    )

    ax.text(
        1.44,
        -0.12,
        "Previsto\ncom NORM",
        ha="center",
        va="top",
        fontsize=8,
        fontweight="bold",
    )

    ax.set_title(
        f"{letra} — {titulo}",
        fontsize=11,
        fontweight="bold",
        color=NAVY,
        pad=10,
    )

    def media_dp(metrica: str, percentual: bool = True) -> str:
        resumo = resumo_folds[metrica]
        if percentual:
            return (
                f"{resumo['media']:.0%} ± "
                f"{resumo['desvio_padrao']:.0%}"
            )
        return (
            f"{resumo['media']:.2f} ± "
            f"{resumo['desvio_padrao']:.2f}"
        )

    ax.text(
        0.94,
        -0.78,
        (
            f"Limiar de Youden: {limite_youden:.3f}\n"
            f"AUC: {media_dp('AUC', percentual=False)}  •  "
            f"Acurácia: {media_dp('ACURACIA')}\n"
            f"Precisão: {media_dp('PRECISAO')}  •  "
            f"F1: {media_dp('F1_SCORE')}\n"
            f"Sensibilidade: {media_dp('SENSIBILIDADE')}  •  "
            f"Especificidade: {media_dp('ESPECIFICIDADE')}"
        ),
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
    )

    ax.text(
        0.94,
        -1.30,
        (
            f"{acertos} de {total} plataformas "
            "classificadas corretamente"
        ),
        ha="center",
        va="center",
        fontsize=7.5,
        color="#6B7280",
    )

    ax.set_xlim(
        -0.45,
        1.95,
    )

    ax.set_ylim(
        -1.48,
        1.98,
    )

    ax.set_aspect(
        "equal"
    )

    ax.axis(
        "off"
    )
def grafico_matrizes_confusao(
    dados_modelo: pd.DataFrame,
    predicoes_modelos: dict,
    salvar_em: str | None = None,
    exibir: bool = True,
):

    ordem_modelos = [
        "Salinidade",
        "Bário",
        "Estrôncio",
        "Salinidade + Bário",
        "Salinidade + Estrôncio",
        "Salinidade + Bário + Estrôncio"
    ]

    letras = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K"
    ]

    if not predicoes_modelos:
        raise ValueError(
            "Nenhuma predicao de modelo disponivel para gerar as matrizes."
        )

    # Usa exatamente os modelos produzidos pela etapa de modelagem. Isso evita
    # que nomes de variaveis opcionais fiquem dessincronizados entre modulos.
    ordem_modelos = list(predicoes_modelos)
    letras = [
        chr(ord("A") + indice)
        for indice in range(len(ordem_modelos))
    ]

    y = dados_modelo[
        "TEM_NORM"
    ].to_numpy()

    numero_colunas = min(4, len(ordem_modelos))
    numero_linhas = int(np.ceil(len(ordem_modelos) / numero_colunas))

    fig, axes = plt.subplots(
        numero_linhas,
        numero_colunas,
        figsize=(5.4 * numero_colunas, 5.0 * numero_linhas),
        squeeze=False,
    )

    axes = axes.flatten()

    for (
        ax,
        nome_modelo,
        letra,
    ) in zip(
        axes,
        ordem_modelos,
        letras,
    ):

        resultado = (
            predicoes_modelos[
                nome_modelo
            ]
        )

        desenhar_matriz_cartoes(
            ax=ax,
            y_real=y,
            y_previsto=resultado[
                "previsoes"
            ],
            probabilidades=resultado[
                "probabilidades"
            ],
            resumo_folds=resultado["resumo_folds"],
            limite_youden=resultado["limite_youden"],
            titulo=nome_modelo,
            letra=letra,
        )

    for ax in axes[len(ordem_modelos):]:
        ax.set_visible(False)

    folds = predicoes_modelos[
        ordem_modelos[0]
    ]["folds"]

    fig.suptitle(
        (
            "Comparação das matrizes de confusão "
            "dos modelos químicos\n"
            f"Validação cruzada estratificada "
            f"com {folds} folds"
        ),
        fontsize=15,
        fontweight="bold",
        color=NAVY,
        y=0.99,
    )

    fig.text(
        0.5,
        0.018,
        (
            "Limiar de classificação determinado pelo índice de Youden, "
            "maximizando conjuntamente sensibilidade e especificidade"
        ),
        ha="center",
        fontsize=9,
        color="#626A73",
    )

    plt.tight_layout(
        rect=[0, 0.08, 1, 0.94],
        h_pad=2.5,
        w_pad=2,
    )

    if salvar_em:

        fig.savefig(
            salvar_em,
            dpi=300,
            bbox_inches="tight",
            facecolor=WHITE,
        )

    if exibir:
        plt.show()
    else:
        plt.close(fig)

    return fig

def grafico_resultado_regressao(
    dados_modelo: pd.DataFrame,
    resultado_modelo: dict,
    salvar_em: str | None = None,
    exibir: bool = True,
):

    dados = dados_modelo.copy()

    y = dados[
        "TEM_NORM"
    ].to_numpy()

    probabilidades = resultado_modelo[
        "probabilidades"
    ]

    previsoes = resultado_modelo[
        "previsoes"
    ]
    limite_youden = resultado_modelo["limite_youden"]
    nome_features = " + ".join(resultado_modelo["features"])

    dados["PROB_NORM"] = probabilidades

    dados["CLASSE_PREVISTA"] = (
        previsoes
    )

    dados["RESULTADO"] = np.select(
        [
            (
                (dados["TEM_NORM"] == 1)
                &
                (dados["CLASSE_PREVISTA"] == 1)
            ),
            (
                (dados["TEM_NORM"] == 0)
                &
                (dados["CLASSE_PREVISTA"] == 0)
            ),
            (
                (dados["TEM_NORM"] == 0)
                &
                (dados["CLASSE_PREVISTA"] == 1)
            ),
            (
                (dados["TEM_NORM"] == 1)
                &
                (dados["CLASSE_PREVISTA"] == 0)
            ),
        ],
        [
            "Verdadeiro positivo",
            "Verdadeiro negativo",
            "Falso positivo",
            "Falso negativo",
        ],
        default="Não classificado",
    )

    auc = roc_auc_score(
        y,
        probabilidades,
    )

    fpr, tpr, _ = roc_curve(
        y,
        probabilidades,
    )

    tn, fp, fn, tp = confusion_matrix(
        y,
        previsoes,
        labels=[0, 1],
    ).ravel()

    sensibilidade = (
        tp / (tp + fn)
    )

    especificidade = (
        tn / (tn + fp)
    )

    dados_plot = (
        dados
        .sort_values(
            "PROB_NORM"
        )
        .reset_index(drop=True)
    )

    dados_plot["POSICAO"] = np.arange(
        len(dados_plot)
    )

    cores_resultado = {
        "Verdadeiro positivo": RED,
        "Verdadeiro negativo": GREEN,
        "Falso positivo": ORANGE,
        "Falso negativo": NAVY,
    }

    cores_pontos = (
        dados_plot["RESULTADO"]
        .map(cores_resultado)
    )

    fig, (
        ax1,
        ax2,
    ) = plt.subplots(
        1,
        2,
        figsize=(16, 7),
        gridspec_kw={
            "width_ratios": [
                0.85,
                1.65,
            ]
        },
    )

    # ROC
    ax1.plot(
        fpr,
        tpr,
        linewidth=3,
        label=(
            f"{nome_features} — "
            f"AUC = {auc:.3f}"
        ),
    )

    ax1.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.3,
        label="Classificação aleatória",
    )

    ax1.set_xlim(
        0,
        1,
    )

    ax1.set_ylim(
        0,
        1.02,
    )

    ax1.set_xlabel(
        "Taxa de falsos positivos\n"
        "(1 − especificidade)"
    )

    ax1.set_ylabel(
        "Taxa de verdadeiros positivos\n"
        "(sensibilidade)"
    )

    ax1.set_title(
        "A. Desempenho discriminante",
        fontsize=13,
        fontweight="bold",
        color=NAVY,
        loc="left",
    )

    ax1.text(
        0.97,
        0.05,
        (
            f"AUC = {auc:.3f}\n"
            f"Limiar de Youden = {limite_youden:.3f}\n"
            f"Sensibilidade = {sensibilidade:.1%}\n"
            f"Especificidade = {especificidade:.1%}"
        ),
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
    )

    ax1.legend(
        loc="upper left",
        frameon=False,
    )

    ax1.grid(
        linestyle=":",
        alpha=0.30,
    )



    # Probabilidade
    ax2.axhspan(
        0,
        limite_youden,
        alpha=0.05,
    )

    ax2.axhspan(
        limite_youden,
        1,
        alpha=0.05,
    )

    ax2.vlines(
        x=dados_plot["POSICAO"],
        ymin=0,
        ymax=dados_plot["PROB_NORM"],
        linewidth=1,
        zorder=1,
    )

    ax2.scatter(
        dados_plot["POSICAO"],
        dados_plot["PROB_NORM"],
        c=cores_pontos,
        s=85,
        edgecolor=WHITE,
        linewidth=0.9,
        zorder=3,
    )

    ax2.axhline(
        limite_youden,
        linestyle="--",
        linewidth=1.7,
        label=f"Limiar de Youden = {limite_youden:.3f}",
    )

    ax2.set_xticks(
        dados_plot["POSICAO"]
    )

    ax2.set_xticklabels(
        dados_plot[
            "LOCAL DA GERAÇÃO"
        ],
        rotation=70,
        ha="right",
        fontsize=8,
    )

    ax2.set_ylim(
        0,
        1.03,
    )

    ax2.set_ylabel(
        "Probabilidade estimada de NORM"
    )

    ax2.set_xlabel(
        "Plataformas ordenadas pela "
        "probabilidade estimada"
    )

    ax2.set_title(
        "B. Probabilidade estimada por plataforma",
        fontsize=13,
        fontweight="bold",
        color=NAVY,
        loc="left",
    )

    legenda = [
    Patch(facecolor=RED, label="Verdadeiro positivo"),
    Patch(facecolor=GREEN, label="Verdadeiro negativo"),
    Patch(facecolor=ORANGE, label="Falso positivo"),
    Patch(facecolor=NAVY, label="Falso negativo"),
    Line2D(
        [0], [0],
        linestyle="--",
        linewidth=1.7,
        color="#1f77b4",
        label=f"Limiar de Youden = {limite_youden:.3f}",
    ),
    ]

    ax2.legend(
        handles=legenda,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),  # abaixo do gráfico
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(
        (
            "Regressão logística para classificação "
            "da ocorrência de NORM"
        ),
        fontsize=16,
        fontweight="bold",
        color=NAVY,
    )

    plt.tight_layout()

    if salvar_em:

        fig.savefig(
            salvar_em,
            dpi=300,
            bbox_inches="tight",
            facecolor=WHITE,
        )

    if exibir:
        plt.show()
    else:
        plt.close(fig)

    return fig

def grafico_superficie_probabilidade_2d(
    dados_modelo: pd.DataFrame,
    modelo: Pipeline,
    resultado_modelo: dict,
    salvar_em: str | None = None,
    exibir: bool = True,
):
    """Plota uma projeção 2D da probabilidade do modelo selecionado."""

    features = resultado_modelo["features"]
    if features == ["SALINIDADE", "RELACAO_BA_SR"]:
        coluna_x, coluna_y = "SALINIDADE", "RELACAO_BA_SR"
        rotulo_y = "Relação Bário/Estrôncio (Ba/Sr)"
        valores_fixos = {}
        complemento_titulo = ""
    elif features == ["SALINIDADE", "BARIO", "ESTRONCIO"]:
        coluna_x, coluna_y = "SALINIDADE", "BARIO"
        rotulo_y = "Bário (mg/L)"
        estroncio_fixo = float(dados_modelo["ESTRONCIO"].median())
        valores_fixos = {"ESTRONCIO": estroncio_fixo}
        complemento_titulo = (
            f" — Estrôncio fixado na mediana ({estroncio_fixo:.2f} mg/L)"
        )
    else:
        raise ValueError(
            f"Features não suportadas pela superfície 2D: {features}"
        )

    colunas = [
        "LOCAL DA GERAÇÃO",
        "TEM_NORM",
        *features,
    ]
    dados = (
        dados_modelo[colunas]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )
    dados = dados[
        dados[features].gt(0).all(axis=1)
    ]

    if dados.empty:
        raise ValueError("Nenhum dado positivo disponivel para o grafico Ba/Sr.")

    def limites_log(serie: pd.Series, margem: float = 0.12):
        minimo = np.log10(serie.min())
        maximo = np.log10(serie.max())
        amplitude = max(maximo - minimo, 0.2)
        return 10 ** (minimo - margem * amplitude), 10 ** (
            maximo + margem * amplitude
        )

    x_min, x_max = limites_log(dados[coluna_x])
    y_min, y_max = limites_log(dados[coluna_y])
    eixo_x = np.geomspace(x_min, x_max, 240)
    eixo_y = np.geomspace(y_min, y_max, 240)
    grade_x, grade_y = np.meshgrid(eixo_x, eixo_y)

    entrada_grade = pd.DataFrame({
        coluna_x: grade_x.ravel(),
        coluna_y: grade_y.ravel(),
    })
    for coluna, valor in valores_fixos.items():
        entrada_grade[coluna] = valor
    entrada_grade = entrada_grade[features]
    probabilidade = modelo.predict_proba(entrada_grade)[:, 1].reshape(
        grade_x.shape
    )
    limiar = float(resultado_modelo["limite_youden"])

    fig, ax = plt.subplots(figsize=(11, 8))
    niveis_preenchimento = np.linspace(0, 1, 11)
    superficie = ax.contourf(
        grade_x,
        grade_y,
        probabilidade,
        levels=niveis_preenchimento,
        cmap="Blues",
        alpha=0.42,
    )

    niveis_referencia = sorted({0.2, 0.4, 0.5, 0.6, 0.8})
    curvas = ax.contour(
        grade_x,
        grade_y,
        probabilidade,
        levels=niveis_referencia,
        colors="#5F6368",
        linestyles="--",
        linewidths=1.0,
    )
    ax.clabel(curvas, fmt=lambda valor: f"{valor:.0%}", fontsize=8)

    curva_youden = ax.contour(
        grade_x,
        grade_y,
        probabilidade,
        levels=[limiar],
        colors="#3F3F3F",
        linewidths=2.6,
    )
    rotulos_youden = ax.clabel(
        curva_youden,
        fmt={limiar: f"Corte ótimo: {limiar:.1%}"},
        fontsize=10,
    )
    for rotulo in rotulos_youden:
        rotulo.set_fontweight("bold")
        rotulo.set_color("#202020")
        rotulo.set_bbox({
            "boxstyle": "round,pad=0.28",
            "facecolor": WHITE,
            "edgecolor": "#3F3F3F",
            "linewidth": 0.8,
            "alpha": 0.95,
        })

    sem_norm = dados[dados["TEM_NORM"] == 0]
    com_norm = dados[dados["TEM_NORM"] == 1]
    ax.scatter(
        sem_norm[coluna_x],
        sem_norm[coluna_y],
        s=78,
        facecolor=WHITE,
        edgecolor="#404040",
        linewidth=1.4,
        label="Sem NORM",
        zorder=5,
    )
    ax.scatter(
        com_norm[coluna_x],
        com_norm[coluna_y],
        s=95,
        facecolor="#D62728",
        edgecolor=WHITE,
        linewidth=1.0,
        label="Com NORM",
        zorder=6,
    )

    for _, linha in dados.iterrows():
        ax.annotate(
            linha["LOCAL DA GERAÇÃO"],
            (linha[coluna_x], linha[coluna_y]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="#4D4D4D",
            zorder=7,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Salinidade (mg/L)")
    ax.set_ylabel(rotulo_y)
    ax.set_title(
        "Salinidade × " + rotulo_y.split(" (")[0]
        + " — probabilidade estimada de NORM"
        + complemento_titulo,
        fontweight="bold",
        color=NAVY,
        loc="left",
    )
    ax.grid(which="both", linestyle=":", alpha=0.18)
    ax.legend(loc="lower left", frameon=True)

    barra = fig.colorbar(superficie, ax=ax, pad=0.03)
    barra.set_label("Probabilidade estimada de NORM")
    barra.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    fig.tight_layout()
    if salvar_em:
        fig.savefig(salvar_em, dpi=300, bbox_inches="tight", facecolor=WHITE)
    if exibir:
        plt.show()
    else:
        plt.close(fig)

    return fig

def grafico_heatmap_metricas(
    resultado_modelos: pd.DataFrame,
    salvar_em: str | None = None,
    exibir: bool = True,
):

    metricas = [
        "AUC",
        "ACURACIA",
        "PRECISAO",
        "SENSIBILIDADE",
        "ESPECIFICIDADE",
        "F1_SCORE",
    ]

    dados = (
        resultado_modelos
        .set_index("MODELO")[
            metricas
        ]
        .sort_values(
            "PRECISAO",
            ascending=False,
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.heatmap(
        dados,
        annot=True,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        fmt=".2f",
        ax=ax,
    )

    ax.set_title(
        "Comparação do desempenho dos modelos",
        fontweight="bold",
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout()

    if salvar_em:

        fig.savefig(
            salvar_em,
            dpi=300,
            bbox_inches="tight",
        )

    if exibir:
        plt.show()
    else:
        plt.close(fig)

    return fig


def grafico_bsw_plataforma(
    df_analise: pd.DataFrame,
    salvar_em: str | None = None,
    exibir: bool = True,
    modo: str = "janela"
):
    """
    Gera gráfico de BSW por plataforma.

    Parameters
    ----------
    df_analise : pd.DataFrame
        Dataset gerado pelo pipeline.

    salvar_em : str | None, opcional
        Caminho para salvar a imagem.
        Se None, não salva.

    exibir : bool, padrão=True
        Define se o gráfico será exibido.

    modo : {"data", "janela"}, padrão="janela"
        Define o período considerado na análise.

        - "data":
          considera o primeiro registro do resíduo.

        - "janela":
          considera o período entre a primeira
          e a última ocorrência do resíduo.

    Returns
    -------
    matplotlib.figure.Figure
        Figura criada.
    """

    logger.info(
        "Plotando gráfico de BSW | modo=%s",
        modo
    )

    # ==========================================================
    # VALIDAÇÃO DO MODO
    # ==========================================================

    if modo not in {"data", "janela"}:
        raise ValueError(
            "modo deve ser 'data' ou 'janela'"
        )

    # ==========================================================
    # TEXTOS DO GRÁFICO
    # ==========================================================

    if modo == "data":

        ylabel = "BSW no primeiro registro (%)"

        titulo = (
            "BSW por plataforma — "
            "primeiro registro do resíduo"
        )

    else:

        ylabel = "BSW médio no período (%)"

        titulo = (
            "BSW por plataforma — "
            "período de ocorrência do resíduo"
        )

    # ==========================================================
    # SEPARAÇÃO DOS GRUPOS
    # ==========================================================

    df_norm = (
        df_analise[
            df_analise["TEM_NORM"] == 1
        ]
        .copy()
    )

    df_oleosa = (
        df_analise[
            (df_analise["TEM_OLEOSA"] == 1)
            &
            (df_analise["TEM_NORM"] == 0)
        ]
        .copy()
    )

    df_plat = pd.concat(
        [
            df_norm.assign(
                GRUPO="NORM"
            ),
            df_oleosa.assign(
                GRUPO="OLEOSA"
            )
        ],
        ignore_index=True
    )

    # ==========================================================
    # ORDENAÇÃO
    # ==========================================================

    df_plat = (
        df_plat
        .sort_values(
            "MEDIANA_BSW_PLAT"
        )
        .reset_index(
            drop=True
        )
    )

    # ==========================================================
    # REFERÊNCIA AUTOMÁTICA
    # ==========================================================

    bsw_referencia = (
        df_norm["MEDIA_BSW_PLAT"]
        .dropna()
        .min()
    )

    logger.info(
        "Referência BSW encontrada | %.2f%%",
        bsw_referencia
    )

    # ==========================================================
    # CORES
    # ==========================================================

    cores = {
        "NORM": "crimson",
        "OLEOSA": "forestgreen"
    }

    # ==========================================================
    # FIGURA
    # ==========================================================

    fig, ax = plt.subplots(
        figsize=(28, 8)
    )

    ax.bar(
        df_plat["LOCAL DA GERAÇÃO"],
        df_plat["MEDIA_BSW_PLAT"],
        color=df_plat["GRUPO"].map(
            cores
        ),
        edgecolor="white",
        linewidth=0.7
    )

    # ==========================================================
    # LINHA DE REFERÊNCIA
    # ==========================================================

    if pd.notna(bsw_referencia):

        ax.axhline(
            y=bsw_referencia,
            color="black",
            linestyle="--",
            linewidth=2
        )

    # ==========================================================
    # EIXOS E TÍTULO
    # ==========================================================

    ax.set_ylabel(
        ylabel,
        fontsize=12
    )

    ax.set_title(
        titulo,
        fontsize=16,
        fontweight="bold"
    )

    ax.grid(
        axis="y",
        linestyle=":",
        alpha=0.4
    )

    ax.yaxis.set_major_formatter(
        mticker.FormatStrFormatter(
            "%.0f%%"
        )
    )

    ax.margins(
        x=0
    )

    # ==========================================================
    # LEGENDA
    # ==========================================================

    handles = [
        Patch(
            color="crimson",
            label="NORM"
        ),

        Patch(
            color="forestgreen",
            label="Oleosa"
        )
    ]

    if pd.notna(bsw_referencia):

        handles.append(
            plt.Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=2,
                label=(
                    f"Referência "
                    f"({bsw_referencia:.1f}%)"
                )
            )
        )

    ax.legend(
        handles=handles,
        fontsize=10,
        loc="upper left"
    )

    # ==========================================================
    # RÓTULOS DO EIXO X
    # ==========================================================

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        fontsize=11
    )

    plt.subplots_adjust(
        bottom=0.20,
        left=0.06,
        right=0.94,
        top=0.90
    )

    # ==========================================================
    # SALVAR
    # ==========================================================

    if salvar_em is not None:

        fig.savefig(
            salvar_em,
            dpi=300,
            bbox_inches="tight"
        )

    # ==========================================================
    # EXIBIR
    # ==========================================================

    if exibir:
        plt.show()

    else:
        plt.close(fig)

    logger.info(
        "Gráfico de BSW finalizado"
    )

    return fig

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scipy.interpolate import griddata
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)


def grafico_quimica_superficie_3d(
    df_analise: pd.DataFrame,
    salvar_em: str | None = None,
    exibir: bool = True,
):
    """
    Gera superfície 3D interpolada das variáveis químicas padronizadas.

    Eixos:
        X = Salinidade
        Y = Relação Bário/Estrôncio
        Z = BSW

    A superfície representa uma interpolação dos valores observados.
    Os pontos reais das plataformas são adicionados sobre a superfície.

    Parameters
    ----------
    df_analise : pd.DataFrame
        DataFrame de análise.

    salvar_em : str | None, padrão=None
        Caminho para salvar o gráfico em HTML.

    exibir : bool, padrão=True
        Se True, exibe o gráfico.

    Returns
    -------
    plotly.graph_objects.Figure
        Figura Plotly criada.
    """

    # ======================================================
    # VARIÁVEIS
    # ======================================================

    variaveis = [
        "MEDIANA_SALINIDADE_PLAT",
        "RELACAO_BARIO_ESTRONCIO",
        "MEDIA_BSW_PLAT",
    ]

    dados = (
        df_analise[
            [
                "LOCAL DA GERAÇÃO",
                "TEM_NORM",
                "PROB_NORM",
                "MASSA_CLASSIFICADA_KG",
                "TIPO_MASSA",
                *variaveis,
            ]
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .copy()
    )

    dados["NORM"] = (
        dados["TEM_NORM"]
        .astype(int)
        .map(
            {
                1: "Com NORM",
                0: "Sem NORM",
            }
        )
    )

    dados["MASSA_CLASSIFICADA_KG"] = dados["MASSA_CLASSIFICADA_KG"].clip(lower=0)
    dados["MASSA_CLASSIFICADA_T"] = dados["MASSA_CLASSIFICADA_KG"] / 1000
    log_massa = np.log1p(dados["MASSA_CLASSIFICADA_KG"])
    dados["TAMANHO_MARCADOR"] = (
        6 + 22 * log_massa / max(log_massa.max(), 1)
    )

    # ======================================================
    # PADRONIZAÇÃO
    # ======================================================

    scaler = StandardScaler()

    colunas_z = [
        "SALINIDADE_Z",
        "RELACAO_BA_SR_Z",
        "BSW_Z",
    ]

    dados[colunas_z] = scaler.fit_transform(
        dados[variaveis]
    )

    # ======================================================
    # DADOS PARA SUPERFÍCIE
    # ======================================================

    x = dados["SALINIDADE_Z"].to_numpy()
    y = dados["RELACAO_BA_SR_Z"].to_numpy()
    z = dados["BSW_Z"].to_numpy()

    # Grade regular para interpolação
    grid_x = np.linspace(
        x.min(),
        x.max(),
        80,
    )

    grid_y = np.linspace(
        y.min(),
        y.max(),
        80,
    )

    X, Y = np.meshgrid(
        grid_x,
        grid_y,
    )

    # ======================================================
    # INTERPOLAÇÃO
    # ======================================================

    Z = griddata(
        points=(x, y),
        values=z,
        xi=(X, Y),
        method="linear",
    )



    # ======================================================
    # GRÁFICO
    # ======================================================

    fig = go.Figure()

    # ------------------------------------------------------
    # SUPERFÍCIE
    # ------------------------------------------------------

    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,

            colorscale="Viridis",

            opacity=0.75,
            showscale=False,

            colorbar=dict(
                title=dict(
                    text="BSW (z-score)",
                    side="top",
                ),
                x=1.08,
                y=0.42,
                len=0.65,
                thickness=20,
            ),

            hovertemplate=(
                "Salinidade: %{x:.2f}<br>"
                "Relação Ba/Sr: %{y:.2f}<br>"
                "BSW estimado: %{z:.2f}"
                "<extra></extra>"
            ),

            name="Superfície química",
        )
    )

    # ======================================================
    # PONTOS REAIS — SEM NORM
    # ======================================================

    sem_norm = dados[
        dados["TEM_NORM"] == 0
    ]

    fig.add_trace(
        go.Scatter3d(
            x=sem_norm["SALINIDADE_Z"],
            y=sem_norm["RELACAO_BA_SR_Z"],
            z=sem_norm["BSW_Z"],

            mode="markers",

            marker=dict(
                size=sem_norm["TAMANHO_MARCADOR"],
                color=sem_norm["PROB_NORM"],
                colorscale="RdYlBu_r",
                cmin=0,
                cmax=1,
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="Probabilidade de NORM",
                        side="right",
                    ),
                    tickformat=".0%",
                    x=1.08,
                    y=0.38,
                    len=0.55,
                    thickness=20,
                ),
                symbol="circle",
                line=dict(
                    color="white",
                    width=1,
                ),
            ),

            text=sem_norm["LOCAL DA GERAÇÃO"],

            customdata=np.column_stack(
                (
                    sem_norm[
                        "MEDIANA_SALINIDADE_PLAT"
                    ],
                    sem_norm[
                        "RELACAO_BARIO_ESTRONCIO"
                    ],
                    sem_norm[
                        "MEDIA_BSW_PLAT"
                    ],
                    sem_norm["MASSA_CLASSIFICADA_T"],
                    sem_norm["PROB_NORM"],
                )
            ),

            hovertemplate=(
                "<b>%{text}</b><br>"
                "Sem NORM<br><br>"
                "Salinidade: %{customdata[0]:.2f}<br>"
                "Relação Ba/Sr: %{customdata[1]:.4f}<br>"
                "BSW: %{customdata[2]:.2f}%<br>"
                "Borra oleosa: %{customdata[3]:,.2f} t<br>"
                "Probabilidade de NORM: %{customdata[4]:.1%}"
                "<extra></extra>"
            ),

            name="Sem NORM",
        )
    )

    # ======================================================
    # PONTOS REAIS — COM NORM
    # ======================================================

    com_norm = dados[
        dados["TEM_NORM"] == 1
    ]

    fig.add_trace(
        go.Scatter3d(
            x=com_norm["SALINIDADE_Z"],
            y=com_norm["RELACAO_BA_SR_Z"],
            z=com_norm["BSW_Z"],

            mode="markers",

            marker=dict(
                size=com_norm["TAMANHO_MARCADOR"],
                color=com_norm["PROB_NORM"],
                colorscale="RdYlBu_r",
                cmin=0,
                cmax=1,
                showscale=False,
                symbol="diamond",
                line=dict(
                    color="white",
                    width=1,
                ),
            ),

            text=com_norm["LOCAL DA GERAÇÃO"],

            customdata=np.column_stack(
                (
                    com_norm[
                        "MEDIANA_SALINIDADE_PLAT"
                    ],
                    com_norm[
                        "RELACAO_BARIO_ESTRONCIO"
                    ],
                    com_norm[
                        "MEDIA_BSW_PLAT"
                    ],
                    com_norm["MASSA_CLASSIFICADA_T"],
                    com_norm["PROB_NORM"],
                )
            ),

            hovertemplate=(
                "<b>%{text}</b><br>"
                "Com NORM<br><br>"
                "Salinidade: %{customdata[0]:.2f}<br>"
                "Relação Ba/Sr: %{customdata[1]:.4f}<br>"
                "BSW: %{customdata[2]:.2f}%<br>"
                "Borra com NORM: %{customdata[3]:,.2f} t<br>"
                "Probabilidade de NORM: %{customdata[4]:.1%}"
                "<extra></extra>"
            ),

            name="Com NORM",
        )
    )

    # ======================================================
    # LAYOUT
    # ======================================================
    fig.update_layout(

        title=(
            "Superfície química — "
            "Salinidade × Relação Ba/Sr × BSW"
        ),

        width=1100,
        height=800,

        scene=dict(

            xaxis_title="Salinidade padronizada (z-score)",

            yaxis_title="Relação Ba/Sr padronizada (z-score)",

            zaxis_title="BSW padronizado (z-score)",

            aspectmode="cube",

            camera=dict(
                eye=dict(
                    x=1.6,
                    y=1.6,
                    z=1.2,
                )
            ),
        ),

        # ==========================================
        # LEGENDA NORM
        # ==========================================

        legend=dict(
            title=dict(
                text="Classificação"
            ),

            x=1.02,
            y=1.0,

            xanchor="left",
            yanchor="top",

            bgcolor="rgba(255,255,255,0.8)",

            bordercolor="lightgray",
            borderwidth=1,
        ),

        # Mais espaço do lado direito
        margin=dict(
            l=20,
            r=240,
            b=20,
            t=70,
        ),
    )

    # ======================================================
    # SALVAR
    # ======================================================

    if salvar_em is not None:

        fig.write_html(
            str(salvar_em)
        )

        logger.info(
            "Superfície química 3D salva em %s",
            salvar_em,
        )

    # ======================================================
    # EXIBIR
    # ======================================================

    if exibir:
        fig.show()

    return fig
