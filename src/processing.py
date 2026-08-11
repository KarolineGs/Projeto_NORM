import sys
import os
import logging
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt
import matplotlib.pyplot as plt
from unidecode import unidecode
import logging

logger =  logging.getLogger()


def media_ponderada(x, valor, peso):
    dados = x[[valor, peso]].dropna()

    if dados.empty or dados[peso].sum() == 0:
        return np.nan

    return np.average(
        dados[valor],
        weights=dados[peso]
    )

def agrupar_residuos_plataforma(
    
    df: pd.DataFrame,
    colunas: list[str] = ['LOCAL DA GERAÇÃO', 'TIPO DE RESÍDUO'],
    colunas_valor: list = ["MASSA DO VOLUME (kg)"]
) -> pd.DataFrame:
    """Agrupa os resíduos por plataforma e tipo, somando a massa."""

    logger.info(
        "Agrupando resíduos pelas colunas: %s",
        colunas
    )

    df = df.copy()

    df_agrupado = (
        df
        .groupby(
            colunas,
            as_index=False
        )[colunas_valor]
        .sum()
    )

    logger.info(
        "Agrupamento finalizado | %d registros",
        len(df_agrupado)
    )

    return df_agrupado
def criar_flags_residuos_plataforma(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Cria flags de NORM e borra oleosa por plataforma."""

    logger.info(
        "Criando flags de resíduos por plataforma"
    )

    plataformas = (
        df
        .groupby("LOCAL DA GERAÇÃO")
        .agg(
            TEM_NORM=(
                "TIPO DE RESÍDUO",
                lambda x: x.str.contains(
                    "BORRA OLEOSA COM NORM",
                    case=False,
                    na=False
                ).any()
            ),
            TEM_OLEOSA=(
                "TIPO DE RESÍDUO",
                lambda x: (
                    x.str.contains(
                        "BORRA OLEOSA",
                        case=False,
                        na=False
                    )
                    &
                    ~x.str.contains(
                        "NORM",
                        case=False,
                        na=False
                    )
                ).any()
            ),
            MASSA_TOTAL=(
                "MASSA DO VOLUME (kg)",
                "sum"
            )
        )
        .reset_index()

    )

    plataformas["TEM_NORM"] = (
        plataformas["TEM_NORM"].astype(int)
    )

    plataformas["TEM_OLEOSA"] = (
        plataformas["TEM_OLEOSA"].astype(int)
    )

    plataformas.loc[
        plataformas["TEM_NORM"] == 1,
        "TEM_OLEOSA"
    ] = 0

    logger.info(
        "Flags criadas | %d plataformas | %d com NORM",
        len(plataformas),
        plataformas["TEM_NORM"].sum()
    )

    return plataformas
def filtrar_periodo_sigre(
    scr_sigre: pd.DataFrame,
    df_fenix: pd.DataFrame,
    plataformas: pd.DataFrame,
    modo: str = "data"
) -> pd.DataFrame:

    sigre = scr_sigre.copy()
    fenix = df_fenix.copy()

    sigre["Mes"] = pd.to_datetime(
        sigre["Mes"],
        errors="coerce"
    )

    fenix["Mes"] = pd.to_datetime(
        fenix["Mes"],
        errors="coerce"
    )

    # Se no Fênix o nome ainda for PLATAFORMA
    if "PLATAFORMA" in fenix.columns:
        fenix = fenix.rename(
            columns={
                "PLATAFORMA": "LOCAL DA GERAÇÃO"
            }
        )

    sigre = sigre.merge(
        plataformas[
            [
                "LOCAL DA GERAÇÃO",
                "TEM_NORM",
                "TEM_OLEOSA"
            ]
        ],
        on="LOCAL DA GERAÇÃO",
        how="inner"
    )

    eh_norm = (
        sigre["TIPO DE RESÍDUO"]
        .str.contains(
            "BORRA OLEOSA COM NORM",
            case=False,
            na=False
        )
    )

    eh_oleosa = (
        sigre["TIPO DE RESÍDUO"]
        .str.contains(
            "BORRA OLEOSA",
            case=False,
            na=False
        )
        &
        ~sigre["TIPO DE RESÍDUO"]
        .str.contains(
            "NORM",
            case=False,
            na=False
        )
    )

    mascara = (
        (
            (sigre["TEM_NORM"] == 1)
            &
            eh_norm
        )
        |
        (
            (sigre["TEM_OLEOSA"] == 1)
            &
            eh_oleosa
        )
    )

    sigre_periodo = (
        sigre
        .loc[mascara]
        .copy()
    )

    # CRIADO AQUI
    periodos_sigre = (
        sigre_periodo
        .groupby("LOCAL DA GERAÇÃO")
        .agg(
            DATA_MIN=("Mes", "min"),
            DATA_MAX=("Mes", "max")
        )
        .reset_index()
    )

    df_agua_periodo = (
        fenix
        .merge(
            periodos_sigre,
            on="LOCAL DA GERAÇÃO",
            how="inner"
        )
    )

    if modo == "data":

        candidatos = (
            df_agua_periodo
            .copy()
        )

        # Diferença com sinal:
        # negativo = antes do primeiro registro
        # zero = mesmo mês
        # positivo = depois do primeiro registro
        candidatos["DIFERENCA_MESES"] = (
            (
                candidatos["Mes"].dt.year
                -
                candidatos["DATA_MIN"].dt.year
            ) * 12
            +
            (
                candidatos["Mes"].dt.month
                -
                candidatos["DATA_MIN"].dt.month
            )
        )

        # Distância absoluta para encontrar
        # o mês mais próximo
        candidatos["DISTANCIA_MESES"] = (
            candidatos[
                "DIFERENCA_MESES"
            ]
            .abs()
        )

        # Seleciona um único mês por plataforma.
        #
        # Primeiro critério:
        # menor distância da DATA_MIN.
        #
        # Segundo critério, em caso de empate:
        # prioriza o mês anterior, pois o valor negativo
        # aparece antes do positivo na ordenação.
        mes_selecionado = (
            candidatos[
                [
                    "LOCAL DA GERAÇÃO",
                    "Mes",
                    "DISTANCIA_MESES",
                    "DIFERENCA_MESES"
                ]
            ]
            .drop_duplicates()
            .sort_values(
                [
                    "LOCAL DA GERAÇÃO",
                    "DISTANCIA_MESES",
                    "DIFERENCA_MESES"
                ]
            )
            .drop_duplicates(
                subset="LOCAL DA GERAÇÃO",
                keep="first"
            )
            .rename(
                columns={
                    "Mes": "MES_SELECIONADO"
                }
            )
        )

        # Mantém todos os registros do Fênix
        # que pertencem ao mês selecionado.
        #
        # Isso evita excluir outros poços da mesma
        # plataforma existentes naquele mês.
        df_agua_periodo = (
            candidatos
            .merge(
                mes_selecionado[
                    [
                        "LOCAL DA GERAÇÃO",
                        "MES_SELECIONADO"
                    ]
                ],
                on="LOCAL DA GERAÇÃO",
                how="inner"
            )
        )

        df_agua_periodo = (
            df_agua_periodo[
                df_agua_periodo["Mes"]
                ==
                df_agua_periodo["MES_SELECIONADO"]
            ]
            .copy()
        )

    elif modo == "janela":

        df_agua_periodo = (
            df_agua_periodo[
                (
                    df_agua_periodo["Mes"]
                    >= df_agua_periodo["DATA_MIN"]
                )
                &
                (
                    df_agua_periodo["Mes"]
                    <= df_agua_periodo["DATA_MAX"]
                )
            ]
            .copy()
        )

    else:
        raise ValueError(
            "modo deve ser 'data' ou 'janela'"
        )

    return df_agua_periodo

def agregar_mensal_plataforma(df_agua_periodo):
    """
    Agrega os dados no nivel PLATAFORMA + Mes.
    """

    mensal_plataforma = (
        df_agua_periodo
        .groupby(
            ['LOCAL DA GERAÇÃO', 'Mes'],
            group_keys=False
        )
        .apply(
            lambda g: pd.Series({

                'AGUA_PLAT':
                    g['QW_mensal_m3'].sum(),

                'OLEO_PLAT':
                    g['QO_mensal_m3'].sum(),

                'BARIO_PLAT':
                    media_ponderada(
                        g,
                        'BARIO',
                        'QW_mensal_m3'
                    ),

                'ESTRONCIO_PLAT':
                    media_ponderada(
                        g,
                        'ESTRONCIO',
                        'QW_mensal_m3'
                    ),

                'SALINIDADE_PLAT':
                    media_ponderada(
                        g,
                        'SALINIDADE',
                        'QW_mensal_m3'
                    ),

                'SULFATO_PLAT':
                    media_ponderada(
                        g,
                        'SULFATO',
                        'QW_mensal_m3'
                    )
            })
        )
        .reset_index()
    )

    mensal_plataforma['BSW_PLAT'] = (
        mensal_plataforma['AGUA_PLAT']
        /
        (
            mensal_plataforma['AGUA_PLAT']
            +
            mensal_plataforma['OLEO_PLAT']
        )
    ) * 100

    return mensal_plataforma

def gerar_estatisticas_plataforma(mensal_plataforma):
    """
    Resume os dados mensais para uma linha por plataforma.
    """

    estatisticas_plataforma = (
        mensal_plataforma
        .groupby('LOCAL DA GERAÇÃO')
        .agg(

            # ÁGUA
            MEDIANA_AGUA_PLAT=(
                'AGUA_PLAT',
                'median'
            ),

            MEDIA_AGUA_PLAT=(
                'AGUA_PLAT',
                'mean'
            ),

            SOMA_AGUA_PLAT=(
                'AGUA_PLAT',
                'sum'
            ),

            # BSW
            MEDIA_BSW_PLAT=(
                'BSW_PLAT',
                'mean'
            ),

            MEDIANA_BSW_PLAT=(
                'BSW_PLAT',
                'median'
            ),

            P25_BSW_PLAT=(
                'BSW_PLAT',
                lambda x: x.quantile(0.25)
            ),

            P75_BSW_PLAT=(
                'BSW_PLAT',
                lambda x: x.quantile(0.75)
            ),

            STD_BSW_PLAT=(
                'BSW_PLAT',
                'std'
            ),

            # BÁRIO
            MEDIANA_BARIO_PLAT=(
                'BARIO_PLAT',
                'median'
            ),

            MEDIA_BARIO_PLAT=(
                'BARIO_PLAT',
                'mean'
            ),

            STD_BARIO_PLAT=(
                'BARIO_PLAT',
                'std'
            ),

            # ESTRÔNCIO
            MEDIANA_ESTRONCIO_PLAT=(
                'ESTRONCIO_PLAT',
                'median'
            ),

            MEDIA_ESTRONCIO_PLAT=(
                'ESTRONCIO_PLAT',
                'mean'
            ),

            STD_ESTRONCIO_PLAT=(
                'ESTRONCIO_PLAT',
                'std'
            ),

            # SALINIDADE
            MEDIANA_SALINIDADE_PLAT=(
                'SALINIDADE_PLAT',
                'median'
            ),

            MEDIA_SALINIDADE_PLAT=(
                'SALINIDADE_PLAT',
                'mean'
            ),

            STD_SALINIDADE_PLAT=(
                'SALINIDADE_PLAT',
                'std'
            ),

            # SULFATO
            MEDIANA_SULFATO_PLAT=(
                'SULFATO_PLAT',
                'median'
            ),

            MEDIA_SULFATO_PLAT=(
                'SULFATO_PLAT',
                'mean'
            ),

            STD_SULFATO_PLAT=(
                'SULFATO_PLAT',
                'std'
            )
        )
        .reset_index()
    )

    return estatisticas_plataforma

def gerar_df_analise(
    plataformas,
    estatisticas_plataforma
):
    """
    Junta as informações das plataformas
    com as features calculadas.
    """

    df_analise = (
        plataformas
        .merge(
            estatisticas_plataforma,
            on='LOCAL DA GERAÇÃO',
            how='left'
        )
    )

    df_analise = (
        df_analise
        .dropna(
            subset=[
                'MEDIA_BSW_PLAT'
            ]
        )
        .copy()
    )

    return df_analise

def processar_df_analise(
    scr_sigre: pd.DataFrame,
    df_fenix: pd.DataFrame,
    modo: str = "data"
) -> pd.DataFrame:

    logger.info(
        "Carregando bases processadas"
    )

    # scr_sigre = pd.read_csv(
    #     r'C:/Users/ALEX/Desktop/Projeto_NORM/data/processed/base_integrada_residuos.csv'
    # )

    # df_fenix = pd.read_csv(
    #     r'C:/Users/ALEX/Desktop/Projeto_NORM/data/processed/fenix_processado.csv'
    # )

    logger.info(
        "Iniciando criação do dataframe de análise | modo=%s",
        modo
    )

    # 1. Classificação
    plataformas = criar_flags_residuos_plataforma(
        scr_sigre
    )

    # 2. Recorte temporal
    df_agua_periodo = filtrar_periodo_sigre(
        scr_sigre=scr_sigre,
        df_fenix=df_fenix,
        plataformas=plataformas,
        modo=modo
    )

    # 3. Agregação mensal
    mensal_plataforma = agregar_mensal_plataforma(
        df_agua_periodo
    )

    # 4. Estatísticas
    estatisticas_plataforma = gerar_estatisticas_plataforma(
        mensal_plataforma
    )

    # 5. Dataset final
    df_analise = gerar_df_analise(
        plataformas=plataformas,
        estatisticas_plataforma=estatisticas_plataforma
    )

    logger.info(
        "Dataframe de análise criado | %d plataformas",
        len(df_analise)
    )

    return df_analise

