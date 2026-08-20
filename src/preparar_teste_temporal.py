import logging
from pathlib import Path

import pandas as pd


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# CAMINHOS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PASTA_PROCESSED = (
    BASE_DIR
    / "data"
    / "processed"
)

PASTA_BATCH_INPUT = (
    BASE_DIR
    / "data"
    / "batch"
    / "input"
)


# ============================================================
# FEATURES DO MODELO
# ============================================================

FEATURES_MODELO = [
    "SALINIDADE",
    "BARIO",
    "ESTRONCIO"
]


# ============================================================
# DATASET TEMPORAL
# ============================================================

def criar_dataset_teste_temporal(
    df_fenix: pd.DataFrame,
    scr_sigre: pd.DataFrame
) -> pd.DataFrame:

    """
    Cria dataset temporal para validação do modelo NORM.

    Granularidade:
        PLATAFORMA + MÊS

    O dataset contém:

        - variáveis químicas;
        - produção;
        - ocorrência real de NORM no mês;
        - primeira ocorrência de NORM;
        - distância temporal até o primeiro NORM;
        - flag indicando se o registro pode ser enviado
          ao modelo.
    """

    logger.info(
        "Criando dataset temporal para validação do modelo NORM"
    )

    fenix = df_fenix.copy()
    residuos = scr_sigre.copy()

    logger.info(
        "FENIX recebido | %d linhas x %d colunas",
        fenix.shape[0],
        fenix.shape[1]
    )

    logger.info(
        "Base de resíduos recebida | %d linhas x %d colunas",
        residuos.shape[0],
        residuos.shape[1]
    )

    # ========================================================
    # 1. CONVERTE DATAS
    # ========================================================

    fenix["Mes"] = pd.to_datetime(
        fenix["Mes"],
        errors="coerce"
    )

    residuos["Mes"] = pd.to_datetime(
        residuos["Mes"],
        errors="coerce"
    )

    # ========================================================
    # 2. IDENTIFICA NORM NOS REGISTROS DE RESÍDUO
    # ========================================================

    tipo_residuo = (
        residuos["TIPO DE RESÍDUO"]
        .astype("string")
        .str.strip()
    )

    eh_borra_com_norm = tipo_residuo.str.contains(
        "BORRA OLEOSA COM NORM",
        case=False,
        na=False,
        regex=False,
    )

    eh_borra_oleosa = (
        tipo_residuo.str.contains(
            "BORRA OLEOSA",
            case=False,
            na=False,
            regex=False,
        )
        & ~tipo_residuo.str.contains(
            "NORM",
            case=False,
            na=False,
            regex=False,
        )
    )

    # Outros tipos de resíduo não definem o alvo do modelo.
    residuos = residuos.loc[
        eh_borra_com_norm | eh_borra_oleosa
    ].copy()

    residuos["TEM_NORM_MES"] = (
        eh_borra_com_norm.loc[residuos.index]
        .astype(int)
    )

    # ========================================================
    # 3. CONSOLIDA NORM POR PLATAFORMA + MÊS
    # ========================================================

    norm_mensal = (
        residuos
        .groupby(
            [
                "LOCAL DA GERAÇÃO",
                "Mes"
            ],
            as_index=False
        )
        .agg(
            TEM_NORM_MES=(
                "TEM_NORM_MES",
                "max"
            )
        )
    )

    classificacao_plataforma = (
        residuos
        .groupby(
            "LOCAL DA GERAÇÃO",
            as_index=False,
        )
        .agg(
            TEM_NORM=(
                "TEM_NORM_MES",
                "max",
            )
        )
    )

    logger.info(
        "Base mensal de NORM criada | %d registros",
        len(norm_mensal)
    )

    # ========================================================
    # 4. PRIMEIRO REGISTRO DE NORM DA PLATAFORMA
    # ========================================================

    primeiro_norm = (
        norm_mensal[
            norm_mensal["TEM_NORM_MES"] == 1
        ]
        .groupby(
            "LOCAL DA GERAÇÃO",
            as_index=False
        )
        .agg(
            DATA_PRIMEIRO_NORM=(
                "Mes",
                "min"
            )
        )
    )

    logger.info(
        "Primeira ocorrência de NORM identificada para %d plataformas",
        len(primeiro_norm)
    )

    # ========================================================
    # 5. SELECIONA COLUNAS DO FÊNIX
    # ========================================================

    colunas_fenix = [
        "LOCAL DA GERAÇÃO",
        "Mes",
        "SALINIDADE",
        "BARIO",
        "ESTRONCIO",
        "QW_mensal_m3",
        "QO_mensal_m3",
        "BSW",
        "N_POCOS"
    ]

    # Evita erro caso alguma coluna operacional não exista
    colunas_fenix = [
        coluna
        for coluna in colunas_fenix
        if coluna in fenix.columns
    ]

    dataset = (
        fenix[
            colunas_fenix
        ]
        .copy()
    )

    dataset["RELACAO_BA_SR"] = (
        dataset["BARIO"]
        .div(dataset["ESTRONCIO"])
        .where(dataset["ESTRONCIO"] > 0)
    )

    # ========================================================
    # 6. JUNTA A OCORRÊNCIA REAL DE NORM NO MÊS
    # ========================================================

    dataset = dataset.merge(
        norm_mensal,
        on=[
            "LOCAL DA GERAÇÃO",
            "Mes"
        ],
        how="left"
    )

    # Ausência no SIGRE/SCR = nenhum NORM registrado naquele mês
    dataset["TEM_NORM_MES"] = (
        dataset["TEM_NORM_MES"]
        .astype("Int64")
    )

    # ========================================================
    # 7. JUNTA DATA DO PRIMEIRO NORM
    # ========================================================

    dataset = dataset.merge(
        primeiro_norm,
        on="LOCAL DA GERAÇÃO",
        how="left"
    )

    # ========================================================
    # 8. TEM_NORM
    # ========================================================
    #
    # Indica se a plataforma apresentou NORM em algum momento
    # da série histórica.
    #

    dataset = dataset.merge(
        classificacao_plataforma,
        on="LOCAL DA GERAÇÃO",
        how="left",
    )

    dataset["TEM_NORM"] = dataset["TEM_NORM"].astype("Int64")

    # ========================================================
    # 9. PRIMEIRO_NORM
    # ========================================================
    #
    # Vale 1 somente no mês da primeira ocorrência registrada.
    #

    dataset["PRIMEIRO_NORM"] = (
        dataset["Mes"]
        ==
        dataset["DATA_PRIMEIRO_NORM"]
    ).astype(int)

    # ========================================================
    # 10. MESES ATÉ O PRIMEIRO NORM
    # ========================================================
    #
    # Exemplo:
    #
    #   6   -> seis meses antes do primeiro NORM
    #   1   -> um mês antes
    #   0   -> mês do primeiro NORM
    #  -1   -> um mês depois
    #  -6   -> seis meses depois
    #
    # Para plataformas sem NORM ficará NaN.
    #

    dataset["MESES_ATE_NORM"] = (
        (
            dataset["DATA_PRIMEIRO_NORM"].dt.year
            -
            dataset["Mes"].dt.year
        ) * 12
        +
        (
            dataset["DATA_PRIMEIRO_NORM"].dt.month
            -
            dataset["Mes"].dt.month
        )
    )

    # ========================================================
    # 11. ELEGIBILIDADE PARA O MODELO
    # ========================================================
    #
    # O modelo só pode receber registros que possuam
    # SALINIDADE e relação BARIO/ESTRONCIO.
    #

    dataset["ELEGIVEL_MODELO"] = (
        dataset[
            FEATURES_MODELO
        ]
        .notna()
        .all(axis=1)
    )

    # ========================================================
    # 12. ORDENAÇÃO
    # ========================================================

    dataset = (
        dataset
        .sort_values(
            [
                "LOCAL DA GERAÇÃO",
                "Mes"
            ]
        )
        .reset_index(drop=True)
    )

    logger.info(
        "Dataset temporal criado | %d registros",
        len(dataset)
    )

    logger.info(
        "Registros elegíveis para o modelo | %d",
        dataset["ELEGIVEL_MODELO"].sum()
    )

    logger.info(
        "Plataformas com NORM | %d",
        dataset.loc[
            dataset["TEM_NORM"] == 1,
            "LOCAL DA GERAÇÃO"
        ].nunique()
    )

    return dataset


# ============================================================
# ENTRADA PARA INFERÊNCIA BATCH
# ============================================================

def gerar_entrada_batch(
    dataset_teste: pd.DataFrame
) -> pd.DataFrame:

    """
    Cria a entrada que será enviada para inferência batch.

    O CSV mantém informações temporais e do NORM real
    para permitir validação posterior.

    IMPORTANTE:
    O modelo deve utilizar exclusivamente:

        SALINIDADE
        RELACAO_BA_SR

    As demais colunas são apenas contexto e validação.
    """

    logger.info(
        "Gerando entrada para inferência batch"
    )

    # ========================================================
    # COLUNAS DE CONTEXTO
    # ========================================================

    colunas_contexto = [
        "LOCAL DA GERAÇÃO",
        "Mes",
        "TEM_NORM",
        "TEM_NORM_MES",
        "PRIMEIRO_NORM",
        "DATA_PRIMEIRO_NORM",
        "MESES_ATE_NORM",
        "ELEGIVEL_MODELO"
    ]

    # Mantém somente as colunas existentes
    colunas_contexto = [
        coluna
        for coluna in colunas_contexto
        if coluna in dataset_teste.columns
    ]

    colunas_saida = (
        colunas_contexto
        +
        FEATURES_MODELO
    )

    # ========================================================
    # SOMENTE REGISTROS COM AS 2 FEATURES
    # ========================================================

    entrada_batch = (
        dataset_teste
        .loc[
            dataset_teste["ELEGIVEL_MODELO"],
            colunas_saida
        ]
        .copy()
    )

    # Nome mais conveniente para inferência
    entrada_batch = entrada_batch.rename(
        columns={
            "LOCAL DA GERAÇÃO": "PLATAFORMA"
        }
    )

    entrada_batch = (
        entrada_batch
        .sort_values(
            [
                "PLATAFORMA",
                "Mes"
            ]
        )
        .reset_index(drop=True)
    )

    logger.info(
        "Entrada batch criada | %d registros",
        len(entrada_batch)
    )

    return entrada_batch


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        )
    )

    logger.info(
        "Iniciando preparação do dataset temporal"
    )

    # ========================================================
    # 1. CARREGA BASES PROCESSADAS
    # ========================================================

    df_fenix = pd.read_csv(
        PASTA_PROCESSED
        / "fenix_processado.csv"
    )

    scr_sigre = pd.read_csv(
        PASTA_PROCESSED
        / "base_integrada_residuos.csv"
    )

    # ========================================================
    # 2. CRIA DATASET TEMPORAL COMPLETO
    # ========================================================

    df_teste_temporal = criar_dataset_teste_temporal(
        df_fenix=df_fenix,
        scr_sigre=scr_sigre
    )

    # ========================================================
    # 3. CRIA ENTRADA PARA O BATCH
    # ========================================================

    entrada_batch = gerar_entrada_batch(
        dataset_teste=df_teste_temporal
    )

    # ========================================================
    # 4. CRIA PASTAS CASO NÃO EXISTAM
    # ========================================================

    PASTA_PROCESSED.mkdir(
        parents=True,
        exist_ok=True
    )

    PASTA_BATCH_INPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # 5. SALVA DATASET TEMPORAL COMPLETO
    # ========================================================

    caminho_dataset_teste = (
        PASTA_PROCESSED
        / "dataset_teste_temporal.csv"
    )

    df_teste_temporal.to_csv(
        caminho_dataset_teste,
        index=False,
        encoding="utf-8-sig"
    )

    logger.info(
        "Dataset temporal salvo em: %s",
        caminho_dataset_teste
    )

    # ========================================================
    # 6. SALVA ENTRADA BATCH
    # ========================================================

    caminho_entrada_batch = (
        PASTA_BATCH_INPUT
        / "entrada_batch_temporal.csv"
    )

    entrada_batch.to_csv(
        caminho_entrada_batch,
        index=False,
        encoding="utf-8-sig"
    )

    logger.info(
        "Entrada batch salva em: %s",
        caminho_entrada_batch
    )

    logger.info(
        "Preparação do dataset temporal finalizada"
    )
