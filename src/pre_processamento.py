import re
import pandas as pd
import logging
import numpy as np
from pathlib import Path
from unidecode import unidecode



# Configura logging
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent   # Diretório base do projeto

PASTA_RAW = BASE_DIR / "data" / "raw"
PASTA_PROCESSED = BASE_DIR / "data" / "processed"


###TRATAMENTO COMUNS AOS BANCOS DE DADOS###########
def carregar_dados(
    nome_arquivo: str,
    pasta: Path = PASTA_RAW
) -> pd.DataFrame:

    
    caminho = pasta / nome_arquivo

    logger.info("Carregando dados de %s", caminho)

    if not caminho.exists():
        logger.error("Arquivo não encontrado: %s", caminho)
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}"
        )

    if caminho.suffix == ".csv":
        try:
            df = pd.read_csv(
                caminho,
                encoding="utf-8"
            )

        except UnicodeDecodeError:
            logger.warning(
                "Falha na leitura UTF-8. Tentando latin1."
            )

            df = pd.read_csv(
                caminho,
                encoding="latin1"
            )

    elif caminho.suffix == ".parquet":
        df = pd.read_parquet(caminho)

    elif caminho.suffix in [".xls", ".xlsx"]:
        df = pd.read_excel(caminho)

    else:
        logger.error(
            "Formato não suportado: %s",
            caminho.suffix
        )

        raise ValueError(
            f"Formato não suportado: {caminho.suffix}"
        )

    logger.info(
        "Dados carregados com sucesso | %d linhas x %d colunas",
        df.shape[0],
        df.shape[1]
    )
    return df

def salvar_csv(
    df: pd.DataFrame,
    nome_arquivo: str,
    pasta: Path = PASTA_PROCESSED
) -> pd.DataFrame:
    """
    Salva um DataFrame como CSV na pasta especificada.
    Cria automaticamente a pasta se ela não existir.
    """

    caminho_pasta = pasta

    caminho_pasta.mkdir(
        parents=True,
        exist_ok=True
    )

    caminho_arquivo = caminho_pasta / nome_arquivo

    logger.info(
        "Salvando arquivo em %s",
        caminho_arquivo
    )

    df.to_csv(
        caminho_arquivo,
        index=False,
        encoding="utf-8-sig"
    )

    logger.info(
        "Arquivo salvo com sucesso | %s | %d linhas x %d colunas",
        nome_arquivo,
        df.shape[0],
        df.shape[1]
    )

    return df

def converter_colunas_data(
    df: pd.DataFrame,
    coluna: str =  'Período',
    formato: str =  '%Y/%m',
    errors:str= "coerce"
) -> pd.DataFrame:
    """
    Converte colunas de datas em string para datetime no DataFrame.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com as colunas a serem convertidas.
    colunas_data : list
        Lista com os nomes das colunas de data a converter.
    formato : str, opcional
        Formato esperado da data, ex: "%Y-%m-%d".
        Se None, pandas tentará inferir o formato automaticamente.
    erros : {"raise", "coerce", "ignore"}, padrão="coerce"
        - "raise": gera erro se houver valor inválido
        - "coerce": converte valores inválidos em NaT
        - "ignore": deixa os valores inválidos como string

    Retorna
    -------
    pd.DataFrame
        DataFrame com as colunas convertidas para datetime.
    """
    df = df.copy()
    logger.info("Convertendo coluna %s para datetime", coluna)
    df[coluna] = pd.to_datetime(df[coluna], errors=errors, format=formato)
    df['dias_mes'] = df[coluna].dt.days_in_month
    df['Ano'] = df[coluna].dt.year
    df['Mes'] = df[coluna].dt.to_period('M').dt.to_timestamp()
    return df

def remover_colunas_nulas(df:pd.DataFrame)-> pd.DataFrame:
    """Remove colunas nulas"""
    df= df.copy()
    logger.info('Removendo colunas nulas do dataset')
    df = df.dropna(axis=1, how='all')
    return df

def tratar_valores_nulos(df: pd.DataFrame, subset:list)-> pd.DataFrame:
    """Remove valores nulos baseado na coluna Instalação Destino no banco de dados SIGEP"""
    logger.info("Tratando valores nulos do banco de dados do SIGEP")
    df= df.copy()
    df.dropna(subset = subset)       
    return df

def tratar_dados_duplicados(df: pd.DataFrame, subset:list)-> pd.DataFrame:
    """Remove valores duplicados do dataset"""
    logger.info('Removendo dados duplicados do banco de dados SIGEP')
    df=df.copy()
    df= df.drop_duplicates(subset=subset)
    return df

def renomear_colunas(df: pd.DataFrame, columns:dict) -> pd.DataFrame:
    """Renomeia colunas do banco de dados do sigep"""
    logger.info('Renomeando nomes de plataformas no banco de dados SIGEP ')
    df = df.copy()
    df = df.rename(columns = columns)
    return df

def ordenar_coluna(df:pd.DataFrame, columns:list) -> pd.DataFrame:
    logger.info('Ordenando coluna %s ', columns)
    df = df.sort_values(columns).reset_index(drop=True)
    return df

def converter_colunas_numericas(
    df: pd.DataFrame,
    colunas: list[str]
) -> pd.DataFrame:
    """Converte as colunas informadas para tipo numérico."""

    logger.info(
        "Convertendo colunas para formato numérico: %s",
        colunas
    )

    df = df.copy()

    for coluna in colunas:
        if coluna in df.columns:
            df[coluna] = pd.to_numeric(
                df[coluna],
                errors="coerce"
            )

    return df

def concat_arquivos(
    arquivo_1: pd.DataFrame,
    arquivo_2: pd.DataFrame,
    coluna: str
) -> pd.DataFrame:
    """Concatena ao primeiro DataFrame apenas os registros mais recentes do segundo."""

    logger.info(
        "Iniciando concatenação dos arquivos pela coluna %s",
        coluna
    )

    arquivo_1 = arquivo_1.copy()
    arquivo_2 = arquivo_2.copy()

    ultima_data = arquivo_1[coluna].max()

    logger.info(
        "Última data encontrada na base histórica: %s",
        ultima_data
    )

    novos_dados = arquivo_2[
        arquivo_2[coluna] > ultima_data
    ]

    logger.info(
        "Novos registros encontrados: %d",
        len(novos_dados)
    )

    df = (
        pd.concat(
            [arquivo_1, novos_dados],
            ignore_index=True
        )
        .sort_values(coluna)
        .reset_index(drop=True)
    )

    logger.info(
        "Concatenação finalizada | %d linhas",
        len(df)
    )

    return df

##### TRATAMENTOS BANCO DE DADOS SIGEP#####

def renomear_plataformas_sigep(df: pd.DataFrame)-> pd.DataFrame:
    """Padronizar nomes das plataformas do banco de dados do SIGEP que serão analisadas"""
    logger.info("Padronizando nomes das plataformas no banco de dados SIGEP")
    df = df.copy()
    df['LOCAL DA GERAÇÃO'] = df['LOCAL DA GERAÇÃO'].replace({
                                                            'FPSO CIDADE DE SANTOS': 'FPCST',
                                                            'FPSO CIDADE DE ANGRA DOS REIS': 'FPCAR',
                                                            'FPSO CIDADE DE ITAGUAI': 'FPCIG',
                                                            'FPSO CIDADE DE SAO PAULO': 'FPCSP',
                                                            'FPSO CIDADE DE SAQUAREMA': 'FPCSQ',
                                                            'FPSO CIDADE DE MARICA': 'FPCMC',
                                                            'FPSO CIDADE DE MANGARATIBA': 'FPCMB',
                                                            'FPSO BRASIL': 'FPSO-BR',
                                                            'FPSO-BRASIL': 'FPSO-BR',
                                                            'FPSO FLUMINENSE': 'FPRJ',
                                                            'FPSO MARLIM SUL': 'FPSO-MLS',
                                                            'FPSO CAPIXABA': 'CAPX',
                                                            'FPSO CIDADE DO RIO DE JANEIRO': 'FPSO-RJ',
                                                            'FPSO RIO DAS OSTRAS': 'FPSO-RO',
                                                            'ESPADARTE FPSO': 'FPSO-ESP'})
    df['LOCAL DA GERAÇÃO'] = (df['LOCAL DA GERAÇÃO'].str.replace(r"(?i)petrobras\s+\d+\s*\(?(P-\d+)?\)?",
                 lambda m: m.group(1) if m.group(1) else "P-" + "".join(filter(str.isdigit, m.group(0))),
                 regex=True))
    df['LOCAL DA GERAÇÃO'] = (
    df['LOCAL DA GERAÇÃO']
    .astype("string")
    .map(lambda x: unidecode(x) if pd.notna(x) else x)
    .str.upper()
                )
    df['LOCAL DA GERAÇÃO'] = df['LOCAL DA GERAÇÃO'].str.upper()
    
    return df

def produção_mensal_sigep(df):
    """Calcula a produção mensal das plataformas do banco de dados do SIGEP considerando os dias do mês"""
    df = df.copy()
    logger.info("Calculando produção mensal do SIGEP")
    df['Óleo (bbl/mes)'] = df['Óleo (bbl/dia)']*df['dias_mes']
    df['Condensado (bbl/mes)'] = df['Condensado (bbl/dia)']*df['dias_mes']
    df['Água (bbl/mes)'] = df['Água (bbl/dia)']*df['dias_mes']
    return df

def processador_sigep() -> pd.DataFrame:
    logger.info('Iniciando pre_processamento do SIGEP')
    df = carregar_dados('2003_2024_SIGEP.csv')
    df = remover_colunas_nulas(df)
    df = tratar_valores_nulos(df, subset=['Instalação Destino'])
    df = tratar_dados_duplicados(df,subset=['Óleo (bbl/dia)'] )
    df = renomear_colunas(df,columns={'Instalação Destino': 'LOCAL DA GERAÇÃO'})
    df = renomear_plataformas_sigep(df)
    df = converter_colunas_data(df)
    df = produção_mensal_sigep(df)
    df=salvar_csv(df, 'sigep_processado.csv')
    logger.info("Processamento do SIGEP finalizado")    
    return df


##### TRATAMENTOS BANCO DE DADOS SIGRE#####

def renomear_plataformas_sigre(df: pd.DataFrame)-> pd.DataFrame:
        """Padronizar nomes das plataformas do banco de dados do SIGRE que serão analisadas"""
        df = df.copy()
        logger.info("Padronizando nomes das plataformas no banco de dados SIGRE")        
        
        df['LOCAL DA GERAÇÃO'] = (
        df['L. de Atuação da Geradora']
        .astype(str)
        .str.split('/')
        .str[-1]
        .str.extract(r'(FPSO[^/]+|P-\d+|P\d+|SS-\d+|CDAN|CAPX|FP\w+)', expand=False)
        .fillna('OUTROS')
        .str.upper()
        .str.strip()
    )
        col = (
            df['LOCAL DA GERAÇÃO']
            .str.replace(r'\(.*?\)', '', regex=True)   # remove (MV29)
            .str.replace(r'^FPSO([A-Z])', r'FPSO \1', regex=True)  # FPSOBR → FPSO BR
            .str.replace(r'\s+', ' ', regex=True)
        )

        col = col.replace({
            r'.*BRASIL.*': 'FPBR',
            r'.*MARICA.*': 'FPMQT', 
            r'.*SAQUAREMA.*': 'FPCSQ',
            r'.*SANTOS.*':'FPCST',
            r'.*ILHABELA.*': 'FPCIB',
            r'.*SAO PAULO.*': 'FPCSP',
            r'.*NITER.*': 'FPCNI',
            r'.*ITAJA.*': 'FPCIT',
            r'.*MANGARATIBA.*': 'FPCMG',
            r'.*PARATY.*': 'FPCPY', 

        }, regex=True)

        col = col.str.replace(r'^P(\d+)$', r'P-\1', regex=True)

        logger.info(
        "Padronizando tipos de resíduos do SIGRE")

        df['LOCAL DA GERAÇÃO'] = col
        df['TIPO DE RESÍDUO'] = df['TIPO DE RESÍDUO'].replace({
            'BORRA OLEOSA COM NORM (CATEGORIA I - ETIQUETA BRANCA)': 'BORRA OLEOSA COM NORM',
            'BORRA OLEOSA COM NORM (CATEGORIA II - ETIQUETA AMARELA)': 'BORRA OLEOSA COM NORM'
        })
        logger.info('Filtrando as colunas necessárias para análise do banco de dados do SIGRE')

        df_filtrado = df[
            ['FCDR','Mes','LOCAL DA GERAÇÃO','TIPO DE RESÍDUO','MASSA DO VOLUME (kg)']
        ]
        return df_filtrado

def processador_sigre()-> pd.DataFrame:

    logger.info('Iniciando pre processamento do SIGRE')
    df = carregar_dados('levantamento_borras+norm_sigre_2025_11_11.xlsx')
    df=remover_colunas_nulas(df)
    df = renomear_colunas(df, columns={'Dt. Geração': 'Período',
                                        'Resíduo': 'TIPO DE RESÍDUO',
                                        'Qtd. Recebida (kg)': 'MASSA DO VOLUME (kg)'})
    df = converter_colunas_data(df)
    df= renomear_plataformas_sigre(df)
    df=salvar_csv(df, 'sigre_processado.csv')
    logger.info('Processamento do SIGRE finalizado')
    return df

##### TRATAMENTOS BANCO DE DADOS SCR#####

def processar_scr()-> pd.DataFrame:
    logger.info('Iniciando pre processamento do SCR')
    df = carregar_dados('Dados_SCR.xlsx')
    df = remover_colunas_nulas(df)
    df = converter_colunas_data(df, coluna= 'Data - Geração')
    df = renomear_colunas(df, columns ={'Data - Geração':'Período',
                                        'Resíduo':'TIPO DE RESÍDUO',
                                        'Unidade Operacional':'LOCAL DA GERAÇÃO',
                                        'Geração - Kg':'MASSA DO VOLUME (kg)'})
    logger.info('Filtrando as colunas necessárias para análise do banco de dados do SCR')
    df = df[['Mes','LOCAL DA GERAÇÃO','TIPO DE RESÍDUO','MASSA DO VOLUME (kg)']]
    df=salvar_csv(df, 'scr_processado.csv')
    logger.info('Processamento do SCR finalizado')
    return df

##### TRATAMENTOS BANCO DE DADOS FENIX#####

ARQUIVOS_AGUA = [
    "ABL-1.xlsx", "ABL-JUB-1.xlsx", "ABL-JUB-2.xlsx",
    "BC_BR.xlsx", "JUB-1.xlsx", "JUB-JUB.xlsx",
    "MLS-1.xlsx", "MLS-2.xlsx", "MRL-1.xlsx", "MRL-2.xlsx",
    "RO-1.xlsx", "RO-2.xlsx", "RO-3.xlsx",
    "SPH.xlsx", "TUP.xlsx", "URG.xlsx", "VD-AB-1.xlsx"
]

COLUNAS_FIXAS = {
    "Nome do Campo": "CAMPO",
    "Nome do Poço": "POCO",
    "Nome da Plataforma": "PLATAFORMA",
    "Sigla da Plataforma": "SIGLA",
    "Data": "DATA",
    "Qw (m³/d)": "QW_M3D",
    "Qo (m³/d)": "QO_M3D",
    "BSW (%)": "BSW",
}

COLUNAS_QUIMICAS = [
    "BÁRIO",
    "ESTRÔNCIO",
    "SALINIDADE"
]
# Nomes sem acento usados a partir da agregação mensal em diante
NOME_ASCII_QUIMICAS = {
    'BÁRIO':      'BARIO',
    'ESTRÔNCIO':  'ESTRONCIO',
    'SALINIDADE': 'SALINIDADE'
}


ALIAS_QUIMICAS = {
    "SALINIDADE DA ÁGUA (MG DE NACL/L)": "SALINIDADE",}

colunas_numericas = [
    "QW_M3D",
    "QO_M3D",
    "BSW"
] + COLUNAS_QUIMICAS


def extrair_parametro_fenix(col: str) -> str:
    match = re.search(r'\[([^\]]+)\]', str(col))
    bruto = match.group(1).strip() if match else str(col).strip()
    return ALIAS_QUIMICAS.get(bruto, bruto)

def carregar_dados_fenix(
    pasta: Path = PASTA_RAW
) -> pd.DataFrame:

    logger.info(
        "Iniciando carregamento de %d arquivos FENIX",
        len(ARQUIVOS_AGUA)
    )

    
    pasta_dados = pasta

    dfs = []

    for arquivo in ARQUIVOS_AGUA:

        caminho = pasta_dados / arquivo

        logger.info(
            "Carregando arquivo FENIX: %s",
            arquivo
        )

        if not caminho.exists():

            logger.error(
                "Arquivo FENIX não encontrado: %s",
                caminho
            )

            raise FileNotFoundError(
                f"Arquivo não encontrado: {caminho}"
            )

        df = pd.read_excel(
            caminho,
            header=3
        )

        logger.info(
            "%s carregado | %d linhas x %d colunas",
            arquivo,
            df.shape[0],
            df.shape[1]
        )

        renomear = {}

        for col in df.columns:

            nome_limpo = str(col).strip()

            if nome_limpo in COLUNAS_FIXAS:
                renomear[col] = COLUNAS_FIXAS[nome_limpo]

            elif "ensaio" not in nome_limpo.lower():

                parametro = extrair_parametro_fenix(col)

                if parametro in COLUNAS_QUIMICAS:
                    renomear[col] = parametro

        df = renomear_colunas(
            df,
            columns=renomear
        )

        colunas_alvo = (
            list(COLUNAS_FIXAS.values())
            + COLUNAS_QUIMICAS
        )

        colunas_presentes = [
            col
            for col in colunas_alvo
            if col in df.columns
        ]

        df = df[colunas_presentes].copy()

        df["_arquivo"] = arquivo

        dfs.append(df)

    df_total = pd.concat(
        dfs,
        ignore_index=True
    )

    logger.info(
        "Carregamento FENIX finalizado | %d arquivos | %d linhas x %d colunas",
        len(dfs),
        df_total.shape[0],
        df_total.shape[1]
    )

    return df_total

def remover_datas_invalidas(
    df: pd.DataFrame,
    coluna: str = "DATA",
    data_min: str = "2000-01-01"
) -> pd.DataFrame:

    logger.info(
        "Removendo datas inválidas ou anteriores a %s na coluna %s",
        data_min,
        coluna
    )

    df = df.copy()

    n_antes = len(df)

    df[coluna] = pd.to_datetime(
        df[coluna],
        errors="coerce"
    )

    data_min = pd.to_datetime(data_min)

    df = df[
        df[coluna].notna()
        & (df[coluna] >= data_min)
    ].reset_index(drop=True)

    logger.info(
        "Linhas removidas por data inválida: %d",
        n_antes - len(df)
    )

    return df

def agregar_mensal_fenix(
    df: pd.DataFrame
) -> pd.DataFrame:

    logger.info(
        "Iniciando agregação mensal do FENIX"
    )

    df = df.copy()

    agregacoes_poco = {
        "CAMPO": ("CAMPO", "first"),
        "QW_M3D": ("QW_M3D", "median"),
        "QO_M3D": ("QO_M3D", "median"),
        "BSW": ("BSW", "median"),
    }

    for coluna in COLUNAS_QUIMICAS:
        if coluna in df.columns:
            agregacoes_poco[coluna] = (coluna, "median")

    # Quimica e vazao aparecem em linhas distintas no FENIX.
    # Consolida primeiro cada poco no mes para parear os dados.
    mensal_poco = (
        df
        .groupby(
            ["SIGLA", "Mes", "POCO"],
            as_index=False,
        )
        .agg(**agregacoes_poco)
    )

    def media_ponderada(grupo, coluna):
        dados = (
            grupo[[coluna, "QW_M3D"]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        dados = dados[
            (dados[coluna] > 0)
            & (dados["QW_M3D"] > 0)
        ]

        if dados.empty:
            return np.nan

        return np.average(
            dados[coluna],
            weights=dados["QW_M3D"],
        )

    registros = []

    for (sigla, mes), grupo in mensal_poco.groupby(
        ["SIGLA", "Mes"]
    ):

        registro = {
            "SIGLA": sigla,
            "Mes": mes,
            "CAMPO": grupo["CAMPO"].dropna().iloc[0]
            if grupo["CAMPO"].notna().any()
            else np.nan,
            "QW_M3D": grupo["QW_M3D"].sum(min_count=1),
            "QO_M3D": grupo["QO_M3D"].sum(min_count=1),
            "BSW": grupo["BSW"].median(),
            "N_POCOS": grupo["POCO"].nunique(),
        }

        for col in COLUNAS_QUIMICAS:

            if col in grupo.columns:

                nome_saida = NOME_ASCII_QUIMICAS[col]

                registro[nome_saida] = media_ponderada(
                    grupo,
                    col
                )

        registros.append(registro)

    agg = pd.DataFrame(registros)

    colunas_quimicas_ascii = [
        NOME_ASCII_QUIMICAS[col]
        for col in COLUNAS_QUIMICAS
        if NOME_ASCII_QUIMICAS[col] in agg.columns
    ]

    agg["dias_mes"] = (
        agg["Mes"]
        .dt.days_in_month
    )

    agg["QW_mensal_m3"] = (
        agg["QW_M3D"]
        * agg["dias_mes"]
    )

    agg["QO_mensal_m3"] = (
        agg["QO_M3D"]
        * agg["dias_mes"]
    )

    mes_min = agg["Mes"].min()
    mes_max = agg["Mes"].max()

    calendario = pd.date_range(
        mes_min,
        mes_max,
        freq="MS"
    )

    idx_completo = pd.MultiIndex.from_product(
        [
            agg["SIGLA"].unique(),
            calendario
        ],
        names=[
            "SIGLA",
            "Mes"
        ]
    )

    agg = (
        agg
        .set_index(["SIGLA", "Mes"])
        .reindex(idx_completo)
        .reset_index()
        .sort_values(["SIGLA", "Mes"])
        .reset_index(drop=True)
    )

    agg["dias_mes"] = (
        agg["Mes"]
        .dt.days_in_month
    )

    for col in colunas_quimicas_ascii:

        agg[col] = (
            agg.groupby("SIGLA")[col]
            .transform(
                lambda x: x.ffill(limit=2)
            )
        )

    for col in colunas_quimicas_ascii:

        agg[f"{col}_kg_mes"] = (
            agg[col]
            * agg["QW_mensal_m3"]
            / 1000
        )

    logger.info(
        "Agregação mensal FENIX finalizada | %d linhas x %d colunas",
        agg.shape[0],
        agg.shape[1]
    )

    logger.info(
        "Período FENIX: %s a %s | plataformas=%d",
        agg["Mes"].min().strftime("%Y-%m"),
        agg["Mes"].max().strftime("%Y-%m"),
        agg["SIGLA"].nunique()
    )

    return agg

def processador_fenix() -> pd.DataFrame:
    logger.info('Iniciando pre processamento do fenix')
    df = carregar_dados_fenix()
    df = remover_colunas_nulas(df)
    df = converter_colunas_data(df, coluna= 'DATA', formato=None)
    df = remover_datas_invalidas(df)    
    df = ordenar_coluna(df, columns=['SIGLA', 'Mes'])
    df = converter_colunas_numericas(df,colunas=colunas_numericas)
    df = agregar_mensal_fenix(df)
    df = renomear_colunas(df, columns={'SIGLA': 'LOCAL DA GERAÇÃO'})
    df=salvar_csv(df, 'fenix_processado.csv')
    logger.info('Processamento do fenix finalizado')

    return df




