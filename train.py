import logging
from pathlib import Path
import joblib

from src.pre_processamento import (
    processador_sigre,
    processador_fenix,
    processar_scr,
    concat_arquivos,
    salvar_csv,
)
from src.processing import (
    processar_df_analise,
)

from src.modelling import (
    preparar_dados_modelo,   
    treinar_modelo_final
    
)

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent

PASTA_MODELOS = BASE_DIR / "outputs" / "models"
PASTA_LOGS = BASE_DIR / "logs"

NOME_MODELO_FINAL = (
    "Salinidade + Bário + Estrôncio"
)

MODO_ANALISE = "janela"


# ==========================================================
# LOGGING
# ==========================================================

def configurar_logging() -> None:

    logging.basicConfig(
        level=logging.INFO,
        
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )

# ==========================================================
# TREINAMENTO DO MODELO
# ==========================================================

def main() -> None:
    """
    Função principal para treinar o modelo de regressão.

    Realiza o pré-processamento dos dados, prepara os dados
    para modelagem e treina o modelo final.
    """

    configurar_logging()
    logger = logging.getLogger(__name__)

    logger.info("Iniciando pipeline do treinamento do modelo")

    try:
        PASTA_MODELOS.mkdir(
        parents=True,
        exist_ok=True,
        )
        logger.info('Iniciando o pré-processamento dos dados')

        df_sigre = processador_sigre()

        df_scr = processar_scr()

        df_fenix = processador_fenix()

        # ==================================================
        # 3. INTEGRAÇÃO SIGRE + SCR
        # ==================================================

        logger.info(
            "Integrando bases SIGRE e SCR"
        )

        scr_sigre = concat_arquivos(
            arquivo_1=df_sigre,
            arquivo_2=df_scr,
            coluna="Mes",
        )

        salvar_csv(
            scr_sigre,
            "base_integrada_residuos.csv",
        )

        # ==================================================
        # 4. DATASET DE ANÁLISE
        # ==================================================

        logger.info(
            "Gerando dataframe de análise | modo=%s",
            MODO_ANALISE,
        )

        df_analise = processar_df_analise(
            scr_sigre=scr_sigre,
            df_fenix=df_fenix,
            modo=MODO_ANALISE,
            
        )

        # ==================================================
        # 5. MODELAGEM
        # ==================================================

        logger.info(
            "Preparando dados para modelagem"
        )

        dados_modelo = preparar_dados_modelo(
            df_analise
        )

        # ==================================================
        # 6. MODELO FINAL
        # ==================================================
        logger.info(
            "Treinando modelo final"
        )

        modelo_final = treinar_modelo_final(
            dados_modelo
        )

        caminho_modelo = PASTA_MODELOS / "modelo_norm.joblib"

        joblib.dump(modelo_final, caminho_modelo)

        logger.info('Modelo final salvo em: %s', caminho_modelo)

        logger.info('Modelo treinado com sucesso')

    except Exception as e:
            logger.error(
                "Erro durante o treinamento do modelo: %s",
                str(e),
                exc_info=True
            )
            raise


if __name__ == "__main__":

    configurar_logging()

    main()

