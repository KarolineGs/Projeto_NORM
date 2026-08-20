import logging
import os
from pathlib import Path

import joblib
import pandas as pd
from google.cloud import storage


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

CAMINHO_MODELO = (
    BASE_DIR
    / "outputs"
    / "models"
    / "modelo_norm.joblib"
)

CAMINHO_ENTRADA = (
    BASE_DIR
    / "data"
    / "batch"
    / "input"
    / "entrada_batch_temporal.csv"
)

CAMINHO_SAIDA = (
    BASE_DIR
    / "data"
    / "batch"
    / "output"
    / "saida_batch.csv"
)


# ==========================================================
# CONFIGURAÇÕES GCP
# ==========================================================

BUCKET_NAME = os.getenv(
    "BUCKET_NAME",
    "projeto-norm-ml-batch"
)

MODO_GCP = os.getenv(
    "MODO_GCP",
    "false"
).lower() == "true"

BLOB_ENTRADA = "batch/input/entrada_batch_temporal.csv"
BLOB_SAIDA = "batch/output/saida_batch.csv"


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
# CLOUD STORAGE
# ==========================================================

def baixar_entrada_gcs(
    bucket_name: str,
    blob_name: str,
    destino: Path,
) -> None:

    logger = logging.getLogger(__name__)

    logger.info(
        "Baixando gs://%s/%s",
        bucket_name,
        blob_name,
    )

    destino.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = storage.Client()

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.download_to_filename(
        str(destino)
    )

    logger.info(
        "Arquivo de entrada baixado com sucesso"
    )


def enviar_saida_gcs(
    bucket_name: str,
    blob_name: str,
    origem: Path,
) -> None:

    logger = logging.getLogger(__name__)

    logger.info(
        "Enviando resultado para gs://%s/%s",
        bucket_name,
        blob_name,
    )

    client = storage.Client()

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.upload_from_filename(
        str(origem)
    )

    logger.info(
        "Resultado enviado ao Cloud Storage"
    )


# ==========================================================
# INFERÊNCIA
# ==========================================================

def main() -> None:

    logger = logging.getLogger(__name__)

    logger.info("Iniciando inferência batch")

    try:

        # ==================================================
        # 1. CARREGAR MODELO
        # ==================================================

        logger.info(
            "Carregando modelo: %s",
            CAMINHO_MODELO
        )

        modelo = joblib.load(
            CAMINHO_MODELO
        )

        # ==================================================
        # 2. OBTER DADOS DE ENTRADA
        # ==================================================

        if MODO_GCP:

            logger.info(
                "Modo GCP ativado"
            )

            baixar_entrada_gcs(
                bucket_name=BUCKET_NAME,
                blob_name=BLOB_ENTRADA,
                destino=CAMINHO_ENTRADA,
            )

        else:

            logger.info(
                "Modo local ativado"
            )

        logger.info(
            "Carregando dados de entrada: %s",
            CAMINHO_ENTRADA
        )

        df = pd.read_csv(
            CAMINHO_ENTRADA
        )

        logger.info(
            "Dados carregados | %d linhas x %d colunas",
            df.shape[0],
            df.shape[1]
        )

        # ==================================================
        # 3. FEATURES
        # ==================================================

        df["RELACAO_BA_SR"] = (
            df["BARIO"]
            .div(df["ESTRONCIO"])
            .where(df["ESTRONCIO"] > 0)
        )

        features = [
            "SALINIDADE",
            "BARIO",
            "ESTRONCIO"        
        ]

        X = df[features]

        # ==================================================
        # 4. INFERÊNCIA
        # ==================================================

        logger.info(
            "Executando predição para %d registros",
            len(X)
        )

        df["PREDICAO_NORM"] = modelo.predict(
            X
        )

        df["PROB_NORM"] = modelo.predict_proba(
            X
        )[:, 1]

        # ==================================================
        # 5. SALVAR LOCALMENTE
        # ==================================================

        CAMINHO_SAIDA.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_csv(
            CAMINHO_SAIDA,
            index=False,
            encoding="utf-8-sig",
        )

        logger.info(
            "Inferência batch finalizada | %d registros processados",
            len(df)
        )

        logger.info(
            "Resultado salvo em: %s",
            CAMINHO_SAIDA
        )

        # ==================================================
        # 6. ENVIAR RESULTADO PARA GCP
        # ==================================================

        if MODO_GCP:

            enviar_saida_gcs(
                bucket_name=BUCKET_NAME,
                blob_name=BLOB_SAIDA,
                origem=CAMINHO_SAIDA,
            )

    except Exception as e:

        logger.error(
            "Erro durante a inferência batch: %s",
            str(e),
            exc_info=True
        )

        raise


if __name__ == "__main__":

    configurar_logging()

    main()
