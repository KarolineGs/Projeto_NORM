import logging
from pathlib import Path
import argparse
import joblib
import pandas as pd


def obter_argumentos():

    """Obtém os argumentos da linha de comando para a predição."""

    parser = argparse.ArgumentParser(
        description="Predição de risco de NORM"
    )

    parser.add_argument(
        "--salinidade",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--bario",  
        type=float,
        required=True,
    )

    parser.add_argument(
        "--estroncio",
        type=float,
        required=True,
    )

    return parser.parse_args() #retorna um objeto contendo os valores dos argumentos fornecidos na linha de comando

def validar_entradas(
        salinidade: float,
        bario: float,
        estroncio: float,
) -> None:
    """Valida os valores de entrada para a predição."""

    entradas ={
        'salinidade':salinidade,
        'bario': bario,
        'estroncio': estroncio
    }

    for nome, valor in entradas.items():
        if valor<=0:
            raise ValueError(
                f"{nome} deve ser um valor positivo e maior que zero."
                f"Valor recebido: {valor}"
            )




# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent # Diretório base do script

CAMINHO_MODELO = (
    BASE_DIR
    / "outputs"
    / "models"
    / "modelo_norm.joblib"
)


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
# PREDIÇÃO
# ==========================================================

def main() -> None:
    

    logger = logging.getLogger(__name__)

    logger.info("Carregando modelo")

    modelo = joblib.load(
        CAMINHO_MODELO
    )

    args = obter_argumentos()

    # 2. Valida os valores
    validar_entradas(
        salinidade=args.salinidade,
        bario=args.bario,
        estroncio=args.estroncio,
    )

    novo_dado = pd.DataFrame(
    {
        "SALINIDADE": [args.salinidade],
        "BARIO": [args.bario],
        "ESTRONCIO": [args.estroncio],
    }
    )   

    features = [
        "SALINIDADE",
        "BARIO",
        "ESTRONCIO",
    ]

    X = novo_dado[
        features
    ]

    predicao = modelo.predict(X)

    probabilidade = modelo.predict_proba(X)

    logger.info(
        "Predição concluída"
    )

    print(
        "Classe prevista:",
        predicao[0],
    )

    print(
        "Probabilidade sem NORM:",
        round(probabilidade[0][0], 4),
    )

    print(
        "Probabilidade com NORM:",
        round(probabilidade[0][1], 4),
    )


if __name__ == "__main__":

    configurar_logging()

    main()