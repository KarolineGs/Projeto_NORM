import logging
from pathlib import Path

from src.pre_processamento import (
    processador_sigep,
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
    comparar_modelos,
    treinar_modelo_final,
    obter_coeficientes,
)

from src.graficos import (
    grafico_bsw_plataforma,
    grafico_scatter_matrix,
    grafico_quimica_3d,
    grafico_boxplots_quimicos,
    grafico_regioes_similaridade,
    grafico_matrizes_confusao,
    grafico_resultado_regressao,
    grafico_heatmap_metricas,
)


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent

PASTA_GRAFICOS = BASE_DIR / "outputs" / "figures"
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
# MAIN
# ==========================================================

def main() -> None:

    logger = logging.getLogger(__name__)

    logger.info("Iniciando pipeline")

    try:

        # ==================================================
        # 1. PREPARAR DIRETÓRIOS
        # ==================================================

        PASTA_GRAFICOS.mkdir(
            parents=True,
            exist_ok=True,
        )
        PASTA_MODELOS.mkdir(
            parents=True,
            exist_ok=True,
        )

        PASTA_LOGS.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger.info(
            "Diretório de gráficos: %s",
            PASTA_GRAFICOS,
        )


        # ==================================================
        # 2. PRÉ-PROCESSAMENTO
        # ==================================================

        logger.info(
            "Iniciando pré-processamento das bases"
        )

        df_sigep = processador_sigep()

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
        salvar_csv(
            df_analise,
            "base_analise.csv",
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


        logger.info(
            "Comparando modelos logísticos"
        )

        resultado_modelos, predicoes_modelos = comparar_modelos(
            dados_modelo=dados_modelo,
            n_splits=6,
            limite=0.45,
        )


        # ==================================================
        # 6. MODELO FINAL
        # ==================================================

        logger.info(
            "Selecionando modelo final | %s",
            NOME_MODELO_FINAL,
        )

        resultado_modelo_final = predicoes_modelos[
            NOME_MODELO_FINAL
        ]


        logger.info(
            "Treinando modelo final"
        )

        modelo_final = treinar_modelo_final(
            dados_modelo
        )

        coeficientes = obter_coeficientes(
            modelo_final
        )


        # ==================================================
        # 7. RESULTADOS
        # ==================================================

        print(
            "\nComparação dos modelos:\n"
        )

        print(
            resultado_modelos.round(3)
        )

        print(
            "\nCoeficientes do modelo final:\n"
        )

        print(
            coeficientes.round(3)
        )


        # ==================================================
        # 8. GRÁFICOS EXPLORATÓRIOS
        # ==================================================

        logger.info(
            "Gerando gráficos exploratórios"
        )


        grafico_bsw_plataforma(
            df_analise=df_analise,
            modo=MODO_ANALISE,
            salvar_em=(
                PASTA_GRAFICOS
                / "01_bsw_plataforma.png"
            ),
            exibir=False,
        )


        grafico_scatter_matrix(
            df_analise=df_analise,
            salvar_em=(
                PASTA_GRAFICOS
                / "02_scatter_matrix.html"
            ),
            exibir=False,
        )


        grafico_quimica_3d(
            df_analise=df_analise,
            salvar_em=(
                PASTA_GRAFICOS
                / "03_quimica_3d.html"
            ),
            exibir=False,
        )


        grafico_boxplots_quimicos(
            df_analise=df_analise,
            salvar_em=(
                PASTA_GRAFICOS
                / "04_boxplots_quimicos.png"
            ),
            exibir=False,
        )


        fig_similaridade, semelhantes = grafico_regioes_similaridade(
            df_analise=df_analise,
            salvar_em=(
                PASTA_GRAFICOS
                / "05_regioes_similaridade.png"
            ),
            exibir=False,
        )


        # ==================================================
        # 9. GRÁFICOS DE MODELAGEM
        # ==================================================

        logger.info(
            "Gerando gráficos dos modelos"
        )


        grafico_matrizes_confusao(
            dados_modelo=dados_modelo,
            predicoes_modelos=predicoes_modelos,
            salvar_em=(
                PASTA_GRAFICOS
                / "06_matrizes_confusao.png"
            ),
            exibir=False,
        )


        grafico_resultado_regressao(
            dados_modelo=dados_modelo,
            resultado_modelo=resultado_modelo_final,
            salvar_em=(
                PASTA_GRAFICOS
                / "07_resultado_regressao.png"
            ),
            exibir=False,
        )


        grafico_heatmap_metricas(
            resultado_modelos=resultado_modelos,
            salvar_em=(
                PASTA_GRAFICOS
                / "08_heatmap_metricas.png"
            ),
            exibir=False,
        )


        # ==================================================
        # 10. FINALIZAÇÃO
        # ==================================================

        logger.info(
            "Gráficos salvos em %s",
            PASTA_GRAFICOS,
        )

        logger.info(
            "Pipeline finalizado com sucesso"
        )


    except Exception:

        logger.exception(
            "Erro durante execução do pipeline"
        )

        raise


# ==========================================================
# EXECUÇÃO
# ==========================================================

if __name__ == "__main__":

    configurar_logging()

    main()