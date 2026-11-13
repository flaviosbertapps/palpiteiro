@echo off
REM ============================================================================
REM Arquivo de Lote para Executar o Script de Previsão Diária de Ações
REM ============================================================================

REM Defina o caminho completo para o seu script Python aqui
SET "PYTHON_SCRIPT_PATH=C:\Users\fsbertaggia\OneDrive - Secretaria da Fazenda e Planejamento do Estado de São Paulo\_Pessoal\Quant\palpiteiro\scripts\e3_screening_regras.py"

REM Defina o nome do seu ambiente Anaconda
SET "ANACONDA_ENV_NAME=p311"

REM --- Não edite abaixo desta linha a menos que saiba o que está fazendo ---

REM Adiciona uma pausa de 10 segundos no final para ver o output se executado manualmente
REM Remova ou comente a linha 'timeout' se for agendar no Agendador de Tarefas
SET "PAUSE_AT_END=true"

REM Mudar para o diretório onde o script está localizado (importante para caminhos relativos no script)
pushd "%~dp0"

echo.
echo =================================================
echo  Iniciando execucao do script de previsao...
echo =================================================
echo.
echo  Ambiente Anaconda: %ANACONDA_ENV_NAME%
echo  Script a ser executado: %PYTHON_SCRIPT_PATH%
echo  Data passada como argumento: %1
echo.

REM Ativar o ambiente Anaconda e executar o script
REM O comando 'call' é importante para garantir que o script de lote continue após a ativação do conda
call conda activate %ANACONDA_ENV_NAME%

REM Verifica se o ambiente foi ativado com sucesso
if %errorlevel% neq 0 (
    echo.
    echo ERRO: Falha ao ativar o ambiente Anaconda '%ANACONDA_ENV_NAME%'.
    echo Verifique se o Anaconda esta no PATH do sistema ou execute este .bat a partir de um "Anaconda Prompt".
    goto End
)

echo.
echo Ambiente '%ANACONDA_ENV_NAME%' ativado. Executando o script Python...
echo.

REM Executar o script Python, passando o primeiro argumento do .bat (%1) para ele
REM %* passa todos os argumentos, mas %1 é suficiente aqui, pois só esperamos uma data.
python "%PYTHON_SCRIPT_PATH%" %1

REM Desativar o ambiente Conda
call conda deactivate
echo.
echo Ambiente desativado.

:End
echo.
echo =================================================
echo  Execucao do script de previsao concluida.
echo =================================================
echo.

REM Restaura o diretório original
popd

if defined PAUSE_AT_END (
    echo Pressione qualquer tecla para fechar esta janela...
    timeout /t 10 > nul