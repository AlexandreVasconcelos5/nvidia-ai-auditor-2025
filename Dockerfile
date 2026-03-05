#Configuração do Ambiente Docker - NVIDIA AI Auditor 2025


#Definição da Imagem Base:
FROM python:3.10-slim

#Definição do Diretório de Trabalho no Contentor:
WORKDIR /app

#Instalação das Dependências De Sistema:
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

#Gestão das Dependências do Python:
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Ingestão do Código Fonte e Ativos do Projeto:
COPY . .

#Configuração de Rede:
EXPOSE 7860

#Configuração do Endereço e Porta para Permitir Acesso Exterior ao Contentor:
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]