import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

from src.pipeline.pipeline_ml import pipeline_teste
from src.models.train_roda_model import data_split
from src.models.predict_class_risk import predict_risk, classify_risk

# Configuração da página
st.set_page_config(
    page_title="Análise de Risco de Crédito",
    page_icon="💳",
    layout="wide"
)

# Função para carregar o modelo com cache
# @st.cache_resource
# def load_model():
#     """Carrega o artefato salvo com pipeline e modelo treinado."""
#     model_path = Path(__file__).parent / 'modelo' / 'xgb.joblib'
#     try:
#         return joblib.load(model_path)
#     except FileNotFoundError:
#         st.error("Erro: Modelo não encontrado. Verifique se o arquivo 'xgb.joblib' existe na pasta 'modelo'.")
#         return None

"""Carrega valores únicos para os selectboxes a partir dos dados processados."""
dados = pd.read_csv(r'dados\processed\df_analises_models.csv')


# Título e descrição principal
st.title("💳 Análise de Risco de Crédito")
st.markdown("""
Esta aplicação demonstra como um modelo de machine learning avalia o risco de inadimplência
para concessão de crédito. Preencha os dados na barra lateral e clique em "Analisar Risco".
""")

# Sidebar para inputs
st.sidebar.header("📝 Informações do Cliente")

# Inputs do usuário
idade = float(st.sidebar.slider('Idade', 18, 100, 30))
grau_escolaridade = st.sidebar.selectbox('Qual o Grau de Escolaridade ?', dados['Grau_escolaridade'].unique())
estado_civil = st.sidebar.selectbox('Qual é o seu estado civil ?', dados['Estado_civil'].unique())
membros_familia = float(st.sidebar.slider('Selecione quantos membros tem na sua família', 1, 20))

carro_proprio = st.sidebar.radio('Possui Carro Próprio?', ['Sim', 'Não'], index=0)
carro_proprio_dict = {'Sim': 1, 'Não':0}
carro_proprio = carro_proprio_dict.get(carro_proprio)

casa_propria = st.sidebar.radio('Possui Casa Própria?', ['Sim', 'Não'], index=0)
casa_propria_dict = {'Sim' : 1, 'Não' : 0}
casa_propria = casa_propria_dict.get(casa_propria)

tipo_moradia = st.sidebar.selectbox('Tipo de Moradia', dados['Moradia'].unique())

categoria_renda = st.sidebar.selectbox('Categoria de Renda', dados['Categoria_de_renda'].unique())

ocupacao = st.sidebar.selectbox('Ocupação', dados['Ocupacao'].unique())

tempo_experiencia = float(st.sidebar.slider('Anos de Experiência', 0, 30, 5))

rendimentos = float(st.sidebar.number_input('Rendimento Anual (R$)', min_value=0.0, value=50000.0, step=500.0))

telefone_trabalho = st.sidebar.radio('Telefone Corporativo?', ['Sim', 'Não'], index=0)
telefone_trabalho_dict = {'Sim' : 1, 'Não' : 0}
telefone_trabalho = telefone_trabalho_dict.get(telefone_trabalho)

telefone_fixo = st.sidebar.radio('Telefone Fixo?', ['Sim', 'Não'], index=0)
telefone_fixo_dict = {'Sim' : 1, 'Não' : 0}
telefone_fixo = telefone_fixo_dict.get(telefone_fixo)

email = st.sidebar.radio('Possui Email?', ['Sim', 'Não'], index=1)
email_dict = {'Sim' : 1, 'Não' : 0}
email = email_dict.get(email)

# Botão para executar análise
if st.sidebar.button("🔍 Analisar Risco"):
    # Criar lista com dados do novo cliente (sem coluna target)
    novo_cliente = [
        0,  # ID_Cliente
        carro_proprio,  # Tem_carro
        casa_propria,  # Tem_casa_propria
        telefone_trabalho,  # Tem_telefone_trabalho
        telefone_fixo,  # Tem_telefone_fixo
        email,  # Tem_email
        membros_familia,  # Tamanho_familia
        rendimentos,  # Rendimento_anual
        idade,  # Idade
        tempo_experiencia,  # Anos_empregado
        categoria_renda,  # Categoria_de_renda
        grau_escolaridade,  # Grau_escolaridade
        estado_civil,  # Estado_civil
        tipo_moradia,  # Moradia
        ocupacao,  # Ocupacao
        0 # target (Mau)
    ]

    # Separando os dados em treino e teste
    treino_df, teste_df = data_split(dados, 0.2)

    #Criando novo cliente
    cliente_predict_df = pd.DataFrame([novo_cliente],columns=teste_df.columns)

    #Concatenando novo cliente ao dataframe dos dados de teste
    teste_novo_cliente  = pd.concat([teste_df,cliente_predict_df],ignore_index=True)

    #Aplicando a pipeline
    teste_novo_cliente = pipeline_teste(teste_novo_cliente)

    #retirando a coluna target
    cliente_pred = teste_novo_cliente.drop(['Mau'], axis=1)

    # Processar dados
    with st.spinner('Processando análise...'):
        # Carregar modelo
        model = joblib.load('modelo/modelo.joblib')
        
        if model is not None:
            # Fazer predição
            pred, prob = predict_risk(model, cliente_pred)
        else:
            pred, prob = None, None

    if pred is not None:
        # Área principal com resultados
        st.header("📊 Resultado da Análise")

        # Métricas em colunas
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Probabilidade de Inadimplência",
                value=f"{prob:.1%}"
            )

        with col2:
            risco, emoji = classify_risk(prob)
            st.metric(
                label="Classificação do Risco",
                value=f"{emoji} {risco}"
            )

        # Mensagem explicativa
        st.markdown("---")
        if pred == 0:
            st.success("✅ **Crédito Aprovado!** Parabéns! Seu perfil apresenta baixo risco de inadimplência.")
        else:
            st.error("❌ **Crédito Rejeitado.** Seu perfil apresenta risco elevado de inadimplência.")

        st.info("""
        💡 **Sobre esta análise:**
        - Esta é uma demonstração educacional de um modelo de machine learning.
        - O resultado não substitui avaliação profissional de crédito.
        - Use apenas para fins de aprendizado e portfólio.
        """)

# Disclaimer na sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("*⚠️ Modelo educacional - Não use para decisões reais de crédito.*")
 
