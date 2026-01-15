# 📊 Análise de Crédito com Machine Learning

## Descrição do Projeto

Este projeto é uma aplicação interativa de análise de crédito desenvolvida com o objetivo de demonstrar técnicas de Machine Learning aplicadas à avaliação de risco de inadimplência. Construído utilizando Python e Streamlit, permite ao usuário simular dados de um cliente e obter uma estimativa probabilística de inadimplência, servindo como uma ferramenta educacional para entender conceitos de modelagem preditiva em finanças.

O projeto faz parte do meu portfólio profissional como Analista de Dados, destacando habilidades em processamento de dados, construção de modelos de ML e desenvolvimento de aplicações web interativas.

## Objetivo

- Demonstrar a aplicação prática de algoritmos de Machine Learning na análise de crédito.
- Fornecer uma interface intuitiva para simulação de cenários de risco de crédito.
- Educar sobre os princípios de avaliação de risco financeiro de forma acessível e visual.
- Apresentar um exemplo completo de pipeline de dados, desde o pré-processamento até a implantação de um modelo preditivo.

## Tecnologias Utilizadas

- **Python**: Linguagem principal para desenvolvimento e análise de dados.
- **Streamlit**: Framework para criação da aplicação web interativa.
- **Pandas & NumPy**: Bibliotecas para manipulação e análise de dados.
- **Scikit-learn**: Biblioteca para implementação de algoritmos de Machine Learning.
- **Joblib**: Para serialização e carregamento do modelo treinado.
- **Matplotlib/Seaborn**: Para visualizações de dados (se aplicável).

## Estrutura do Projeto

```
analise_de_credito/
├── app.py                    # Arquivo principal da aplicação Streamlit
├── requirements.txt          # Lista de dependências do projeto
├── dados/                    # Diretório contendo os dados utilizados
│   ├── raw/                  # Dados brutos originais
│   ├── interim/              # Dados intermediários após processamento inicial
│   └── processed/            # Dados finais processados para modelagem
├── modelo/                   # Modelo treinado salvo
├── src/                      # Código fonte auxiliar
│   ├── etl/                  # Scripts de extração, transformação e carregamento
│   ├── models/               # Funções de construção e treinamento do modelo
│   └── pipeline/             # Pipeline completo de Machine Learning
├── notebooks/                # Notebooks Jupyter para análise exploratória
└── README.md                 # Documentação do projeto
```

## Como Executar o Projeto Localmente

### Pré-requisitos

- Python 3.10 ou superior instalado
- Git para clonar o repositório

### Passos para Execução

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/JacksonvBarbosa/Analise-de-Credito
   ```

2. **Crie um ambiente virtual (recomendado):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute a aplicação:**
   ```bash
   streamlit run app.py
   ```

5. **Acesse a aplicação:**
   Abra seu navegador e vá para `http://localhost:8501`

## Como Acessar a Aplicação Online

A aplicação está disponível publicamente através do Streamlit Community Cloud. Você pode acessá-la diretamente em: [Link da Aplicação Online](https://zctegbmmwcn2gwrhccrytq.streamlit.app/) (substitua pelo link real quando disponível).

## Funcionamento da Aplicação

A aplicação oferece uma interface simples e intuitiva onde o usuário pode:

1. **Inserir dados do cliente:** Campos para informações como idade, renda, histórico de crédito, etc.
2. **Simular cenário:** Após inserir os dados, o usuário clica em "Analisar" para obter a previsão.
3. **Visualizar resultado:** A aplicação retorna a probabilidade estimada de inadimplência, acompanhada de uma interpretação visual (ex: gráfico de barras ou gauge).
4. **Explorar insights:** Seções adicionais podem incluir explicações sobre fatores que influenciaram a previsão.

O processo é totalmente interativo e não requer conhecimento técnico avançado.

## Modelo de Machine Learning

O modelo utilizado é um algoritmo de classificação baseado em ensemble (como Random Forest ou Gradient Boosting), treinado para prever a probabilidade de inadimplência com base em variáveis históricas de clientes.

- **Pré-processamento:** Inclui limpeza de dados, tratamento de valores ausentes, codificação de variáveis categóricas e normalização de features numéricas.
- **Treinamento:** O modelo é treinado com dados históricos balanceados, utilizando técnicas de validação cruzada para evitar overfitting.
- **Persistência:** O modelo treinado é salvo utilizando Joblib para rápida carga durante a execução da aplicação.

## Métricas Utilizadas

Durante o desenvolvimento e validação do modelo, foram consideradas métricas padrão para problemas de classificação binária:

- **Acurácia:** Proporção de previsões corretas.
- **Precisão:** Proporção de verdadeiros positivos entre as previsões positivas.
- **Recall (Sensibilidade):** Proporção de verdadeiros positivos identificados.
- **AUC-ROC:** Área sob a curva ROC, medida da capacidade discriminativa do modelo.
- **F1-Score:** Média harmônica entre precisão e recall.

Essas métricas são avaliadas em conjunto para garantir um equilíbrio entre detecção de inadimplentes e minimização de falsos positivos.

## Aviso Legal

⚠️ **IMPORTANTE:** Este projeto é exclusivamente educacional e demonstrativo. Os dados utilizados são fictícios ou anonimizados, e o modelo não foi validado para uso em produção. **Não utilize esta aplicação para tomar decisões reais de concessão de crédito ou avaliação de risco financeiro.** Qualquer uso comercial ou decisório deve ser baseado em modelos e dados validados por profissionais qualificados e instituições reguladas.

## Próximos Passos / Melhorias Futuras

- **Expansão de Features:** Adicionar mais variáveis preditoras e dados externos (ex: índices econômicos).
- **Otimização do Modelo:** Experimentar algoritmos mais avançados como redes neurais ou ensemble methods híbridos.
- **Interface Aprimorada:** Implementar dashboards mais ricos com visualizações interativas.
- **Deploy Avançado:** Migrar para plataformas de produção como Heroku ou AWS para maior escalabilidade.
- **Validação Cruzada:** Incorporar testes A/B e validação com dados reais (quando disponíveis).
- **Documentação Técnica:** Adicionar notebooks detalhados com análise exploratória e tuning de hiperparâmetros.

---

**Autor:** Jackson dos Santos Ventura  
**LinkedIn:** [Perfil do Linkedin](www.linkedin.com/in/jackson-dos-santos-ventura-716290b4)  
**Portfólio:** [Portfólio](https://github.com/JacksonvBarbosa/Analise-de-Credito)

⭐ Se este projeto foi útil, considere dar uma estrela no repositório!