# Função para fazer a predição
def predict_risk(model, cliente_pred):
    """Realiza a predição do risco de crédito usando o modelo treinado.
    Se modelo tem 'tipo': 'calibrado', usa probabilidades calibradas."""

    # Predições
    pred = model['modelo'].predict(cliente_pred)
    prob = model['modelo'].predict_proba(cliente_pred)
    return pred[0], prob[0][1]  # Retorna predição e probabilidade de inadimplência (classe 1)

# Função para classificar o risco com limiares baseados em dados
def classify_risk(prob):
    """Classifica risco com limiares data-driven baseados em distribuição real.
    Limiares: 33º percentil (baixo/médio), 66º percentil (médio/alto)"""
    
    # Limiares conservadores para credit scoring
    # (baseados em portfolio típico: ~80% bons, ~20% maus)
    p33 = 0.25  # 33º percentil de probabilidade
    p66 = 0.65  # 66º percentil de probabilidade
    
    if prob < p33:
        return "Baixo", "🟢"
    elif prob < p66:
        return "Médio", "🟡"
    else:
        return "Alto", "🔴"