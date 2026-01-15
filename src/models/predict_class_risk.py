# Função para fazer a predição
def predict_risk(model, cliente_pred):
    """Realiza a predição do risco de crédito usando o modelo treinado."""

    # Predições
    pred = model['modelo'].predict(cliente_pred)
    prob = model['modelo'].predict_proba(cliente_pred)
    return pred[0], prob[0][1]  # Retorna predição e probabilidade de inadimplência (classe 1)

# Função para classificar o risco baseado na probabilidade
def classify_risk(prob):
    """Classifica o risco baseado na probabilidade de inadimplência."""
    if prob < 0.3:
        return "Baixo", "🟢"
    elif prob < 0.7:
        return "Médio", "🟡"
    else:
        return "Alto", "🔴"