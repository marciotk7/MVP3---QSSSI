#  Classificação de Reviews de Jogos com Machine Learning

##  Descrição do Projeto

MVP que classifica automaticamente reviews de jogos como **positivas** ou **negativas** utilizando técnicas clássicas de Machine Learning.

### Como funciona?

1. **Treinamento**: O modelo aprende padrões em milhares de reviews rotuladas
2. **Vetorização TF-IDF**: Transforma texto em números que o modelo entende
3. **Classificação**: 4 algoritmos são comparados (KNN, Árvore de Decisão, Naive Bayes, SVM)
4. **Predição**: O melhor modelo é usado para classificar novas reviews

---

## Estrutura do Projeto

```
Marcio MVP/
├── notebook/
│   └── game_reviews_classification.ipynb  # Notebook do Google Colab
├── backend/
│   ├── app.py                             # API FastAPI
│   ├── model/
│   │   ├── sentiment_model.pkl            # Modelo treinado (gerado pelo notebook)
│   │   └── vectorizer.pkl                 # Vetorizador TF-IDF (gerado pelo notebook)
│   └── requirements.txt                   # Dependências do backend
├── frontend/
│   ├── index.html                         # Interface web
│   ├── style.css                          # Estilos
│   └── script.js                          # Lógica do frontend
├── tests/
│   ├── test_model.py                      # Testes automatizados
│   └── requirements.txt                   # Dependências de teste
├── data/
│   └── README.md                          # Instruções sobre o dataset
└── README.md                              # Este arquivo
```

---

##  Como Executar

### 1. Notebook (Google Colab)

1. Abra o arquivo `notebook/game_reviews_classification.ipynb` no Google Colab
2. Execute todas as células em ordem
3. O modelo treinado será exportado para `model/sentiment_model.pkl`

### 2. Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

A API estará disponível em: `http://localhost:8000`

### 3. Frontend

Abra o arquivo `frontend/index.html` no navegador ou use um servidor local:

```bash
cd frontend
python -m http.server 3000
```

Acesse: `http://localhost:3000`

### 4. Testes

```bash
cd tests
pip install -r requirements.txt
pytest test_model.py -v
```

---

## 📊 Métricas e Resultados

| Modelo | Acurácia | Precisão | Recall | F1-Score |
|--------|----------|----------|--------|----------|
| KNN | - | - | - | - |
| Árvore de Decisão | - | - | - | - |
| Naive Bayes | - | - | - | - |
| SVM | - | - | - | - |

> **Nota**: Os valores serão preenchidos após executar o notebook.

---

## 🔍 Explicação do Ranking/Classificação

### Por que TF-IDF?
- **TF (Term Frequency)**: Conta quantas vezes uma palavra aparece no texto
- **IDF (Inverse Document Frequency)**: Penaliza palavras muito comuns
- **Resultado**: Palavras importantes para o sentimento têm peso maior

### Por que esses 4 modelos?
1. **KNN**: Classifica baseado nos vizinhos mais próximos
2. **Árvore de Decisão**: Cria regras de decisão interpretáveis
3. **Naive Bayes**: Probabilístico, bom para texto
4. **SVM**: Encontra a melhor fronteira de separação

### Critério de Escolha
O modelo com **maior F1-Score** é selecionado, pois equilibra precisão e recall.

---

## 📝 Requisitos Atendidos

- [x] Notebook com dataset, pré-processamento e treinamento
- [x] Comparação de 4 modelos clássicos
- [x] Exportação do melhor modelo
- [x] API para predição
- [x] Frontend simples e funcional
- [x] Testes automatizados com threshold de 80%
- [x] Documentação clara

---

## Marcio Barros

Projeto desenvolvido para demonstração de Machine Learning aplicado a análise de sentimentos em reviews de jogos.
