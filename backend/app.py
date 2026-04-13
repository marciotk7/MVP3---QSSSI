"""
API FastAPI para Classificação de Reviews de Jogos

Este backend carrega o modelo de Machine Learning treinado e disponibiliza
um endpoint para classificar reviews como positivas ou negativas.

Endpoints:
- GET /: Informações da API
- GET /health: Status de saúde da API
- POST /predict: Classificar uma review
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re
import os

#importar NLTK (pode falhar se não estiver instalado)
try:
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except:
    NLTK_AVAILABLE = False
    stop_words = set()
    lemmatizer = None

# Inicializar FastAPI
app = FastAPI(
    title="Game Reviews Classifier API",
    description="API para classificação de reviews de jogos usando Machine Learning",
    version="1.0.0"
)

# Configurar CORS para permitir requisições do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carregar modelo e vetorizador
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "model", "vectorizer.pkl")

model = None
vectorizer = None

def load_model():
    """Carrega o modelo e vetorizador do disco."""
    global model, vectorizer
    
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            print("✅ Modelo e vetorizador carregados com sucesso!")
            return True
        else:
            print("⚠️ Arquivos do modelo não encontrados.")
            print(f"   Esperado: {MODEL_PATH}")
            print(f"   Esperado: {VECTORIZER_PATH}")
            return False
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return False

# Carregar modelo na inicialização
load_model()


# ============================================================
# Schemas Pydantic
# ============================================================

class ReviewInput(BaseModel):
    """Schema para entrada de review."""
    text: str
    game_id: str = None
    game_name: str = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "This game is absolutely amazing! Best RPG ever!",
                "game_id": "570",
                "game_name": "Dota 2"
            }
        }

class PredictionOutput(BaseModel):
    """Schema para saída da predição."""
    text: str
    cleaned_text: str
    sentiment: str
    sentiment_code: int
    confidence: float
    game_info: dict = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "This game is absolutely amazing!",
                "cleaned_text": "game absolutely amazing",
                "sentiment": "Positivo",
                "sentiment_code": 1,
                "confidence": 0.95,
                "game_info": {
                    "game_id": "570",
                    "game_name": "Dota 2",
                    "game_image": "https://cdn.akamai.steamstatic.com/steam/apps/570/header.jpg"
                }
            }
        }

class HealthResponse(BaseModel):
    """Schema para resposta de health check."""
    status: str
    model_loaded: bool
    vectorizer_loaded: bool


# ============================================================
# Funções auxiliares
# ============================================================

def preprocess_text(text: str) -> str:
    """
    Pré-processa o texto para classificação.
    
    Etapas:
    1. Converter para minúsculas
    2. Remover pontuação e caracteres especiais
    3. Remover stopwords
    4. Aplicar lemmatization
    
    Args:
        text: Texto original
        
    Returns:
        Texto pré-processado
    """
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remover pontuação e números
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Tokenizar
    tokens = text.split()
    
    # 4. Remover stopwords e aplicar lemmatization
    if NLTK_AVAILABLE and lemmatizer:
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    else:
        # Fallback simples se NLTK não estiver disponível
        tokens = [word for word in tokens if len(word) > 2]
    
    return ' '.join(tokens)


# ============================================================
# Endpoints
# ============================================================

@app.get("/")
async def root():
    """Endpoint raiz com informações da API."""
    return {
        "name": "Game Reviews Classifier API",
        "version": "1.0.0",
        "description": "API para classificação de reviews de jogos",
        "endpoints": {
            "GET /": "Informações da API",
            "GET /health": "Status de saúde",
            "POST /predict": "Classificar uma review"
        },
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Verifica o status de saúde da API."""
    return HealthResponse(
        status="healthy" if model and vectorizer else "degraded",
        model_loaded=model is not None,
        vectorizer_loaded=vectorizer is not None
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict_sentiment(review: ReviewInput):
    """
    Classifica o sentimento de uma review de jogo.
    
    Args:
        review: Objeto contendo o texto da review
        
    Returns:
        Predição com sentimento (Positivo/Negativo) e confiança
    """
    # Verificar se modelo está carregado
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Execute o notebook para gerar os arquivos do modelo."
        )
    
    # Validar entrada
    if not review.text or len(review.text.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="O texto da review não pode estar vazio."
        )
    
    try:
        # Pré-processar texto
        cleaned_text = preprocess_text(review.text)
        
        # Vetorizar
        text_vectorized = vectorizer.transform([cleaned_text])
        
        # Fazer predição
        prediction = model.predict(text_vectorized)[0]
        
        # Tentar obter probabilidade (nem todos os modelos suportam)
        try:
            probabilities = model.predict_proba(text_vectorized)[0]
            confidence = float(max(probabilities))
        except:
            
            confidence = 0.85
        
        # Montar resposta
        sentiment = "Positivo" if prediction == 1 else "Negativo"
        
        # Adicionar informações do jogo se fornecidas
        game_info = None
        if review.game_id and review.game_name:
            game_info = {
                "game_id": review.game_id,
                "game_name": review.game_name,
                "game_image": f"https://cdn.akamai.steamstatic.com/steam/apps/{review.game_id}/header.jpg"
            }
        
        return PredictionOutput(
            text=review.text,
            cleaned_text=cleaned_text,
            sentiment=sentiment,
            sentiment_code=int(prediction),
            confidence=confidence,
            game_info=game_info
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar predição: {str(e)}"
        )


@app.post("/reload-model")
async def reload_model():
    """Recarrega o modelo do disco."""
    success = load_model()
    
    if success:
        return {"status": "success", "message": "Modelo recarregado com sucesso!"}
    else:
        raise HTTPException(
            status_code=500,
            detail="Falha ao recarregar modelo. Verifique se os arquivos existem."
        )


# ============================================================
# Execução
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
