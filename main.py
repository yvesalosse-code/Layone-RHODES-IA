from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="API Classification Texte")

# On charge le modèle une seule fois au démarrage
model = joblib.load("texte_model.pkl")

# Ce que l'utilisateur doit envoyer
class Demande(BaseModel):
    texte: str

@app.get("/")
def accueil():
    return {"message": "API en ligne ! "}

@app.post("/predict")
def predict(demande: Demande):
    # Le pipeline fait tout seul : texte -> chiffres -> prédiction
    prediction = model.predict([demande.texte])[0]
    return {
        "texte_recu": demande.texte,
        "categorie": prediction
    }