import joblib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("C:\\Users\\nando\\Desktop\\spotify-sentiment-analysis\\model\\spotify_sentiment_model.pkl")


class ReviewRequest(BaseModel):
    review: str


@app.get("/")
def home():
    return {"message": "Spotify Sentiment Analysis API"}


@app.post("/predict")
def predict_sentiment(request: ReviewRequest):

    prediction = model.predict([request.review])[0]

    probabilities = model.predict_proba([request.review])[0]

    results = {
        str(class_name): round(float(probability) * 100, 2)
        for class_name, probability in zip(model.classes_, probabilities)
    }

    return {
        "review": request.review,
        "sentiment": str(prediction),
        "probabilities": results
    }