from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import uvicorn

# Create FastAPI app
app = FastAPI(title="NLP Model API")

# Load the trained model
try:
    model = joblib.load('model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
except:
    raise Exception("Model files not found. Make sure to save your trained model first.")

# Define input schema
class TextInput(BaseModel):
    text: str

# Define prediction endpoint
@app.post("/predict")
async def predict(input_data: TextInput):
    try:
        # Vectorize the input text
        text_vectorized = vectorizer.transform([input_data.text])
        
        # Make prediction
        prediction = model.predict(text_vectorized)
        probability = model.predict_proba(text_vectorized).max()
        
        return {
            "prediction": int(prediction[0]),
            "probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 