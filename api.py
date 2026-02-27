from fastapi import FastAPI, UploadFile, File
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import uvicorn

app = FastAPI()

model = load_model("brain_model.h5")

class_names = ["glioma", "meningioma", "notumor", "pituitary"]

def preprocess(image):
    image = image.resize((224,224))
    image = np.array(image)/255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = preprocess(image)

    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    predicted_class = class_names[class_index]

    if predicted_class == "notumor":
        risk = "Low"
    elif confidence > 0.85:
        risk = "High"
    elif confidence > 0.6:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "prediction": predicted_class,
        "confidence": round(confidence,2),
        "risk": risk
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)