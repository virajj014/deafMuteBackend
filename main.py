from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import tensorflow as tf
from typing import List
import io
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="Sign Language Classifier API")

# Load your trained model and class mappings
MODEL_PATH = "sign_language_model.h5"
CLASSES_PATH = "classes.npy"  # Saved from your training script

# Load model and classes
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = np.load(CLASSES_PATH, allow_pickle=True)
except Exception as e:
    raise RuntimeError(f"Failed to load model or classes: {str(e)}")

print(f"Model loaded with {len(class_names)} classes: {list(class_names)}")

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Resize image from center and preprocess for model prediction"""
    try:
        # Read image from bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")
        
        # Convert to RGB (model was trained with RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get center crop and resize to 50x50
        h, w = img.shape[:2]
        size = min(h, w)
        start_h = (h - size) // 2
        start_w = (w - size) // 2
        cropped = img[start_h:start_h+size, start_w:start_w+size]
        resized = cv2.resize(cropped, (50, 50))
        
        # Normalize and add batch dimension
        processed = (resized.astype(np.float32) / 255.0)[np.newaxis, ...]
        return processed
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")

@app.post("/predict", response_model=List[str])
async def predict(files: List[UploadFile] = File(...)):
    """Endpoint that accepts multiple images and returns predictions"""
    try:
        start_time = datetime.now()
        predictions = []
        
        for file in files:
            # Read image file
            contents = await file.read()
            
            # Preprocess image
            try:
                processed_img = preprocess_image(contents)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            
            # Make prediction
            pred = model.predict(processed_img)
            class_idx = np.argmax(pred[0])
            class_name = class_names[class_idx]
            predictions.append(class_name)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        print(f"Processed {len(files)} images in {processing_time:.2f} seconds")
        
        return predictions
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": True, "classes": list(class_names)}

if __name__ == "__main__":
    import uvicorn
    # Get port from environment variable or default to 8000
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)