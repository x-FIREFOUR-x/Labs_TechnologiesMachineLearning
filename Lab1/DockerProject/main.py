from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = FastAPI()

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="resnet_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# CIFAR-10 class labels
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((227, 227))
    image = np.array(image).astype("float32")
    image = tf.image.per_image_standardization(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image format")

    image_bytes = await file.read()

    try:
        input_data = preprocess_image(image_bytes)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        predicted_index = int(np.argmax(output_data))
        predicted_label = class_names[predicted_index]
        confidence = float(np.max(output_data))

        return JSONResponse({
            "predicted_class": predicted_label,
            "confidence": confidence
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

'''
Run server, use command in cmd>
    uvicorn main:app --reload
    
make predict, use command in cmd>
    curl -X POST "http://127.0.0.1:8000/predict" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@dog.jpg"
'''