from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import tempfile
import logging
import os
from detection_model import VideoClassifier


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Forces TensorFlow to use the CPU

# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Import the VideoClassifier class from the detection model file


# Initialize your model (adjust as needed)
model_path = 'detection_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

video_classifier = VideoClassifier(model_path)

@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Upload Video for Prediction</title>
    </head>
    <body>
        <h1>Upload a Video for Prediction</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="video/*" required>
            <button type="submit">Upload and Predict</button>
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    if file.content_type not in ["video/mp4", "video/avi", "video/mov"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")

    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_file:
        temp_file.write(await file.read())
        temp_file.flush()

        try:
            final_class, average_probability = video_classifier.process_video(temp_file.name)
            response = {
                "prediction": final_class,
                "probability": float(average_probability)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    return JSONResponse(content=response)