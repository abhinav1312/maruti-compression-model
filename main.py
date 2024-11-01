from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import tempfile
import os

# Import the VideoClassifier class from the detection model file
from detection_model import VideoClassifier

app = FastAPI()

# Initialize your model (adjust as needed)
video_classifier = VideoClassifier('detection_model.h5')

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