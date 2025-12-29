from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from analyze_resume import analyze_resume  # your new ML code

app = FastAPI()

origins = [
    "https://pro-front-final.vercel.app",  # your Vercel frontend URL
    "http://localhost:3000",               # for local testing
]
# Enable CORS for your Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,         # allow requests from these origins
    allow_credentials=True,
    allow_methods=["*"],           # allow all HTTP methods
    allow_headers=["*"],           # allow all headers
)


@app.post("/predict")
async def predict_career(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        # Read PDF bytes directly
        result = analyze_resume(file.file)

        return {
            "success": True,
            "data": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/")
async def root():
    return {"message": "Resume Analysis API is running!"}
