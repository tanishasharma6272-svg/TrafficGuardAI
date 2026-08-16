from fastapi import FastAPI

app = FastAPI(
    title="TrafficGuard AI API",
    description="Backend API for traffic risk prediction and police deployment.",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "message": "TrafficGuard AI API is running",
        "status": "ok",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }