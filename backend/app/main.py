from fastapi import FastAPI
from app.routes.risk import router as risk_router

app = FastAPI(
    title="TrafficGuard AI API",
    description="Backend API for traffic risk prediction and police deployment.",
    version="0.1.0",
)

# Include API Routers
app.include_router(risk_router)


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