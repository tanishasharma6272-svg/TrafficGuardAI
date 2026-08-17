from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.risk import router as risk_router
from app.routes.ml_risk import router as ml_risk_router
from app.routes.ml_explanation import router as ml_explanation_router
from app.routes.deployment import router as deployment_router

# FastAPI application initialization
app = FastAPI(
    title="TrafficGuard AI API",
    description="Backend API for traffic risk prediction and police deployment.",
    version="0.1.0",
)

# Allow the local React/Vite development server to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers (legacy risk, ML risk, SHAP explanation, and deployment routers)
app.include_router(risk_router)
app.include_router(ml_risk_router)
app.include_router(ml_explanation_router)
app.include_router(deployment_router)


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