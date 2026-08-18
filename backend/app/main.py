import os

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

# Allow local dev server + deployed frontend to call the API.
allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# Add production frontend URL(s) via env var (comma-separated if more than one)
prod_origins = os.getenv("FRONTEND_URL", "")
if prod_origins:
    allowed_origins.extend([origin.strip() for origin in prod_origins.split(",") if origin.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
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
