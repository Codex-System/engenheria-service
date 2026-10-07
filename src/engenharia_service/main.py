from fastapi import FastAPI
from src.engenharia_service.api.routes import router

app = FastAPI(title="Motor de IA e Visão - Construtora Alfa")

app.include_router(router, prefix="/api/v1")

