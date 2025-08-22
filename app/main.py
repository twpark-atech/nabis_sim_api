# app/main.py
from fastapi import FastAPI
from app.api.v1.stations import router as stations_router
from app.api.v1.routes import router as routes_router
from app.api.v1.paths import router as paths_router
from app.api.v1.scenarios import router as scenarios_router
from app.api.v1.profile import router as profile_router

app = FastAPI(title="Simulation API", version="0.1.0")

app.include_router(stations_router, prefix="/api/v1/stations", tags=["stations"])
app.include_router(routes_router, prefix="/api/v1/routes", tags=["routes"])
app.include_router(paths_router, prefix="/api/v1/paths", tags=["paths"])
app.include_router(scenarios_router, prefix="/api/v1/scenarios", tags=["scenarios"])
app.include_router(profile_router, prefix="/api/v1/profile", tags=["profile"])