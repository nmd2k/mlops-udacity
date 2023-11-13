import os
import uvicorn
from fastapi import FastAPI, APIRouter

from fastapi.middleware.cors import CORSMiddleware
import model.model
from app.routers.prediction import router as prediction_router

app = FastAPI()

origins = str(os.environ.get("CORS_ALLOW_ORIGINS")).split(",") if \
          os.environ.get("CORS_ALLOW_ORIGINS") else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    pass

# Configure a router
router = APIRouter()
router.include_router(prediction_router)
app.include_router(router)

uvicorn.Config(app, log_level="debug", access_log=True)
