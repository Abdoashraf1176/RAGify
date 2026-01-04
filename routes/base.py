from fastapi import FastAPI, APIRouter, Depends
from helpers.config import get_settings
from motor.motor_asyncio import AsyncIOMotorClient

base_router = APIRouter()


@base_router.get("/")
async def welcome():
    app_settings = get_settings()
    app_name = app_settings.APP_NAME
    app_version = app_settings.APP_VERSION
    print('_________________________', app_name)
    return {
        "app_name": app_name,
        "app_version": app_version,

    }
