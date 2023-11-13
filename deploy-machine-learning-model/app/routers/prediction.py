"""utilities for business logic"""
import sys
sys.path.append(["../"])

import logging
from pydantic import BaseModel, Field

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.business.prediction import InferenceModel
from app.exception.custom_exception import CustomException


logger = logging.getLogger(__name__)
router = APIRouter(tags=["Model functions"])


class InputData(BaseModel):
    """Define input data"""
    # Using the first row of census.csv as sample
    age: int = Field(None, example=39)
    workclass: str = Field(None, example='State-gov')
    fnlgt: int = Field(None, example=77516)
    education: str = Field(None, example='Bachelors')
    education_num: int = Field(None, example=13)
    marital_status: str = Field(None, example='Never-married')
    occupation: str = Field(None, example='Adm-clerical')
    relationship: str = Field(None, example='Not-in-family')
    race: str = Field(None, example='White')
    sex: str = Field(None, example='Female')
    capital_gain: int = Field(None, example=2174)
    capital_loss: int = Field(None, example=0)
    hours_per_week: int = Field(None, example=40)
    native_country: str = Field(None, example='United-States')

@router.get("/")
async def get_items():
    return {"message": "Hello, welcome to our app!"}


@router.post("/predict")
async def predict(data: InputData):
    """Logic of prediction function"""
    try:
        model = InferenceModel()
        result = await model.predict(data.dict())

        return JSONResponse(
            content=result,
            status_code=200
        )
    except CustomException as exc:
        return JSONResponse(
            content={
                "message": str(exc)
            },
            status_code=500
        )
