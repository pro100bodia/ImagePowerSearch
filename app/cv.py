from pydantic import BaseModel
from typing import Optional, List
from fastapi import UploadFile
from abc import ABC, abstractmethod
from datetime import datetime

import repository
import pytz

timezone = pytz.timezone('Europe/Istanbul')


class ImageModel(BaseModel):
    objects: Optional[List[str]] = []
    faces: Optional[List[str]] = []
    captions: Optional[str] = "None of the captions were found yet"


class BaseInformationRetriever(ABC):
    name: None

    @abstractmethod
    def retrieve(self, image: UploadFile):
        pass


class ObjectInformationRetriever(BaseInformationRetriever):
    name = 'objects'

    def retrieve(self, image: UploadFile):
        timestamp = datetime.now(timezone).strftime('%H:%M')
        return {"guy" + timestamp, "dinosaur" + timestamp}


class FacesInformationRetriever(BaseInformationRetriever):
    name = 'faces'

    def retrieve(self, image: UploadFile):
        url = 'http://localhost:8000/who-is?person_id=1'
        timestamp = datetime.now(timezone).strftime('%H:%M')
        return {"dude" + timestamp}


class CaptionsInformationRetriever(BaseInformationRetriever):
    name = 'captions'

    def retrieve(self, image: UploadFile):
        return "A guy standing next to a dinosaur"


retrievers = {ObjectInformationRetriever(), FacesInformationRetriever(), CaptionsInformationRetriever()}


def retrieve_information(image):
    image_model = ImageModel()
    for retriever in retrievers:
        setattr(image_model, retriever.name, retriever.retrieve(image))

    repository.save_information(image, image_model)

    return image_model
