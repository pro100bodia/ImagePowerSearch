from fastapi import FastAPI, UploadFile, File
from typing import Optional, Annotated

import cv
import repository

app = FastAPI()


@app.post("/upload-image")
def upload_image(image: Annotated[bytes, File()]):
    return cv.retrieve_information(image)


@app.patch("/who-is")
def name_a_person(person_id: int, name: str):
    return repository.alter_person(person_id, name)


@app.get("/search")
def search_image(query: str):
    return repository.search(query)

@app.get("/image/{image_id}")
def get_image(image_id: int):
    return repository.get_image(image_id)
