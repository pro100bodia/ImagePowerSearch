from abc import ABC, abstractmethod
from typing import Optional, List
from darknetpy.detector import Detector
from consul import Consul

import cv2
import re
import face_recognition
import tensorflow as tf
import tensorflow_hub as hub
from fastapi import UploadFile
from pydantic import BaseModel
import torch
import torchvision
import torchvision.transforms as T

import repository


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

    def retrieve(self, image_file: UploadFile):
        consul_client = Consul(host='localhost', port=8500)

        model = consul_client.kv.get("object_detection/model")[1]['Value']
        model = re.sub("b|\'", "", str(model))

        try:
            if model is "yolo":
                detector = Detector('models/coco.data',
                                    'models/yolo.cfg',
                                    'models/yolo.weights')

                results = detector.detect(image_file)
            else:
                model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
                model.eval()

                image = cv2.imread(image_file)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

                transform = T.Compose([T.ToTensor()])
                input_image = transform(image)
                input_image = input_image.unsqueeze(0)

                with torch.no_grad():
                    predictions = model(input_image)

                boxes = predictions[0]['boxes']
                labels = predictions[0]['labels']

                results = []
                for label in labels:
                    class_name = model.class_names[label]
                    results.append(class_name)
        except:
            results = []

        return results


class FacesInformationRetriever(BaseInformationRetriever):
    name = 'faces'

    def retrieve(self, image: UploadFile):
        try:
            known_image = face_recognition.load_image_file("photos/test.JPEG")
            unknown_image = face_recognition.load_image_file("emo.JPEG")

            known_encoding = face_recognition.face_encodings(known_image)[0]
            unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

            results = face_recognition.compare_faces([known_encoding], unknown_encoding)

        except:
            results = []

        return results


class CaptionsInformationRetriever(BaseInformationRetriever):
    name = 'captions'

    def retrieve(self, image_file: UploadFile):
        try:
            model = hub.load("https://tfhub.dev/google/show_and_tell/1")

            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image_tensor = tf.convert_to_tensor(image)
            image_tensor = tf.expand_dims(image_tensor, axis=0)

            captions = model(image_tensor)
            caption = captions['captions'][0].numpy().decode()

            results = {caption}
        except:
            results = []

        return results


retrievers = {ObjectInformationRetriever(), FacesInformationRetriever(), CaptionsInformationRetriever()}


def retrieve_information(image):
    image_model = ImageModel()
    for retriever in retrievers:
        setattr(image_model, retriever.name, retriever.retrieve(image))

    repository.save_information(image, image_model)

    return image_model
