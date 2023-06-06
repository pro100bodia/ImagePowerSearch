from pydantic import BaseModel
from typing import Optional, List
from fastapi import UploadFile
from abc import ABC, abstractmethod
import face_recognition
# import darknet
import cv2
import torch
import torchvision
import torchvision.transforms as T
import tensorflow as tf
import tensorflow_hub as hub

import repository
import pytz


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
        try:
            # config_file = "models/yolov4.cfg"
            # weights_file = "models/yolov4.weights"
            # names_file = "models/coco.names"
            #
            # network, class_names, class_colors = darknet.load_network(
            #     config_file,
            #     names_file,
            #     weights_file,
            #     batch_size=1
            # )
            #
            # image = cv2.imread(image_file)
            #
            # sized = cv2.resize(image, (darknet.network_width(network), darknet.network_height(network)))
            # darknet_image = darknet.make_image(darknet.network_width(network), darknet.network_height(network), 3)
            # darknet.copy_image_from_bytes(darknet_image, sized.tobytes())
            #
            # detections = darknet.detect_image(network, class_names, darknet_image)
            #
            # results = []
            # for detection in detections:
            #     class_label = detection[0].decode()  # Decoding the class label bytes to string
            #     results.append(class_label)

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
            image_tensor = tf.expand_dims(image_tensor, axis=0)  # Add batch dimension

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
