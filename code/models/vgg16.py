import torch
from torchvision import models
import torch.nn as nn
import os

import configparser

config = configparser.RawConfigParser()
config.read(os.path.join(os.getcwd(),"config/config.property"))

LABELS = dict(config.items("LABELS"))
CLASS_LABEL_LIST = LABELS["labels"].split(",")

class vgg16ModelHandler:
    """ 
        Purpose: Creates object for Model Class
        Input: Pretrained weight of the model, Number of classes, Device for prediction 
        Output: Object for Model class
    """
    def __init__(self, pretrained_weight_path, number_of_classes, device, logger):
        self.number_of_classes = number_of_classes
        self.device = device
        self.pretrained_weight_path = pretrained_weight_path
        self.logger = logger
        self.logger.info("Creating Model")
        self.model = models.vgg16()
        self.logger.info("Model Created Successfully")

    def loadModelWeights(self):
        """ 
            Purpose: Transforms the final layer of the model and loads the pretrained weights
            Input: None
            Output: None
        """
        self.logger.info("Loading Model Weights from file " + self.pretrained_weight_path)
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, 11)
        model_path = torch.load(self.pretrained_weight_path,map_location=torch.device(self.device))
        self.model.load_state_dict(model_path)
        self.logger.info("Successfully Loaded Model")


    def getClassName(self, predicted_label):
        """ 
            Purpose: Gives the class from the predicted index
            Input: Predicted Label
            Output: Predicted Class
        """
        return CLASS_LABEL_LIST[predicted_label]

    def predictModel(self, processed_image):
        """ 
            Purpose: Predicts the result for a image tensor
            Input: Image tensor
            Output: Class predicted for the image
        """
        self.model.eval()
        with torch.no_grad():
            processed_image = processed_image.to(self.device)
            output_prediction = self.model(processed_image)
            _, predicted_label = torch.max(output_prediction, 1)
            predicted_class = self.getClassName(predicted_label[0])
        return predicted_class