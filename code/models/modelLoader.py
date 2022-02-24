import logging 
import os
import torch

from models.vgg16 import vgg16ModelHandler
from models.efficientnetB3 import efficientNetB3ModelHandler
from exceptionHandler import raiseException

import configparser

config = configparser.RawConfigParser()
config.read(os.path.join(os.getcwd(),"config/config.property"))

LABELS = dict(config.items("LABELS"))
CLASS_LABEL_LIST = LABELS["labels"].split(",")
NUMBER_OF_LABELS = len(CLASS_LABEL_LIST)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def loadPredictionModel(args, logger):
    """ 
        Purpose: Loads the model selected by the user
        Input: User arguments for model and pretrained path
        Output: Prediction model
    """
    selected_model = args.model
    pretrained_weight_path = args.pretrained_path
    if(selected_model == "VGG16"):
        try:
            model = vgg16ModelHandler(pretrained_weight_path, NUMBER_OF_LABELS, DEVICE, logger)
        except:
            raiseException("Unable to create Model class")
    elif(selected_model == "EFFB3"):
        try:
            model = efficientNetB3ModelHandler(pretrained_weight_path, NUMBER_OF_LABELS, DEVICE, logger)
        except:
            raiseException("Unable to create Model class")

    else:
        logger.exception("Invalid Model Choice")
        raise raiseException("Invalid Model Choice "+args.model+" is not allowed")

    try:
        model.loadModelWeights()
    except Exception as e:
        logger.exception("Unable to load Model Weights")
        raise raiseException("Unable to load Model Weights")
    return model


def getModelPrediction(temple_prediction_model, processed_image_tensor_list,logger):
    """ 
        Purpose: Predicts the results for the list of processed tensor
        Input: Prediction Model, List of processed tensor
        Output: List of dictionary with document name and Predicted class
    """
    output_prediction = []
    logger.info("Starting Model Prediction")
    for processed_image_tensor in processed_image_tensor_list:
        try:
            predicted_class = temple_prediction_model.predictModel(processed_image_tensor.processed_image)
            output_prediction.append({"ImageName":processed_image_tensor.input_image_path.split("/")[-1],"Prediction":predicted_class})
        except Exception as e:
            logger.exception("Unable to get prediction for file "+processed_image_tensor.input_image_path.split("/")[-1])
            logger.exception(str(e))
    logger.info("Model Prediction Done")
    return output_prediction


