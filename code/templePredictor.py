import os
import configparser
import argparse
import logging
import torch
import pandas as pd

from models.modelLoader import loadPredictionModel, getModelPrediction
from dataloader.dataloader import processInputImage
from exceptionHandler import raiseException
from argumentParser import parseArguments

if(not os.path.isdir(os.path.join(os.getcwd(),"logs/"))):
    os.mkdir(os.path.join(os.getcwd(),"logs/"))

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] [%(process)d]  %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

LOG_FILE_PATH = os.path.join(os.getcwd(), "logs/templePrediction.log")
fileHandler = logging.FileHandler(LOG_FILE_PATH)
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler) 

def writePredictionOutput(predicted_output, output_file_directory):
    """ 
        Purpose: Save the output in the user decided folder
        Input: List of dictionary with image name and prediction, directory where file should be saved
        Output: Path of the output csv
    """
    try:
        logger.info("Saving the results in the following directory " + output_file_directory)
        output_file_path = os.path.join(output_file_directory,"prediction.csv")
        prediction_df = pd.DataFrame(predicted_output)
        prediction_df.to_csv(output_file_path, index = False)
        logger.info("Output file save in the following path " + output_file_path)
    except Exception as e:
        logger.exception("Unable to save the results")
        raise raiseException("Unable to save the results")
    return output_file_path

def main():
    """ 
        Purpose: Main function for execution
        Input: None
        Output: None
    """
    args = parseArguments(logger)
    temple_prediction_model = loadPredictionModel(args, logger)
    processed_image_tensor = processInputImage(args.input_path, args.image_shape, logger)
    predicted_output = getModelPrediction(temple_prediction_model, processed_image_tensor, logger)
    output_file_path = writePredictionOutput(predicted_output, args.output_dir)
    return None

if __name__ == "__main__":
    main()