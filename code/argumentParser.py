import argparse
import configparser
import os

from exceptionHandler import raiseException

config = configparser.RawConfigParser()
config.read(os.path.join(os.getcwd(),"config/config.property"))

DEFAULT_VALUES = dict(config.items("DEFAULT"))
DEFAULT_MODEL = DEFAULT_VALUES["model"]
DEFAULT_PRETRAINED_MODEL_PATH = os.path.join(os.getcwd(),DEFAULT_VALUES["pretrained_model_path"])
DEFAULT_OUTPUT_DIRECTORY = os.path.join(os.getcwd(),DEFAULT_VALUES["output_directory"])
DEFAULT_INPUT_SHAPE = int(DEFAULT_VALUES["input_shape"])

def parseArguments(logger):
    """ 
        Purpose: Processes the arguments given by the user
        Input: Logger
        Output: Arguments given by the user
    """
    try:
        logger.info("Loading User Arguments")
        parser = argparse.ArgumentParser(description='PyTorch Semantic Video Segmentation training')
        parser.add_argument('--input_path', type=str, required=True, help="Path to the directory containing input images")
        parser.add_argument('--model', type=str, default = DEFAULT_MODEL, help = "Model to be used for prediction; Default: EFFB3")
        parser.add_argument('--pretrained_path', type=str, default = DEFAULT_PRETRAINED_MODEL_PATH, help="Path of the pretrained Model; Default: models/temple-classifier-eff-best.pt")
        parser.add_argument('--output_dir', type=str, default = DEFAULT_OUTPUT_DIRECTORY, help="Path where output csv is shared; Default: data/output/")
        parser.add_argument('--image_shape', type=int, default = DEFAULT_INPUT_SHAPE, help="Shape of Input Image; Default is 300")
        args = parser.parse_args()
        logger.info("Input Path: " + args.input_path)
        logger.info("Evaluation Model: " + args.model)
        logger.info("Pretrained Model Path: " + args.pretrained_path)
        logger.info("Output Directory: " + args.output_dir)
        logger.info("Size of Input Image: 3*" + str(args.image_shape)+"*"+str(args.image_shape))
        return args
    except:
        logger.exception("Exception while loading user arguments")
        raise raiseException("Exception while loading user arguments")