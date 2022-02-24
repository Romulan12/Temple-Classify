import os
from PIL import Image
from exceptionHandler import raiseException
from torchvision import transforms

ALLOWED_EXTENSIONS = ["png", "jpg", "jpeg", "tif", "tiff"]

class ImageLoader:
    def __init__(self, input_image_path, image_shape):
        """ 
            Purpose: Creates object for ImageLoader Class
            Input: ImagePath, Shape to which image should be transformed
            Output: Object for ImageLoader class
        """
        self.data_transform = transforms.Compose([
                transforms.RandomResizedCrop(image_shape),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.input_image_path = input_image_path
        self.image_label = "O"
        self.processed_image = self.processImage()

    def processImage(self):
        """ 
            Purpose: Transforms the image in a tensor that can be processed by the model
            Input: Image Path
            Output: Processed Tensor
        """
        image = Image.open(self.input_image_path)
        processed_image = self.data_transform(image).unsqueeze(0)
        return processed_image

def processDocument(document_path, logger):
    """ 
        Purpose: Takes a files and checks if it is valid or not; only files with predefined extensions are allowed
        Input: Image Path
        Output: List with acceptable Document Path
    """
    processed_image_list = []
    file_extension = document_path.split(".")[-1].lower()
    if(file_extension in ALLOWED_EXTENSIONS):
        processed_image_list.append(document_path)
    else:
        logger.info(document_path + " not processed due to invalid extension")
    return processed_image_list

def processDirectory(input_directory, logger):
    """ 
        Purpose: Takes the input path directory and returns a list of images present
        Input: Input Path; logger
        Output: list of images present in the directory
    """
    image_file_list = []
    logger.info("Parsing files present in the folder " + input_directory)
    logger.info(str(len(os.listdir(input_directory))) + " files present in the directory")
    for content in os.listdir(input_directory):
        file_path = os.path.join(input_directory, content)
        if(os.path.isdir(file_path)):
            image_file_list += processDirectory(file_path, logger)
        else:
            try:
                image_file_list += processDocument(file_path, logger)
            except Exception as e:
                logger.exception(file_path + " not processed")
                logger.exception(str(e))
    return image_file_list

def processInputImage(input_image_directory, image_shape, logger):
    """ 
        Purpose: Takes the input path directory and returns a list of tensors for model prediction
        Input: Input Path; Shape in which model accepts the image; logger
        Output: list of ImageLoader object with image name and processed tensor
    """
    try:
        image_list = processDirectory(input_image_directory, logger)
        preprocessed_tensor_list = []
        for image_name in image_list:
            image_handler = ImageLoader(image_name, image_shape)
            preprocessed_tensor_list.append(image_handler)
    except:
        logger.exception("Unable to proprocess images")
        raise raiseException("Unable to proprocess images")

    if(len(preprocessed_tensor_list) == 0):
        logger.exception("No Valid Image Found in the Directory " + input_directory)
        raise raiseException("No Valid Image Found in the Directory " + input_directory)
    return preprocessed_tensor_list




