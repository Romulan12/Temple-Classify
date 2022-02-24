class raiseException(Exception):
    """ 
        Purpose: Stops the exection upon an issue
        Input: error message 
        Output: None
    """
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message