import sys 
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    exc_type, exc_obj, exc_tb = sys.exc_info()  # ✅ Fix unpacking error
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error in script: {file_name}, Line {exc_tb.tb_lineno}, Message: {str(error)}"
    return error_message
    

class CustomException (Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message