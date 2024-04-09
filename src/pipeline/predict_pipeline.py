import sys

from src.exceptions import CustomException
from src.logger import logging
from src.components.model_trainer import Summarizer


class PredictSummary:
    def __init__(self) -> None:
        self.summarizer = Summarizer()

    def predict(self, text):
        try:
            summary = self.summarizer.summarize(text)
            logging.info('PIPELINE: Text Summarized')

            return summary
        
        except Exception as e:
            raise CustomException(e, sys)