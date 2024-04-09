import sys

from src.exceptions import CustomException
from src.logger import logging
from src.components.model_trainer import AudioModel
from src.components.data_transformation import DataTransformation

class DataTransformationPipeline:
    def __init__(self) -> None:
        self.audio_model = AudioModel()
        self.data_transformer = DataTransformation()

    def transform_data(self, mp3):
        try:
            transcript = self.audio_model.initiate_audio_recognition(mp3)
            logging.info('PIPELINE: Transcript Created')

            punctuated_text = self.data_transformer.punctuate_and_capitalize_text(transcript)
            logging.info('PIPELINE: Punctuation and Capitalization Done')

            segmented_transcript = self.data_transformer.tokenize_and_segment_data(punctuated_text)
            logging.info('PIPELINE: Transcript Segmented')

            return segmented_transcript
        
        except Exception as e:
            raise CustomException(e, sys)