import json
import sys

from vosk import Model, KaldiRecognizer
from dataclasses import dataclass
from transformers import pipeline

from src.config import FRAME_RATE, CHANNELS, VOSK_MODEL
from src.exceptions import CustomException
from src.logger import logging


@dataclass
class AudioModelConfig:
    frame_rate = FRAME_RATE
    channels = CHANNELS
    model_name = VOSK_MODEL

class AudioModel:
    def __init__(self) -> None:
        self.audio_model_config = AudioModelConfig()

    def initiate_audio_recognition(self, mp3_audio):
        try:
            #Initializing Model and Recognizer
            model = Model(model_name=self.audio_model_config.model_name)
            rec = KaldiRecognizer(model, self.audio_model_config.frame_rate)
            rec.SetWords(True)
            logging.info('Recognizer Created')

            # Transcribing audio
            step = 45000
            transcript = ""
            for i in range(0, len(mp3_audio), step):
                print(f"Progress: {i/len(mp3_audio)}")
                segment = mp3_audio[i:i+step]
                rec.AcceptWaveform(segment.raw_data)
                result = rec.Result()
                text = json.loads(result)["text"]
                transcript += text

            logging.info('Audio Transcripted')
            return transcript
        
        except Exception as e:
            raise CustomException(e,sys)\
            

class Summarizer():
    def __init__(self) -> None:
        pass

    def summarize(self, text):
        try:
            summarizer = pipeline('summarization')
            summaries = summarizer(text)
            
            summary = ""
            for item in summaries:
                summary += item['summary_text'] + "\n\n"

            return summary
        
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    from src.utils import load_audio_file
    from src.components.data_transformation import DataTransformation

    mp3 = load_audio_file(r"C:\Users\ashmi\OneDrive\Documents\Projects\speech_summarizer\speech_recognition_marketplace.mp3", channels=1, frame_rate=16000)


    obj=AudioModel()
    transcript = obj.initiate_audio_recognition(mp3)
    print('\n\nTRANSCRIPT: ', transcript)

    trans_obj = DataTransformation()
    punctuated_text = trans_obj.punctuate_and_capitalize_text(transcript)
    print('\n\nPUNCTUATED TEXT: ', punctuated_text)
    segmented_transcript = trans_obj.tokenize_and_segment_data(transcript)
    print('\n\nSEGMENTED TRANSCRIPT:', segmented_transcript)

    summ = Summarizer()
    summary = summ.summarize(segmented_transcript)
    print('\n\nSUMMARY: ', summary)

