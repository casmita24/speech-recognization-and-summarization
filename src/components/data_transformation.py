import sys

from src.exceptions import CustomException
from src.logger import logging
from src.config import PUNCTUATION_MODEL, SUMMARIZATION_MODEL, MAX_SEGMENT_LENGTH

from dataclasses import dataclass
from transformers import pipeline, AutoTokenizer
from transformers import T5Tokenizer, TFT5ForConditionalGeneration, AutoModelForSeq2SeqLM

@dataclass
class DataTransformationConfig:
    punctuation_model = PUNCTUATION_MODEL
    summarization_model = SUMMARIZATION_MODEL
    max_segment_length = MAX_SEGMENT_LENGTH


class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def punctuate_and_capitalize_text(self, text):
        try:
            # pipe = pipeline("text2text-generation", model="SJ-Ray/Re-Punctuate")
            # logging.info('Punctuation Pipeline created')
            # punctuated_text = pipe(text)

            # result = punctuated_text[0]['generated_text']
            # logging.info('Punctuated Text Generated')

            # return result

            # tokenizer = T5Tokenizer.from_pretrained('SJ-Ray/Re-Punctuate')
            # model = TFT5ForConditionalGeneration.from_pretrained('SJ-Ray/Re-Punctuate')

            tokenizer = AutoTokenizer.from_pretrained("SJ-Ray/Re-Punctuate")
            model = AutoModelForSeq2SeqLM.from_pretrained("SJ-Ray/Re-Punctuate", from_tf=True)

            # Define input text
            input_text = 'the story of this brave brilliant athlete whose very being was questioned so publicly is one that still captures the imagination'

            # Encode input text
            input_ids = tokenizer.encode("punctuate: " + input_text, return_tensors="pt") 

            # Generate output
            result = model.generate(input_ids)

            # Decode and print the result
            output_text = tokenizer.decode(result[0], skip_special_tokens=True)
            return output_text
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def tokenize_and_segment_data(self, text):
        try:
            tokenizer = AutoTokenizer.from_pretrained(DataTransformationConfig.summarization_model)
            logging.info('Tokenizer created')
            # Tokenize the transcript
            tokenized_transcript = tokenizer.tokenize(text)
            logging.info('Text Tokeinzed')

            segmented_transcript = []
            for i in range(0, len(tokenized_transcript), DataTransformationConfig.max_segment_length):
                segment = tokenized_transcript[i:i + DataTransformationConfig.max_segment_length]
                segmented_transcript.append(tokenizer.convert_tokens_to_string(segment))
            
            logging.info('Text Segmented')

            return segmented_transcript
        
        except Exception as e:
            raise CustomException(e, sys)
