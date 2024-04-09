from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import PredictSummary
from src.pipeline.data_transformation_pipeline import DataTransformationPipeline
from src.utils import load_audio_file
from src.config import CHANNELS, FRAME_RATE

application = Flask(__name__)

app = application

#Route for a home page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/summarize', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    
    else:
        audio_file = request.files['audioFile']
    
        if audio_file.filename == '':
            return "No selected file"
        
        mp3 = load_audio_file(file_path=audio_file, channels=CHANNELS, frame_rate=FRAME_RATE)

        data_transformer = DataTransformationPipeline()
        transcript = data_transformer.transform_data(mp3)
        print('\n\nTRANSCRIPT:\n', transcript)
        
        summarizer = PredictSummary()
        summary = summarizer.predict(transcript)
        print('\n\nSUMMARY:\n', summary)

        return render_template('home.html', summary=summary)

@application.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)