from pydub import AudioSegment

def load_audio_file(file_path, channels, frame_rate):
    mp3 = AudioSegment.from_mp3(file_path)
    mp3 = mp3.set_channels(channels)
    mp3 = mp3.set_frame_rate(frame_rate)
    return mp3


def save_transcriptions(file_path, transcript_text):
    with open(file_path, 'w') as file:
        file.write(transcript_text)


def read_transcriptions(file_path):
    with open(file_path) as file:
        transcript = file.read()

    return transcript