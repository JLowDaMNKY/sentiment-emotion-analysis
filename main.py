# libraries used for computer vision / emotion analysis
import cv2
from deepface import DeepFace

# libraries used for natural language processing / sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import caller

# libraries used for ui
import threading
import PySimpleGUI as sg

# libraries used for logging
import time
import logging

class Main:

    def __init__(self):
        super().__init__()

        self.init_ui()
        self.emote_capture = threading.Thread(target=self.emotion, daemon=True)
        self.audio_capture = threading.Thread(target=self.audio_input, daemon=True)

        self.vid = cv2.VideoCapture(0)

        self.create_log()

        self.ui_helper()



    def create_log(self):
        log_dir = "Final Project/logs/log.txt"
        with open(log_dir, 'w') as f:
            f.write(str(time.localtime))

        logging.basicConfig(filename=log_dir, level=logging.DEBUG, format="%(asctime)s %(message)s", filemode='w')
        self.logger=logging.getLogger() 
        self.logger.setLevel(logging.DEBUG)



    def init_ui(self):
        sg.theme('DarkAmber')
        layout = [
            [sg.Text("OpenCV Demo", size=(60, 1), justification="center")],
            [sg.Image(filename="", key="-IMAGE-")],
            [sg.Text('Emotion: ', size=(60, 1), justification="center", key="-EMOTION-")],
            [sg.Button("Start Session", size=(10, 1))],
            [sg.Button("Exit", size=(10, 1))],    
            ]
        self.window = sg.Window("OpenCV Integration", layout, location=(800, 400))


    def ui_helper(self):
        while True:   
            self.event, self.values = self.window.read(timeout=20) #type: ignore
            if self.event == "Exit" or self.event == sg.WIN_CLOSED:
                self.vid.release()
                break
            if self.event == "Start Session":
                self.emote_capture.start()
                self.audio_capture.start()

    def emotion(self):
        while True:
            ret, frame = self.vid.read()
            
            try:
                result = DeepFace.analyze(frame,actions=['emotion'], prog_bar=False)
                self.emotion_value = max(result['emotion'])
                self.real_emotion = max(result['emotion'], key=result['emotion'].get)
                self.window["-EMOTION-"].update("Emotion: " + self.real_emotion)
            except:
                pass

            imgbytes = cv2.imencode(".png", frame)[1].tobytes()
            self.window["-IMAGE-"].update(data=imgbytes)


    def audio_input(self):
        while True:
            audio = caller.recognize_from_microphone() # use azure speech-to-text to get audio from microphone
            sentiment = SentimentIntensityAnalyzer() 
            audio_sentiment = sentiment.polarity_scores(audio) # use vadersentiment to get sentiment of audio 
            print(audio) # Return sentiment score of audio
            self.logger.info("{}: {} {}".format(time.asctime(time.localtime()), audio, audio_sentiment))
                

if __name__ == '__main__':
    main = Main()
    