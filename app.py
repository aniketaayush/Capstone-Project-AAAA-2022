from flask import Flask, render_template, request
from flask import jsonify
from flask_cors import CORS, cross_origin
from preprocess import preprocess
from run_server import model_wrapper
import time
import cv2
import os

def model(filename):
    return {'genreatedFileName':filename}

app = Flask(__name__)
cors = CORS(app)
@cross_origin()
@app.route('/generateVideo', methods =["POST"])
def hello():
    # print(request.json)
    # resp=model(request.json['fileName'])
    # return jsonify(resp)
    start = time.perf_counter()
    input_frm_cnt, input_vid_size = preprocess(request.json['fileName'])
    metrics = model_wrapper()
    end = time.perf_counter()

    vid_path = "converted_vid_noaudio/" + str(request.json['fileName'])
    vidcap = cv2.VideoCapture(vid_path)
    final_frm_cnt = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    final_vid_size = os.path.getsize(vid_path)
    
    percent_size_reduction = ((final_vid_size - input_vid_size)/final_vid_size)*100
    print("Video Data: ", input_frm_cnt,input_vid_size,final_frm_cnt,final_vid_size,percent_size_reduction,end-start)
    
    metrics["input_frm_cnt"] = str(input_frm_cnt)
    metrics["input_vid_size"] = str(input_vid_size/1024) #KB
    metrics["final_frm_cnt"] = str(final_frm_cnt)
    metrics["final_vid_size"] = str(final_vid_size/1024)
    metrics["time"] = str(end-start)
    metrics["percent_size_reduction"] = str(percent_size_reduction)
    metrics["frames_generated"] = str(final_frm_cnt - input_frm_cnt)

    return jsonify({"resp": 'final_with_audio.mp4','metrics': metrics})

app.run(debug=True)
