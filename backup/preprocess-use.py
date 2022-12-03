import os
import shutil
import cv2
import moviepy.editor
import numpy as np

data_folder_path = os.getcwd()+"/data"
if os.path.exists(data_folder_path):
    shutil.rmtree(data_folder_path)

path = os.getcwd()+"/data/comedian/videos/rem"
os.makedirs(path)

c = 1
while c<=999:
    text2 = str(c).rjust(3, '0')
    foldername = "video"+text2
    os.makedirs(os.getcwd()+"/data/comedian/videos/" + foldername)
    c+=1

i = 196

vid_name = str(i)+".mp4"
video_path = os.getcwd()+'/vid/'+vid_name
vidcap = cv2.VideoCapture(video_path)
print(vid_name)
fps = vidcap.get(cv2.CAP_PROP_FPS)
frame_count = (vidcap.get(cv2.CAP_PROP_FRAME_COUNT) // 10) * 10
print(frame_count, "fps",fps)

#Write Metadata
metadata = open(os.getcwd()+"/data/comedian/videos/rem/md.txt","w")
metadata.write(str(fps)+","+str(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) % 10))
metadata.close()
#Audio
audio_save_path = os.getcwd()+"/data/comedian/videos/rem/"
audio_filename = "audio"

#Write audio
video = moviepy.editor.VideoFileClip(video_path)
audio = video.audio
audio.write_audiofile(audio_save_path + audio_filename + ".mp3")

success,image = vidcap.read()
resize = cv2.resize(image, (128, 128), cv2.IMREAD_UNCHANGED)

count = 1
rem_num = 1
folder = 1

#For tracking which frames to skip
tracker = 0
track_counter = 0

while success:
    if frame_count > 0:
        if(tracker == 0):
            text = str(count).rjust(5, '0')
            text2 = str(folder).rjust(3, '0')
            cv2.imwrite(os.getcwd()+"/data/comedian/videos/video"+ text2 +"/frame_"+text+".jpg", resize)     # save frame as JPEG file      
            success,image = vidcap.read()
            if success:
                resize = cv2.resize(image, (128, 128), cv2.IMREAD_UNCHANGED)
            #print('Read a new frame: ', success)
            count += 1
            frame_count-=1
            track_counter+=1
            if(track_counter == 5):
                tracker = 1-tracker
                track_counter = 0
                folder += 1
                count = 1
        else:
            frame_count-=1
            success,image = vidcap.read()
            if success:
                resize = cv2.resize(image, (128, 128), cv2.IMREAD_UNCHANGED)
            track_counter+=1
            if(track_counter==5):
                tracker = 1 - tracker
                track_counter = 0
    else:
        cv2.imwrite(os.getcwd()+"/data/comedian/videos/rem/"+str(rem_num)+".jpg", resize)     # save frame as JPEG file      
        success,image = vidcap.read()
        if success:
            resize = cv2.resize(image, (128, 128), cv2.IMREAD_UNCHANGED)
            #print('Read a new frame: ', success)
            count += 1
            rem_num+=1

black_frame = np.zeros((128, 128, 3), dtype = "uint8")



for fol in range(1,folder):
    img_no = 6
    text2 = str(fol).rjust(3, '0')
    fol_path = os.getcwd()+"/data/comedian/videos/video"+ text2
    for _ in range(5):
        text = str(img_no).rjust(5, '0')
        cv2.imwrite(fol_path+"/frame_"+text+".jpg",black_frame)
        img_no+=1
