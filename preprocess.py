import os
import shutil
import cv2
import moviepy.editor

def preprocess(vidName):
    data_folder_path = os.getcwd()+"/data"
    if os.path.exists(data_folder_path):
        shutil.rmtree(data_folder_path)
    
    res_folder_path = os.getcwd() + "/results/predrnn/test_result"
    if(os.path.exists(res_folder_path)):
        shutil.rmtree(res_folder_path)

    path = os.getcwd()+"/data/comedian/videos/rem"
    os.makedirs(path)

    c = 1
    while c<=3:
        text2 = str(c).rjust(3, '0')
        foldername = "video"+text2
        os.makedirs(os.getcwd()+"/data/comedian/videos/" + foldername)
        c+=1

    i = 196
    limit = 196

    while i<=limit:
        vid_name = str(vidName)
        video_path = os.getcwd()+'/vid/'+vid_name
        vidcap = cv2.VideoCapture(video_path)
        print(vid_name)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_count = (vidcap.get(cv2.CAP_PROP_FRAME_COUNT) // 10) * 10
        print(frame_count, "fps",fps)
        if i == limit: #Always using last video for testing
            
            metadata = open(os.getcwd()+"/data/comedian/videos/rem/md.txt","w")
            metadata.write(str(fps)+","+str(int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) % 10))
            metadata.close()
            #Audio
            audio_save_path = os.getcwd()+"/data/comedian/videos/rem/"
            audio_filename = "audio"

            video = moviepy.editor.VideoFileClip(video_path)
            audio = video.audio
            audio.write_audiofile(audio_save_path + audio_filename + ".mp3")

        success,image = vidcap.read()
        resize = cv2.resize(image, (128, 128), cv2.IMREAD_UNCHANGED)
        count = 1
        rem_num = 1
        while success:
            if frame_count > 0:
                text = str(count).rjust(5, '0')
                text2 = str(i-193).rjust(3, '0')
                cv2.imwrite(os.getcwd()+"/data/comedian/videos/video"+ text2 +"/frame_"+text+".jpg", resize)     # save frame as JPEG file      
                success,image = vidcap.read()
                if success:
                    resize = cv2.resize(image, (128, 128), cv2.IMREAD_UNCHANGED)
                #print('Read a new frame: ', success)
                count += 1
                frame_count-=1
            elif i == limit:
                cv2.imwrite(os.getcwd()+"/data/comedian/videos/rem/"+str(rem_num)+".jpg", resize)     # save frame as JPEG file      
                success,image = vidcap.read()
                if success:
                    resize = cv2.resize(image, (128, 128), cv2.IMREAD_UNCHANGED)
                #print('Read a new frame: ', success)
                count += 1
                rem_num+=1
            else:
                break
        i+=1

        #Construction of Partial Video
        con_vid_path = "converted_vid/"+str(vidName)
        conv_vid = cv2.VideoCapture(con_vid_path)
        conv_fps =  conv_vid.get(cv2.CAP_PROP_FPS)
        conv_frame_cnt = conv_vid.get(cv2.CAP_PROP_FRAME_COUNT)
        print("Conv frame cnt:",conv_frame_cnt,"fps:",fps)
        rem_count = conv_frame_cnt % 10
        img_arr = []
        rem_frames = []
        for _ in range(int(conv_frame_cnt - rem_count)):
            success,image = conv_vid.read()
            resize = cv2.resize(image, (128, 128), cv2.IMREAD_UNCHANGED)
            img_arr.append(resize)

        for _ in range(int(rem_count)):
            success,image = conv_vid.read()
            resize = cv2.resize(image, (128, 128), cv2.IMREAD_UNCHANGED)
            rem_frames.append(resize)

        save_path =  'data/comedian/videos/rem/partial_video.mp4'
        out = cv2.VideoWriter(save_path ,cv2.VideoWriter_fourcc(*'DIVX'), conv_fps, (128,128))
        remove = 0
        cnt = 0
        for j in img_arr:
            if(remove == 0):
                out.write(j)
            cnt+=1
            if(cnt==5):
                remove = 1 - remove
                cnt = 0
        for j in rem_frames:
            out.write(j)
        out.release()
        print("Partial Video Saved Successfully !")

        #Returns partial video (frame_count,file_size)
        
        return cv2.VideoCapture(save_path).get(cv2.CAP_PROP_FRAME_COUNT),os.path.getsize(save_path)

if __name__ == '__main__':
    print(preprocess('320.mp4'))
