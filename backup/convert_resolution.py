import sys
import cv2

def convert_vid(vid_no):
    vid_path = 'vid/'+ vid_no + ".mp4"
    vidcap = cv2.VideoCapture(vid_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    img_arr = []
    for _ in range(int(frame_count)):
        success,image = vidcap.read()
        resize = cv2.resize(image, (128, 128), cv2.IMREAD_UNCHANGED)
        img_arr.append(resize)
    final_op_path = 'converted_vid/'+vid_no+".mp4"
    out = cv2.VideoWriter(final_op_path ,cv2.VideoWriter_fourcc(*'DIVX'), fps, (128,128))
    for i in img_arr:
        out.write(i)
    out.release()

if __name__ == '__main__':
    sz = len(sys.argv)
    if(sz < 2):
        print("Please pass the video numbers (space seperated) an argument. Ex - python convert_resolution.py 196 198")
        sys.exit(1)
    
    for i in range(1,sz):
        convert_vid(sys.argv[i])
    

