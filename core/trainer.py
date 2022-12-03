import os
import os.path
import datetime
import cv2
import numpy as np
from skimage import metrics as m
from core.utils import preprocess, metrics
import lpips
import torch
import sys
from moviepy.editor import *


loss_fn_alex = lpips.LPIPS(net='alex')



def train(model, ims, real_input_flag, configs, itr):
    cost = model.train(ims, real_input_flag)
    if configs.reverse_input:
        ims_rev = np.flip(ims, axis=1).copy()
        cost += model.train(ims_rev, real_input_flag)
        cost = cost / 2

    if itr % configs.display_interval == 0:
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: ' + str(itr))
        print('training loss: ' + str(cost))

def vid_gen(test_input_handle, configs, mode="test"):
  n=len(test_input_handle.indices)
  k=configs.input_length
  l=configs.total_length - configs.input_length
  img_array=[]
  txt = open(os.getcwd()+"/data/comedian/videos/rem/md.txt","r")
  metadata = txt.readline().split(',')
  fps=float(metadata[0])
  for i in range(1,n+1):
      for j in range(1,k+1):
          if mode=="test":
            img_path=os.getcwd()+'/results/predrnn/test_result/'+str(i)+"/gt"+str(j)+".png"
          else:
            img_path=os.getcwd()+'/results/predrnn/prediction_result/'+str(i)+"/gt"+str(j)+".png"
          if os.path.exists(img_path):
              img=cv2.imread(img_path)
              img_array.append(img)
          else:
              print("doesnot exist"+img_path)
      for h in range(l+1,k+l+1):
          if mode=="test":
            img_path=os.getcwd()+'/results/predrnn/test_result/'+str(i)+"/pd"+str(h)+".png"
          else:
            img_path=os.getcwd()+'/results/predrnn/prediction_result/'+str(i)+"/pd"+str(h)+".png"
          if os.path.exists(img_path):
              img=cv2.imread(img_path)
              img_array.append(img)
          else:
              print("doesnot exist"+img_path)
      # break
  video_path = os.getcwd()+'/results/predrnn/final_vid.mp4'
  out = cv2.VideoWriter(video_path ,cv2.VideoWriter_fourcc(*'DIVX'), fps, (128,128))
  path = os.getcwd()+'/data/comedian/videos/rem/'
  for i in range(int(metadata[1])):
    img=cv2.imread(path+str(i+1)+".jpg")
    img_array.append(img)
  for i in range(len(img_array)):
      out.write(img_array[i])
  out.release()
  #Adding audio to the generated video
  audio_path = os.getcwd()+"/data/comedian/videos/rem/audio.mp3"
  video_save_path = os.getcwd()+"/results/predrnn/"
  final_vid_filename = "final_with_audio"

  clip = VideoFileClip(video_path)
  #clip = clip.subclip(0, 4) #Custom Vid length to crop
  audioclip = AudioFileClip(audio_path)
  #audioclip = audioclip.subclip(0,4) #Custom Audio Len to Crop
  videoclip = clip.set_audio(audioclip)
  final_clip = videoclip.set_fps(fps)
  final_clip.write_videofile(video_save_path+final_vid_filename+".mp4")

def test(model, test_input_handle, configs, itr):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')
    test_input_handle.begin(do_shuffle=False)
    res_path = os.path.join(configs.gen_frm_dir, str(itr))
    os.mkdir(res_path)
    avg_mse = 0
    batch_id = 0
    img_mse, ssim, psnr = [], [], []
    lp = []

    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)
        lp.append(0)

    # reverse schedule sampling
    if configs.reverse_scheduled_sampling == 1:
        mask_input = 1
    else:
        mask_input = configs.input_length

    real_input_flag = np.zeros(
        (configs.batch_size,
         configs.total_length - mask_input - 1,
         configs.img_width // configs.patch_size,
         configs.img_width // configs.patch_size,
         configs.patch_size ** 2 * configs.img_channel))

    if configs.reverse_scheduled_sampling == 1:
        real_input_flag[:, :configs.input_length - 1, :, :] = 1.0

    while (test_input_handle.no_batch_left() == False):
        batch_id = batch_id + 1
        test_ims = test_input_handle.get_batch()
        test_dat = preprocess.reshape_patch(test_ims, configs.patch_size)
        # test_ims = test_ims[:, :, :, :, :configs.img_channel]
        img_gen = model.test(test_dat, real_input_flag)

        img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        output_length = configs.total_length - configs.input_length 
        img_out = img_gen[:, -output_length:]

        # MSE per frame
        for i in range(output_length):
            x = test_ims[:, i + configs.input_length, :, :, :]
            gx = img_out[:, i, :, :, :]
            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            mse = np.square(x - gx).sum()
            img_mse[i] += mse
            avg_mse += mse
            # cal lpips
            img_x = np.zeros([configs.batch_size, 3, configs.img_width, configs.img_width])
            if configs.img_channel == 3:
                img_x[:, 0, :, :] = x[:, :, :, 0]
                img_x[:, 1, :, :] = x[:, :, :, 1]
                img_x[:, 2, :, :] = x[:, :, :, 2]
            else:
                img_x[:, 0, :, :] = x[:, :, :, 0]
                img_x[:, 1, :, :] = x[:, :, :, 0]
                img_x[:, 2, :, :] = x[:, :, :, 0]
            img_x = torch.FloatTensor(img_x)
            img_gx = np.zeros([configs.batch_size, 3, configs.img_width, configs.img_width])
            if configs.img_channel == 3:
                img_gx[:, 0, :, :] = gx[:, :, :, 0]
                img_gx[:, 1, :, :] = gx[:, :, :, 1]
                img_gx[:, 2, :, :] = gx[:, :, :, 2]
            else:
                img_gx[:, 0, :, :] = gx[:, :, :, 0]
                img_gx[:, 1, :, :] = gx[:, :, :, 0]
                img_gx[:, 2, :, :] = gx[:, :, :, 0]
            img_gx = torch.FloatTensor(img_gx)
            lp_loss = loss_fn_alex(img_x, img_gx)
            lp[i] += torch.mean(lp_loss).item()

            real_frm = np.uint8(x * 255)
            pred_frm = np.uint8(gx * 255)

            psnr[i] += metrics.batch_psnr(pred_frm, real_frm)
            for b in range(configs.batch_size):
                score, _ = m.structural_similarity(pred_frm[b], real_frm[b], full=True, multichannel=True)
                ssim[i] += score

        # save prediction examples
        frameSize = (128, 128)
        if batch_id <= configs.num_save_samples:
            path = os.path.join(res_path, str(batch_id))
            os.mkdir(path)
            for i in range(configs.total_length):
                name = 'gt' + str(i + 1) + '.png'
                file_name = os.path.join(path, name)
                img_gt = np.uint8(test_ims[0, i, :, :, :] * 255)
                cv2.imwrite(file_name, img_gt)
            for i in range(output_length):
                name = 'pd' + str(i + 1 + configs.input_length) + '.png'
                file_name = os.path.join(path, name)
                img_pd = img_out[0, i, :, :, :]
                img_pd = np.maximum(img_pd, 0)
                img_pd = np.minimum(img_pd, 1)
                img_pd = np.uint8(img_pd * 255)
                cv2.imwrite(file_name, img_pd)
        test_input_handle.next()
    vid_gen(test_input_handle, configs, mode="test")

    avg_mse = avg_mse / (batch_id * configs.batch_size)
    print('mse per seq: ' + str(avg_mse))
    for i in range(configs.total_length - configs.input_length):
        t=img_mse[i] / (batch_id * configs.batch_size)
        print("AVG MSE for "+str(i+6)+"th frame = "+str(t))


    ssim = np.asarray(ssim, dtype=np.float32) / (configs.batch_size * batch_id)
    print('ssim per frame: ' + str(np.mean(ssim)))
    for i in range(configs.total_length - configs.input_length):
        print("AVG SSIM for "+str(i+6)+"th frame = "+str(ssim[i]))

    psnr = np.asarray(psnr, dtype=np.float32) / (batch_id)
    print('psnr per frame: ' + str(np.mean(psnr)))
    for i in range(configs.total_length - configs.input_length):
        print("AVG PSRN for "+str(i+6)+"th frame = "+str(psnr[i]))

    lp = np.asarray(lp, dtype=np.float32) / (batch_id)
    print('lpips per frame: ' + str(np.mean(lp)))
    for i in range(configs.total_length - configs.input_length):
        print("AVG LPIPS for "+str(i+6)+"th frame = "+str(lp[i]))
    
    met  = {}
    met['mse'] = str(avg_mse)
    met['ssim'] = str(np.mean(ssim))
    met['psnr'] = str(np.mean(psnr))
    met['lp'] = str(np.mean(lp))
    return met

def use(model, use_input_handle, configs, itr):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'prediction...')
    use_input_handle.begin(do_shuffle=False)
    res_path = os.path.join(configs.gen_frm_dir, str(itr))
    os.mkdir(res_path)
    avg_mse = 0
    batch_id = 0
    img_mse, ssim, psnr = [], [], []
    lp = []

    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)
        lp.append(0)

    # reverse schedule sampling
    if configs.reverse_scheduled_sampling == 1:
        mask_input = 1
    else:
        mask_input = configs.input_length

    real_input_flag = np.zeros(
        (configs.batch_size,
         configs.total_length - mask_input - 1,
         configs.img_width // configs.patch_size,
         configs.img_width // configs.patch_size,
         configs.patch_size ** 2 * configs.img_channel))

    if configs.reverse_scheduled_sampling == 1:
        real_input_flag[:, :configs.input_length - 1, :, :] = 1.0

    while (use_input_handle.no_batch_left() == False):
        batch_id = batch_id + 1
        test_ims = use_input_handle.get_batch()
        test_dat = preprocess.reshape_patch(test_ims, configs.patch_size)
        # test_ims = test_ims[:, :, :, :, :configs.img_channel]
        img_gen = model.test(test_dat, real_input_flag)

        img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        output_length = configs.total_length - configs.input_length 
        img_out = img_gen[:, -output_length:]

        # save prediction examples
        frameSize = (128, 128)
        if batch_id <= configs.num_save_samples:
            path = os.path.join(res_path, str(batch_id))
            os.mkdir(path)
            for i in range(configs.input_length):
                name = 'gt' + str(i + 1) + '.png'
                file_name = os.path.join(path, name)
                img_gt = np.uint8(test_ims[0, i, :, :, :] * 255)
                cv2.imwrite(file_name, img_gt)
            for i in range(output_length):
                name = 'pd' + str(i + 1 + configs.input_length) + '.png'
                file_name = os.path.join(path, name)
                img_pd = img_out[0, i, :, :, :]
                img_pd = np.maximum(img_pd, 0)
                img_pd = np.minimum(img_pd, 1)
                img_pd = np.uint8(img_pd * 255)
                cv2.imwrite(file_name, img_pd)
        use_input_handle.next()
    vid_gen(use_input_handle, configs, mode="pred")