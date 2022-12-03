import numpy as np
import os
import cv2
from PIL import Image
import logging
import random

logger = logging.getLogger(__name__)

class InputHandle:
    def __init__(self, datas, indices, input_param):
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        self.image_width = input_param['image_width']
        self.datas = datas
        self.indices = indices
        self.current_position = 0
        self.current_batch_indices = []
        self.current_input_length = input_param['seq_length']

    def total(self):
        return len(self.indices)

    def begin(self, do_shuffle=True):
        logger.info("Initialization for read data ")
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def next(self):
        self.current_position += self.minibatch_size
        if self.no_batch_left():
            return None
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def no_batch_left(self):
        if self.current_position + self.minibatch_size > self.total():
            return True
        else:
            return False

    def get_batch(self):
        if self.no_batch_left():
            logger.error(
                "There is no batch left in " + self.name + ". Consider to user iterators.begin() to rescan from the beginning of the iterators")
            return None
        input_batch = np.zeros(
            (self.minibatch_size, self.current_input_length, self.image_width, self.image_width, 3)).astype(
            self.input_data_type)
        for i in range(self.minibatch_size):
            batch_ind = self.current_batch_indices[i]
            begin = batch_ind
            end = begin + self.current_input_length
            data_slice = self.datas[begin:end, :, :, :]
            # print("data_slice", data_slice.shape)
            input_batch[i, :self.current_input_length, :, :, :] = data_slice
            
        input_batch = input_batch.astype(self.input_data_type)
        return input_batch

    def print_stat(self):
        logger.info("Iterator Name: " + self.name)
        logger.info("    current_position: " + str(self.current_position))
        logger.info("    Minibatch Size: " + str(self.minibatch_size))
        logger.info("    total Size: " + str(self.total()))
        logger.info("    current_input_length: " + str(self.current_input_length))
        logger.info("    Input Data Type: " + str(self.input_data_type))

class DataProcess:
    def __init__(self, input_param):
        self.paths = input_param['paths']
        self.category = ['videos']
        self.image_width = input_param['image_width']
        self.input_length = input_param['input_length']

        self.train_videos =["%03d" % x for x in range(1,input_param['train_end_no']+1)] 
        self.test_videos = ["%03d" % x for x in range(input_param['train_end_no']+1,input_param['test_end_no']+1)] 
        self.use_videos = ["%03d" % x for x in range(input_param['use_start_no'],input_param['use_end_no']+1)] 

        self.input_param = input_param
        self.seq_len = input_param['seq_length']

    def load_data(self, paths, mode='test'):
        '''
        frame -- action -- video_seq(a dir)
        :param paths: action_path list
        :return:
        '''

        path = paths[0]
        if mode == 'train':
            video_id = self.train_videos
        elif mode == 'test':
            video_id = self.test_videos
        elif mode == 'use':
            video_id = self.use_videos
        else:
            print("ERROR!")
        print('begin load data' + str(path))

        frames_np = []
        frames_file_name = []
        frames_video_mark = []
        frames_category = []
        video_mark = 0
        dup_frames_np = []
        dup_frames_file_name = []
        dup_frames_video_mark = []
        dup_frames_category = []

        c_dir_list = self.category
        frame_category_flag = -1
        for c_dir in c_dir_list: 
            if c_dir in self.category:
                frame_category_flag = 1 
            else:
                print("category error!!!")

            c_dir_path = os.path.join(path, c_dir)
            p_c_dir_list = os.listdir(c_dir_path)

            for p_c_dir in p_c_dir_list: 
                if p_c_dir[5:8] not in video_id:
                    continue
                if mode == 'train':
                    print("%%%%%%%%%%______________________________Training______________________________%%%%%%%%%%")
                else:
                    print("%%%%%%%%%%______________________________Testing______________________________%%%%%%%%%%")
                print(f"%%%%%%%%%%______________________________video {p_c_dir[5:8]}______________________________%%%%%%%%%%")
                video_mark += 1
                dir_path = os.path.join(c_dir_path, p_c_dir)
                filelist = os.listdir(dir_path)
                filelist = os.listdir(dir_path)
                filelist.sort()
                for file in filelist: 
                    frame_im = cv2.imread(os.path.join(dir_path, file), cv2.IMREAD_COLOR)
                    frame_np = np.array(frame_im)  # (1000, 1000) numpy array
                    if mode == 'use':
                        filler = np.zeros((128, 128, 3), dtype = "uint8")
                        ext = file[len(file)-4:]
                        back_file = file[6:len(file)-4]
                        back_file = "%05d" % (int(back_file)+self.input_length)
                        dup_frames_np.append(filler)
                        dup_frames_file_name.append("frame_"+back_file+ext)
                        dup_frames_video_mark.append(video_mark)
                        dup_frames_category.append(frame_category_flag)
                    frames_np.append(frame_np)
                    frames_file_name.append(file)
                    frames_video_mark.append(video_mark)
                    frames_category.append(frame_category_flag)
        if mode == 'use':
            frames_np.extend(dup_frames_np)
            frames_file_name.extend(dup_frames_file_name)
            frames_video_mark.extend(dup_frames_video_mark)
            frames_category.extend(dup_frames_category)
        indices = []
        index = len(frames_video_mark) - 1
        while index >= self.seq_len - 1:
            if frames_video_mark[index] == frames_video_mark[index - self.seq_len + 1]:
                end = int(frames_file_name[index][6:11])
                start = int(frames_file_name[index - self.seq_len + 1][6:11])
                if end - start == self.seq_len - 1:
                    indices.append(index - self.seq_len + 1)
                    if frames_category[index] == 1:
                        index -= self.seq_len - 1
                    elif frames_category[index] == 2:
                        index -= 2
                    else:
                        print("category error 2 !!!")
            index -= 1
        frames_np = np.asarray(frames_np)
        data = np.zeros((frames_np.shape[0], self.image_width, self.image_width , 3))
        for i in range(len(frames_np)):
            temp = np.float32(frames_np[i, :, :, :])
            data[i,:,:,:]=cv2.resize(temp,(self.image_width,self.image_width))/255
        indices = indices[::-1]
        print("there are " + str(data.shape[0]) + " pictures")
        print("there are " + str(len(indices)) + " sequences")
        return data, indices

    def get_train_input_handle(self):
        train_data, train_indices = self.load_data(self.paths, mode='train')
        return InputHandle(train_data, train_indices, self.input_param)

    def get_test_input_handle(self):
        test_data, test_indices = self.load_data(self.paths, mode='test')
        return InputHandle(test_data, test_indices, self.input_param)
    
    def get_use_input_handle(self):
        test_data, test_indices = self.load_data(self.paths, mode='use')
        return InputHandle(test_data, test_indices, self.input_param)