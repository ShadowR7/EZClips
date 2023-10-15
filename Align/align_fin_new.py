import numpy as np
import cv2
import time
import subprocess
import multiprocessing
import os

from moviepy.editor import ImageSequenceClip


# from method import img_combine
# from method import up_sample as us


'''
这是修改过后版本的对齐，结尾可以倒着读取帧，而不需要从视频中间往后读
'''


def align_fin(original_frames, clip_length, seq):
    # image_files = sorted(os.listdir(self.image_folder), key=lambda x: int(x.split('.')[0]))
    # print(len(image_files))
    frame_nums = []
    non_zero_counts = []
    ref_frames = []
    kernel = cv2.getGaussianKernel(3, 0)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    diffs = []

    if seq:
        # 读取前5张图片作为静态背景图像
        for i in range(5):
            # image_path = os.path.join(self.image_folder, image_files[i])
            frame = original_frames[i]
            # frame = cv2.imread(image_path)
            if frame is not None:
                ref_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            else:
                print("无法读取图片")
                break

        for i in range(clip_length - 1):
            # image_path = os.path.join(self.image_folder, image_files[i])
            frame = original_frames[i]
                #cv2.imread(image_path)

            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(ref_frames[0], gray)

                diff = cv2.filter2D(diff, -1, kernel)

                _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2)

                frame_nums.append(i)
                non_zero_counts.append(cv2.countNonZero(thresh))

                ref_frames.pop(0)
                ref_frames.append(gray)
            else:
                break

    else:
        print("需要倒着读取图片")
        ref_frames = []
        for i in range(5, 0, -1):
            # image_path = os.path.join(self.image_folder, image_files[-i])
            # frame = cv2.imread(image_path)
            frame = original_frames[-i]
            if frame is not None:
                ref_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            else:
                break

        for i in range(len(original_frames) - 6, clip_length - 1, -1):
            index = i
            # image_path = os.path.join(self.image_folder, image_files[index])
            # print(image_path)
            # frame = cv2.imread(image_path)
            frame = original_frames[index]
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(ref_frames[0], gray)

                diff = cv2.filter2D(diff, -1, kernel)

                _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2)

                frame_nums.append(index)
                non_zero_counts.append(cv2.countNonZero(thresh))

                if len(non_zero_counts) > 1:
                    diff = non_zero_counts[-1] - non_zero_counts[-2]
                    diffs.append(diff)

                ref_frames.pop(0)
                ref_frames.append(gray)
            else:
                break

        # print("倒着读完了")

    non_zero_counts = np.convolve(non_zero_counts, np.ones(5) / 5, mode='same')

    idx_sort = np.argsort(non_zero_counts)
    num_frames = len(non_zero_counts)
    fifth_small_idx = int(num_frames * 0.05)
    fifth_small_count = non_zero_counts[idx_sort[fifth_small_idx]]

    small_frames = []
    for i in range(num_frames):
        if non_zero_counts[i] <= fifth_small_count:
            small_frames.append(frame_nums[i])

    # self.queue.put(small_frames[len(small_frames) - 1])
    return small_frames[len(small_frames) - 1]


# def create_video_from_images(folder_path, output_path, fps):
#     folder_name = os.path.basename(folder_path)
#     output_file = os.path.join(output_path, folder_name + ".mp4")
#
#     # 如果文件已存在，则自动进行编号
#     counter = 1
#     while os.path.exists(output_file):
#         output_file = os.path.join(output_path, folder_name + "_" + str(counter) + ".mp4")
#         counter += 1
#
#     output_file = output_file.replace("\\", "/")
#     print("output_file =", output_file)
#
#
#     # 获取文件夹中的所有图片文件
#     image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg'))])
#
#     # 根据图片名称中的数字排序
#     image_files = sorted(image_files, key=lambda x: int(''.join(filter(str.isdigit, x))))
#     # print(image_files)
#
#     # 根据帧率创建一个空白视频剪辑
#     video_clip = ImageSequenceClip([os.path.join(folder_path, image_file) for image_file in image_files], fps=fps)
#
#     # 保存视频剪辑为文件
#     video_clip.write_videofile(output_file, codec='libx264')
#
#

#     video_name = os.path.basename(input_path)
#     print("video_name = " + video_name)
#     # output_path = '../datasets/enhanced_frames/' + video_name  作为主函数时路径用 ../
#     output_path = 'datasets/enhanced_frames/' + video_name
#     print("output_path = " + output_path)
#     us.up_sample(input_path, output_path, x_begin, x_end)
#     print("放大完成")
#
#     input_path = output_path
#     output_path = 'results'
#
#     print("当前路径" + os.getcwd())
#
#     # create_video_from_images(input_path, x_begin-1, x_end-1, output_path, fps=25)
#     create_video_from_images(input_path, output_path, fps=25)
#     # img_combine.img_combine(input_path, output_path)
#
#     end_time = time.time()
#     run_time = end_time - start_time
#     print("素材" + input_path + " 处理时间：", run_time)
#
#
# if __name__ == "__main__":
#     # input_path = 'G:/PYNM/FIN/results/mode2_stable.mp4'
#     input_path = 'G:/PYNM/FIN/datasets/tmp_image/mode2_stable'
#     # input_path = 'G:/PYNM/FIN/datasets/input_video/mode_stable.mp4'
#     duiqi(input_path)
