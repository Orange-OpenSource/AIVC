import subprocess
import math
import os
from os import listdir
from os.path import isfile, join
from PIL import Image


def get_png_resolution(png_path):
    """
    Return the resolution of a PNG image

    Args:
        png_path ([str]): [Absolute path of the png]

    Returns:
        [tuple]: [(width, height) of the png]
    """

    im = Image.open(png_path)
    width, height = im.size
    return (width, height)


def get_video_resolution(video_path):
    """
    Return the resolution of a video

    Args:
        video_path ([str]): [Absolute path of the video]

    Returns:
        [tuple]: [(width, height) of the video]
    """
    # Handle raw yuv file
    if video_path.endswith('.yuv'):
        output = video_path.split('/')[-1].split('_')[1].split('x')
        width = float(output[0])
        height = float(output[1])
        return (width, height)

    cmd = 'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 ' + video_path
    output = subprocess.getoutput(cmd).split(',')
    width = float(output[0])
    height = float(output[1])
    return (width, height)


def get_video_fps(video_path):
    """
    Return the framerate of a video

    Args:
        video_path ([str]): [Absolute path of the video]

    Returns:

        [int]: [framerate of the video]
    """
    # Handle raw yuv file
    if video_path.endswith('.yuv'):
        output = video_path.split('/')[-1].split('_')[2]
        return float(output)

    cmd = 'ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate ' + video_path
    output = subprocess.getoutput(cmd)
    # Output is something as x/y, so we need to compute it
    output = output.split('/')
    fps = round(float(output[0]) / float(output[1]))
    return fps


def get_video_duration(video_path):
    """
    Return the duration of a video in seconds

    Args:
        video_path ([str]): [Absolute path of the video]

    Returns:

        [float]: [duration of the video in seconds]
    """
    # Handle raw yuv file
    if video_path.endswith('.yuv'):
        video_size_byte = get_size_file(video_path)
        w, h = get_video_resolution(video_path)
        # ! All the sequences have a 8 bit depth
        bit_depth = 8
        # bit_depth = int(video_path.split('/')[-1].split('_')[4].rstrip('b.yuv'))
        byte_per_frame = 1.5 * h * w * bit_depth / 8
        fps = get_video_fps(video_path)
        nb_frames = video_size_byte / byte_per_frame
        duration = nb_frames / fps
        # print('Res: ' + str(w) + 'x' + str(h) + ' Nb frames: ' + str(nb_frames))
        return duration

    cmd = 'ffprobe -v error -select_streams v:0 -show_entries stream=duration -of default=noprint_wrappers=1:nokey=1 ' + video_path
    output = float(subprocess.getoutput(cmd))
    return output


def get_files_in_dir(dir_path):
    """
    Return a list of all the files in a given directory

    Args:
        dir_path ([str]): [Absolute path of the directory]

    Returns:
        [list of str]: [List of all the files in dir_path]
    """

    return [f for f in listdir(dir_path) if isfile(join(dir_path, f))]


def get_subdir_in_dir(dir_path):
    """
    Return a list of all the subdirectories in a given directory

    Args:
        dir_path ([str]): [Absolute path of the directory]

    Returns:
        [list of str]: [List of all the subdirectories in dir_path]
    """

    return [os.path.join(dir_path, o) for o in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path,o))]


def get_size_file(file_path):
    """
    Return size of file in bytes

    Args:
        file_path ([str]): [Path of the file]

    Returns:
        [int]: [Size of the file in bytes]
    """
    return os.path.getsize(file_path)
