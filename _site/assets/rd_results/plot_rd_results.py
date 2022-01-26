"""
Better plot of the rate distortion results.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from numpy.core.defchararray import startswith

color_per_codec = {'AIVC': '#2A9D8F', 'x264': '#E9C46A', 'x265': '#E76F51'}
linestyle_per_config = {'RA': 'solid', 'LDP': 'dotted', 'AI': 'dashed'}


def parse_one_res_file(res_path):
    """Parse a .txt result file and return a dic storing the results

    Args:
        res_path (str): path of the res file

    Returns:
        dict: Dictionnary gathering the results. structure is:
        res[seq_name][config] = {
            'psnr': [], 'ms_ssim_db': [], 'size_byte': [], 'rate_mbits': []
        }
    """
    res = {}

    # Read the file
    with open(res_path, 'r') as f_in:
        all_lines = [line.rstrip('\n').rstrip(',') for line in f_in.readlines()]
    # Parse it
    for line in all_lines:
        if line.startswith('PSNR [dB]') or line == '':
            continue

        elif line.startswith('Sequence'):
            seq_name = line.split(': ')[-1]
            res[seq_name] = {}

        elif line.startswith('config: '):
            config = line.split(': ')[-1]
            res[seq_name][config] = {}
            list_key = ['psnr', 'ms_ssim_db', 'size_byte', 'rate_mbits']
            for k in list_key:
                res[seq_name][config][k] = []

        # Add results
        else:
            line = [float(x) for x in line.split(',')]
            for idx, k in enumerate(list_key):
                res[seq_name][config][k].append(line[idx])

    return res


def initialize_fig(fig_title, xlim=(0., 10.), ylim=(10., 30.)):
    fig = plt.figure()

    video_size = fig_title.split('_')[1]
    if video_size == '1920x1080':
        xlim = (0., 10.)
        ylim = (8., 24)
    elif video_size == '832x480':
        xlim = (0., 6.)
        ylim = (8., 24)
    elif video_size == '416x240':
        xlim = (0., 1.6)
        ylim = (8., 24.)
    elif video_size == '1280x720':
        xlim = (0., 6.)
        ylim = (12., 24.)

    plt.title(fig_title)
    plt.xlabel('Rate [Mbit/s]')
    plt.ylabel('Quality MS-SSIM [dB]')
    plt.xlim(xlim)
    plt.ylim(ylim)



    ax = plt.gca()

    nb_major_x_ticks = 4
    nb_minor_x_ticks = 2
    nb_major_y_ticks = 4
    nb_minor_y_ticks = 2

    x_major_multiple = (xlim[1] - xlim[0]) / nb_major_x_ticks
    x_minor_multiple = x_major_multiple / nb_minor_x_ticks

    y_major_multiple = (ylim[1] - ylim[0]) / nb_major_y_ticks
    y_minor_multiple = y_major_multiple / nb_minor_y_ticks


    ax.xaxis.set_major_locator(MultipleLocator(x_major_multiple))
    ax.xaxis.set_minor_locator(MultipleLocator(x_minor_multiple))
    ax.yaxis.set_major_locator(MultipleLocator(y_major_multiple))
    ax.yaxis.set_minor_locator(MultipleLocator(y_minor_multiple))
    plt.grid(b=True, which='major', color='darkgrey')
    plt.grid(b=True, which='minor', color='gainsboro')

    print(f'Figure title: {fig_title}')


def print_rd(rate, quality, name):
    # print(f'Name: {name}')
    # print(f'Rate: {rate}')
    # print(f'Quality: {quality}')

    if name.split('_')[0].split('-')[1] == 'AI':
        return

    plt.plot(
        rate,
        quality,
        label=name.split('_')[0],
        color=color_per_codec.get(name.split('-')[0]),
        linewidth=1.85,
        marker='o' if name.split('-')[0] == 'AIVC' else ',',
        markersize=7,
        linestyle=linestyle_per_config.get(name.split('_')[0].split('-')[1])
    )


def show_figure():
    plt.legend(handlelength=6)
    plt.show()


def save_figure(path):
    plt.legend(handlelength=6)
    plt.savefig(path, dpi=300)


# Parse result files
home_path = './'
res_nn = parse_one_res_file(f'{home_path}aivc_hevc_ctc.txt')
res_x264 = parse_one_res_file(f'{home_path}x264_hevc_ctc.txt')
res_x265 = parse_one_res_file(f'{home_path}x265_hevc_ctc.txt')

overall_res = {'AIVC': res_nn, 'x265': res_x265, 'x264': res_x264}

for seq_name in res_nn:
    initialize_fig(seq_name)
    for config in res_nn.get(seq_name):
        for codec_name, codec_res in overall_res.items():
            # Overide the LDP config name for cause of different intra period
            if codec_name in ['x264', 'x265']:
                if config == 'LDP_IP8_GOP32':
                    config = 'LDP_IP32_GOP32'
            try:
                rate = codec_res.get(seq_name).get(config).get('rate_mbits')
                ms_ssim_db = codec_res.get(seq_name).get(config).get('ms_ssim_db')
            except AttributeError:
                print(f'Missing data for {seq_name}, {config}, {codec_name}')

            print_rd(rate, ms_ssim_db, codec_name + '-' + config)
    save_figure(f'{home_path}{seq_name}.png')