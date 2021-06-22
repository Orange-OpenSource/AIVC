import argparse
import glob
from clic21.metrics import evaluate

"""
Use the CLIC21 metrics to measure the quality on ALL the images contained in
the <compressed> arguments.
"""


# ============================ Arguments parsing ============================ #
parser = argparse.ArgumentParser()

parser.add_argument(
    '--raw', default='', type=str, help='Path with the raw PNGs for the input data.'
)

parser.add_argument(
    '--compressed', default='', type=str, help='Path with the compressed PNGs'
)
args = parser.parse_args()

path_raw = args.raw
if not(path_raw.endswith('/')):
    path_raw += '/'

path_compressed = args.compressed
if not(path_compressed.endswith('/')):
    path_compressed += '/'
# ============================ Arguments parsing ============================ #

# Get the indices of all images in path_compress
compressed_images_idx = [int(x.split('/')[-1].rstrip('_y.png')) for x in glob.glob(path_compressed + '*_y.png')]

target = {}
submit = {}
for idx in compressed_images_idx:
    for c in ['y', 'u', 'v']:
        cur_key = str(idx) + '_' + c
        submit[cur_key] = path_raw + str(idx) + '_' + c + '.png'
        target[cur_key] = path_compressed + str(idx) + '_' + c + '.png'


results = evaluate(submit, target)

print('PSNR    [dB]: ' + '%.5f' % (results.get('PSNR')))
print('MS-SSIM     : ' + '%.5f' % (results.get('MSSSIM')))
print('MS-SSIM [dB]: ' + '%.5f' % (results.get('MSSSIM_dB')))
# ========================= Evaluate the encoding =========================== #
