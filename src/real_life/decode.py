"""
Class defining the decoder of a frame
"""
import os
import glob
import torch
import time

from torch.nn import ReplicationPad2d, Module

from real_life.bitstream import ArithmeticCoder
from real_life.utils import BITSTREAM_SUFFIX, GOP_HEADER_SUFFIX, GOP_SUFFIX,\
                            VIDEO_HEADER_SUFFIX
from real_life.check_md5sum import compute_md5sum, read_md5sum
from real_life.header import read_gop_header, read_video_header
from func_util.nn_util import get_value, push_dic_to_device
from func_util.GOP_structure import FRAME_B, FRAME_P, FRAME_I, get_name_frame_code,\
                                    get_depth_gop
from func_util.img_processing import cast_before_png_saving, save_tensor_as_img,\
                                     get_y_u_v, save_yuv_separately
from func_util.console_display import print_log_msg

"""
In the entire file, the data dimension dictionnary contains the resolution (h x w)
of different things:

    'x'   : Resolution (h_x, w_x) of the Y channel of the input image
    'x_uv': Resolution (h_x_uv, w_x_uv) of the U & V channels of the input image
    'y'   : Resolution (h_y, w_y) of the latents Y for MOFNet and CodecNet
    'z'   : Resolution (h_z, w_z) of the latents Z for MOFNet and CodecNet

"""

def decode_one_video(param):

    DEFAULT_PARAM = {
        # Decoder with which we're going to decode the GOP
        'decoder': None,
        # Absolute path of the bitstream for this video
        'bitstream_path': '',
        # On which device the decoder runs
        'device': 'cpu',
        # Folder in which we output the decoded frames.
        # the filename will be derived from the bitstream_name
        'out_dir': '',
        # Set to true to generate more stuff, useful for debug
        'flag_bitstream_debug': False,        
    }

    decoder = get_value('decoder', param, DEFAULT_PARAM)
    bitstream_path = get_value('bitstream_path', param, DEFAULT_PARAM)
    device = get_value('device', param, DEFAULT_PARAM)
    out_dir = get_value('out_dir', param, DEFAULT_PARAM)
    flag_bitstream_debug = get_value('flag_bitstream_debug', param, DEFAULT_PARAM)

    decoder = decoder.eval()

    if not(out_dir.endswith('/')):
        out_dir += '/'

    # Clean the output directory
    os.system('mkdir -p ' + out_dir)
    os.system('rm -r ' + out_dir)
    os.system('mkdir -p ' + out_dir)

    # ! HARDCODED
    bitstream_dir = './tmp_out_bitstream/'

    print_log_msg('INFO', 'Bitstream path', '', bitstream_path)
    print_log_msg('INFO', 'Bitstream processing dir.', '', bitstream_dir)
    print_log_msg('INFO', 'Decoded frames directory', '', out_dir)

    uncat_one_video({'video_file': bitstream_path, 'out_dir': './tmp_out_bitstream/'})
    data_dim, nb_GOP, idx_starting_frame, idx_end_frame = read_video_header({
        'header_path': bitstream_dir + VIDEO_HEADER_SUFFIX
    })

    first_video_frame = idx_starting_frame # int(bitstream_dir.split('/')[-2].split('_')[-1])
    
    print_log_msg('INFO', 'Number of GOPs in bitstream', '', int(nb_GOP))
    print_log_msg('INFO', 'Index first frame', '', int(first_video_frame))
    print_log_msg('INFO', 'Index last frame', '', int(idx_end_frame))

    print_log_msg('INFO', 'Start decoding', '', '')

    start_time = time.time()
    first_GOP_frame_idx = first_video_frame
    for idx_GOP in range(nb_GOP):
        uncat_one_GOP({
            'gop_file': bitstream_dir + str(idx_GOP) + GOP_SUFFIX,
            'idx_first_gop_frame': first_GOP_frame_idx,
        })

        GOP_struct, idx_rate = read_gop_header({
            'header_path': bitstream_dir + str(idx_GOP) + GOP_HEADER_SUFFIX
        })

        list_bitstream_path = []

        # Create list of the bitstream in CODING ORDER
        # * + 1 because we want to compress the last idx_code 
        # GOP_2: index GOP is 0, 1, 2 and we want to code 0, 1 and 2
        for idx_code in range(get_depth_gop(GOP_struct) + 1):
            # get_name_frame_to_code[0]: there should be only one element in the list
            # no parallel coding of frame at the same temporal layer
            relative_idx_frame_to_code = int(get_name_frame_code(GOP_struct, idx_code)[0].split('_')[-1])
            list_bitstream_path.append(
                bitstream_dir + str(first_GOP_frame_idx + relative_idx_frame_to_code)
            )

        with torch.no_grad():
            decode_one_GOP({
                'decoder': decoder,
                'device': device,
                'out_dir': out_dir,
                'data_dim': data_dim,
                'GOP_struct': GOP_struct,
                'list_bitstream_path': list_bitstream_path,
                'flag_bitstream_debug': flag_bitstream_debug,
                'idx_rate': idx_rate,
            })

        # This is the future I-frame
        first_GOP_frame_idx += len(GOP_struct)

    # We're done for this sequence!
    elapsed_time = time.time() - start_time
    nb_frames = idx_end_frame - idx_starting_frame + 1
    print_log_msg('INFO', 'Decoding done', '', '')
    print_log_msg('RESULT', 'Number of frames', '[frame]', int(nb_frames))
    print_log_msg('RESULT', 'Decoding time', '[s]', '%.1f' % (elapsed_time))
    print_log_msg('RESULT', 'Decoding FPS', '[frame/s]', '%.1f' % (nb_frames / elapsed_time))

    # Remove the padded frames
    remove_padded_frames({'out_dir': out_dir, 'idx_end_frame': idx_end_frame})

    # Delete the temporary directory used to process the bitsream
    os.system('rm -r ' + bitstream_dir)


def remove_padded_frames(param):
    """
    During the encoding process, some supplementary frames are added to the
    video in order to process full GOPs.

    This function removes those supplementary frames, which have been decoded
    as PNGs.
    """

    DEFAULT_PARAM = {
        # Directory where the decoded PNGs are save
        'out_dir': None,
        # Index of the last coded frames (included). All frames whose index
        # is greater than idx_end_frame will be removed
        'idx_end_frame': None,
    }

    # =========================== RETRIEVE INPUTS =========================== #
    out_dir = get_value('out_dir', param, DEFAULT_PARAM)
    idx_end_frame = get_value('idx_end_frame', param, DEFAULT_PARAM)
    # =========================== RETRIEVE INPUTS =========================== #

    if not(out_dir.endswith('/')):
        out_dir += '/'

    list_decoded_png = glob.glob(out_dir + '*.png')

    # Remove all frames whose index > idx_end_frame
    for cur_file in list_decoded_png:
        idx = int(cur_file.split('/')[-1].split('_')[0])
        if idx > idx_end_frame:
            os.system('rm ' + cur_file)

    return


def decode_one_GOP(param):

    DEFAULT_PARAM = {
        # Decoder with which we're going to decode the GOP
        'decoder': None,
        # List of the bitstream files for all frames of this GOP
        # Given without suffix. 
        # ! Given in coding order, not temporal order!
        'list_bitstream_path': None,
        # Structure of the GOP,
        'GOP_struct': None,
        # If true, print a lot of message
        'verbose': True,
        # Data dimension (resolution) (as dictionnary of tuples), one for 'x', one for 'y'
        # and one for 'z'
        'data_dim': None,
        # On which device the decoder runs
        'device': 'cpu',
        # Folder in which we output the decoded frames.
        # the filename will be derived from the bitstream_name
        'out_dir': '',
        # Set to true to generate more stuff, useful for debug
        'flag_bitstream_debug': False,   
        # For multi rate
        'idx_rate': 0.,     
    }
    
    decoder = get_value('decoder', param, DEFAULT_PARAM)
    list_bitstream_path = get_value('list_bitstream_path', param, DEFAULT_PARAM)
    GOP_struct = get_value('GOP_struct', param, DEFAULT_PARAM)
    verbose = get_value('verbose', param, DEFAULT_PARAM)
    data_dim = get_value('data_dim', param, DEFAULT_PARAM)
    device = get_value('device', param, DEFAULT_PARAM)
    out_dir = get_value('out_dir', param, DEFAULT_PARAM)
    flag_bitstream_debug = get_value('flag_bitstream_debug', param, DEFAULT_PARAM)
    idx_rate = get_value('idx_rate', param, DEFAULT_PARAM)

    if verbose and not(len(list_bitstream_path) == len(GOP_struct)):
        print('[ERROR] decode_one_GOP')
        print('[ERROR] Number of frames in GOP_struct is different from the number of bitstream_path')
        print('[ERROR] len(list_bitstream_path): ' + str(len(list_bitstream_path)))
        print('[ERROR] len(GOP_struct)         : ' + str(len(GOP_struct)))
        print('[ERROR] Exiting')
        return
    
    nb_frames = len(GOP_struct)

    # Used to store the decoded frames as YUV 420 dictionnaries
    decoded_GOP = {}

    h_x, w_x = data_dim.get('x')
    h_x_uv, w_x_uv = data_dim.get('x_uv')

    for idx_code, cur_bitstream in enumerate(list_bitstream_path):
        
        # ! Same as above, we have only one frame per idx_code because
        # ! there is no parallel coding per temporal layer. Thus each idx_code
        # ! correspond to exactly one frame, resulting in a list with a single element
        # 'frame_0' or 'frame_1' etc.
        frame_name_inside_GOP = get_name_frame_code(GOP_struct, idx_code)[0]
        cur_frame_type = GOP_struct.get(frame_name_inside_GOP).get('type')

        # Retrieve reference frames if any
        if cur_frame_type == FRAME_B:
            next_ref = decoded_GOP.get(GOP_struct.get(frame_name_inside_GOP).get('next_ref'))
        else:
            next_ref = {
                'y': torch.zeros((1, 1, h_x, w_x), device=device),
                'u': torch.zeros((1, 1, h_x_uv, w_x_uv), device=device),
                'v': torch.zeros((1, 1, h_x_uv, w_x_uv), device=device),
            }

        if (cur_frame_type == FRAME_P) or (cur_frame_type == FRAME_B):
            prev_ref = decoded_GOP.get(GOP_struct.get(frame_name_inside_GOP).get('prev_ref'))
        else:
            prev_ref = {
                'y': torch.zeros((1, 1, h_x, w_x), device=device),
                'u': torch.zeros((1, 1, h_x_uv, w_x_uv), device=device),
                'v': torch.zeros((1, 1, h_x_uv, w_x_uv), device=device),
            } 


        decoded_frame = decoder.decode({
            'prev_dic': prev_ref,
            'next_dic': next_ref,
            'frame_type': cur_frame_type,
            'bitstream_path': cur_bitstream,
            'data_dim': data_dim,
            'flag_bitstream_debug': flag_bitstream_debug,
            'idx_rate': idx_rate,
            'device': device,
        })
        decoded_frame = cast_before_png_saving({'x': decoded_frame, 'data_type': 'yuv_dic'})

        decoded_GOP[frame_name_inside_GOP] = decoded_frame

        # Save frame as png: frame_name = padded index
        frame_name = cur_bitstream.split('/')[-1]
        # frame_name: sequence_name + idx + '.png'        
        # '_'.join(cur_bitstream.split('/')[-2].split('_')[:-1]) + '_' + cur_bitstream.split('/')[-1]

        if flag_bitstream_debug:
            save_tensor_as_img(decoded_frame, out_dir + frame_name + '.png', mode='yuv420')

        # print('Decoder decoded frame y: ' + str(decoded_frame.get('y').abs().sum()))
        # print('Saving decoded frame: ' + str(out_dir + frame_name))
        save_yuv_separately(decoded_frame, out_dir + frame_name)

        # Compare the md5sum of these PNGs with the ones from the encoder
        if flag_bitstream_debug:
            # RealLife/debug/SequenceName/
            root_debug_path = '/'.join(cur_bitstream.split('/')[:-3]) + '/debug/' + cur_bitstream.split('/')[-2] + '/'
            # Index of the frame (in the video)
            idx_frame = cur_bitstream.split('/')[-1]

            for c in ['y', 'u', 'v']:
                encoder_md5 = read_md5sum({
                    'in_file': root_debug_path + str(idx_frame) + '_' + c + '.md5'
                })
                decoder_md5 = compute_md5sum({
                    'in_file': out_dir + frame_name + '_' + c + '.png'
                })

                msg = frame_name + '_' + c + ': '
                if encoder_md5 != decoder_md5:
                    msg += '\n' + '-' * 80 + '\n'
                    msg += 'Incorrect reconstruction!\n'
                    msg += '-' * 80 + '\n'
                else:
                    msg += 'Identical reconstruction!'
                print(msg)
            print('')


def uncat_one_GOP(param):
    
    DEFAULT_PARAM = {
        # Absolute path of the file containing the GOP bitstream. 
        # This file will be deleted at the end and replaced by one GOP header
        # and all the separated frame bitstreams.
        'gop_file': '',
        # Index of the first frame of the gop in the video
        'idx_first_gop_frame': 0,
    }

    gop_file = get_value('gop_file', param, DEFAULT_PARAM)
    idx_first_gop_frame = get_value('idx_first_gop_frame', param, DEFAULT_PARAM)

    # Get the bytes
    with open(gop_file, 'rb') as fin:
        byte_stream = fin.read()

    # Extract the header
    # ! This is hardcoded
    GOP_HEADER_SIZE_BYTES = 2
    byte_header = byte_stream[:GOP_HEADER_SIZE_BYTES]
    header_file = gop_file.rstrip(GOP_SUFFIX) + GOP_HEADER_SUFFIX
    with open(header_file, 'wb') as fout:
        fout.write(byte_header)
    byte_stream = byte_stream[GOP_HEADER_SIZE_BYTES:]

    # Get the GOP structure to check how many frames we have
    GOP_struct, _ = read_gop_header({'header_path': header_file})

    # Extract each frame
    for i in range(len(GOP_struct)):
        frame_idx = i + idx_first_gop_frame
        out_file = '/'.join(gop_file.split('/')[:-1]) + '/' + str(frame_idx) + BITSTREAM_SUFFIX

        # Get the number of bytes for this frame
        nb_bytes_cur_frame = int.from_bytes(byte_stream[0:4], byteorder='big')
        byte_stream = byte_stream[4:]
        # write them
        with open(out_file, 'wb') as fout:
            fout.write(byte_stream[:nb_bytes_cur_frame])
        # skip them
        byte_stream = byte_stream[nb_bytes_cur_frame:]

    # Remove the gop file
    os.system('rm ' + gop_file)


def uncat_one_video(param):

    DEFAULT_PARAM = {
        # Absolute path of the file containing the video bitstream. 
        # This file will be deleted at the end and replaced by one video header
        # and all the separated gop bitstream
        'video_file': '',
        # Directory in which the video_file will be uncat
        'out_dir': '',
    }

    video_file = get_value('video_file', param, DEFAULT_PARAM)
    out_dir = get_value('out_dir', param, DEFAULT_PARAM)

    # Move the video file to a temporary .bin extension
    # os.system('mv ' + video_file + ' ' + video_file + '.bin')
    # Instanciate a directory to extract all the different bitstreams
    bitstream_dir = out_dir
    os.system('mkdir -p ' + bitstream_dir)

    # Get the bytes
    with open(video_file, 'rb') as fin:
        byte_stream = fin.read()

    # Extract the header
    # ! This is hardcoded
    VIDEO_HEADER_SIZE_BYTES = 18
    byte_header = byte_stream[:VIDEO_HEADER_SIZE_BYTES]
    header_file = bitstream_dir + VIDEO_HEADER_SUFFIX
    with open(header_file, 'wb') as fout:
        fout.write(byte_header)
    byte_stream = byte_stream[VIDEO_HEADER_SIZE_BYTES:]

    # Get the number of GOPs from the video header
    _, nb_gop, _, _ = read_video_header({'header_path': header_file})

    for i in range(nb_gop):
        out_file = bitstream_dir + str(i) + GOP_SUFFIX

        # Get the number of bytes for this frame
        nb_bytes_cur_gop= int.from_bytes(byte_stream[0:4], byteorder='big')
        byte_stream = byte_stream[4:]
        # write them
        with open(out_file, 'wb') as fout:
            fout.write(byte_stream[:nb_bytes_cur_gop])
        # skip them
        byte_stream = byte_stream[nb_bytes_cur_gop:]
    
    # # Remove the video file
    # os.system('rm ' + video_file + '.bin')


class Decoder(Module):
    """
    Entire decoder of the system. Built from a complete system <FullNet>
    """

    def __init__(self, param):
        super(Decoder, self).__init__()

        DEFAULT_PARAM = {
            # Complete system (encoder + decoder) from which we're going
            # to copy the decoder part
            'full_net': None,
        }

        full_net = get_value('full_net', param, DEFAULT_PARAM)

        # Retrieve the decoder element from full_net and construct
        # the MOFNet and CodecNet decoder
        self.codec_net_dec = CodecNetDecoder({'codec_net': full_net.codec_net})
        self.mofnet_dec = MOFNetDecoder({'mofnet': full_net.mode_net})

        # Hard-wired operations
        self.motion_compensation = full_net.motion_compensation
        self.in_layer = full_net.in_layer
        self.out_layer = full_net.out_layer

    def decode(self, param):
        DEFAULT_PARAM = {
            # YUV dictionnary of the previous frame
            'prev_dic': None,
            # YUV dictionnary of the next frame
            'next_dic': None,
            # A scalar, indicating the type of the frame for
            # each of the B examples which are either: FRAME_I, FRAME_P or FRAME_B
            # In most case it should have B = 1.
            'frame_type': None,
            # Absolute path of the bitstream
            'bitstream_path': None,
            # Data dimension (as dictionnary of tuples), one for 'x', one for 'y'
            # and one for 'z'
            'data_dim': None,
            # Set to true to generate more stuff, useful for debug
            'flag_bitstream_debug': False,
            # For multi rate
            'idx_rate': 0.,
            # On which device the decoder runs
            'device': 'cpu',
        }

        # Parse inputs
        p = get_value('prev_dic', param, DEFAULT_PARAM)
        n = get_value('next_dic', param, DEFAULT_PARAM)
        frame_type = get_value('frame_type', param, DEFAULT_PARAM)
        bitstream_path = get_value('bitstream_path', param, DEFAULT_PARAM)
        data_dim = get_value('data_dim', param, DEFAULT_PARAM)
        flag_bitstream_debug = get_value('flag_bitstream_debug', param, DEFAULT_PARAM)
        idx_rate = get_value('idx_rate', param, DEFAULT_PARAM)
        device = get_value('device', param, DEFAULT_PARAM)

        # Push references to device
        p = push_dic_to_device(p, device)
        n = push_dic_to_device(n, device)

        # ===== PRE-PROCESSING ===== #
        prev_ref = self.in_layer(p)
        next_ref = self.in_layer(n)
        # ===== PRE-PROCESSING ===== #

        h_x, w_x = data_dim.get('x')

        # I-frames, no MOFNet and no prediction
        if frame_type == FRAME_I:
            # Dummy alpha, dummy prediction and skip part
            alpha = torch.ones((1, 3, h_x, w_x), device=device)
            warped_ref = torch.zeros((1, 3, h_x, w_x), device=device)
            skip_part = torch.zeros((1, 3, h_x, w_x), device=device)
        else:
            # Decode MOFNet output
            tmp_out = self.mofnet_dec.decode({
                'bitstream_path': bitstream_path,
                'frame_type': frame_type,
                'prev': prev_ref,
                'next': next_ref,
                'data_dim': data_dim,
                'flag_bitstream_debug': flag_bitstream_debug,
                'idx_rate': idx_rate,
                'device': device,
            })

            alpha = tmp_out.get('alpha')
            beta = tmp_out.get('beta')
            v_prev = tmp_out.get('v_prev')
            v_next = tmp_out.get('v_next')

            # Perform motion compensation
            tmp_out = self.motion_compensation({
                'prev': prev_ref,
                'next': next_ref,
                'v_prev': v_prev,
                'v_next': v_next,
                'beta': beta,
                'interpol_mode': 'bilinear',
            })

            warped_ref = tmp_out.get('x_warp')

            # Skip part
            skip_part = (1 - alpha) * warped_ref

        # CodecNet
        codec_net_out = self.codec_net_dec.decode({
            'bitstream_path': bitstream_path,
            'frame_type': frame_type,
            'prediction': warped_ref * alpha,
            'data_dim': data_dim,
            'flag_bitstream_debug': flag_bitstream_debug,
            'idx_rate': idx_rate,
                'device': device,
        })

        output_frame = codec_net_out + skip_part

        # So far, output frame is 444. Transform it into a YUV 420 dictionnary
        # ===== DOWNSCALING AND 420 STUFF ===== #
        output_frame = self.out_layer(output_frame)

        # Downscaled version of u and v can be smaller than their true size by one pixel.
        # Difference in size should be of 0 or 1 pixel
        output_frame_y, output_frame_u, output_frame_v = get_y_u_v(output_frame)

        # Refers to the U and V channels of the input image
        h_x_uv, w_x_uv = data_dim.get('x_uv')

        nb_pad_row = abs(h_x_uv - output_frame_u.size()[2])
        nb_pad_col = abs(w_x_uv - output_frame_u.size()[3])
        my_padding = ReplicationPad2d((0, nb_pad_col, 0, nb_pad_row))

        # Remove supplementary pixels if needed
        output_frame = {
            'y': output_frame_y[:, :, :h_x, :w_x],
            'u': my_padding(output_frame_u)[:, :, :h_x_uv, :w_x_uv],
            'v': my_padding(output_frame_v)[:, :, :h_x_uv, :w_x_uv],
        }
        # ===== DOWNSCALING AND 420 STUFF ===== #

        # ===== QUANTIZE THE OUTPUT TENSOR ===== #
        output_frame = cast_before_png_saving(
            {'x': output_frame, 'data_type': 'yuv_dic'}
        )
        # ===== QUANTIZE THE OUTPUT TENSOR ===== #

        return output_frame


class CodecNetDecoder(Module):
    """
    Decoder of the MOFNet, build from a complete MOFNet.
    """

    def __init__(self, param):
        super(CodecNetDecoder, self).__init__()

        DEFAULT_PARAM = {
            # Complete codecnet (encoder + decoder) from which we're going
            # to copy the decoder part
            'codec_net': None,
        }

        codec_net = get_value('codec_net', param, DEFAULT_PARAM)

        # Construct the conditional decoder
        self.codec_dec = ConditionalDecoder({'conditional_net': codec_net.codec_net})

    def decode(self, param):
        DEFAULT_PARAM = {
            # Absolute path of the bitstream
            'bitstream_path': None,
            # A scalar, indicating the type of the frame for
            # each of the B examples which are either: FRAME_I, FRAME_P or FRAME_B
            # In most case it should have B = 1.
            'frame_type': None,
            # 4-D predicted frame \tilde{x}_t * alpha
            'prediction': None,
            # Data dimension (as dictionnary of tuples), one for 'x', one for 'y'
            # and one for 'z'
            'data_dim': None,
            # Set to true to generate more stuff, useful for debug
            'flag_bitstream_debug': False,
            # For multi rate
            'idx_rate': 0.,
            # On which device the decoder runs
            'device': 'cpu',
        }

        bitstream_path = get_value('bitstream_path', param, DEFAULT_PARAM)
        frame_type = get_value('frame_type', param, DEFAULT_PARAM)
        prediction = get_value('prediction', param, DEFAULT_PARAM)
        data_dim = get_value('data_dim', param, DEFAULT_PARAM)
        flag_bitstream_debug = get_value('flag_bitstream_debug', param, DEFAULT_PARAM)
        idx_rate = get_value('idx_rate', param, DEFAULT_PARAM)
        device = get_value('device', param, DEFAULT_PARAM)        

        # Check if we need the shortcut and compute its input
        flag_shortcut = frame_type != FRAME_I
        if flag_shortcut:
            in_shortcut = prediction
        else:
            in_shortcut = None

        # Perform MOFNet decoding
        codecnet_out = self.codec_dec.decode({
            'bitstream_path': bitstream_path,
            'frame_type': frame_type,
            'flag_shortcut': flag_shortcut,
            'in_shortcut': in_shortcut,
            'data_dim': data_dim,
            'flag_bitstream_debug': flag_bitstream_debug,
            'latent_name': 'codecnet',
            'idx_rate': idx_rate,
            'device': device,
        })

        h_x, w_x = data_dim.get('x')
        x_hat = codecnet_out[:, :, :h_x, :w_x]

        return x_hat


class MOFNetDecoder(Module):

    """
    Decoder of the MOFNet, build from a complete MOFNet.
    """

    def __init__(self, param):
        super(MOFNetDecoder, self).__init__()

        DEFAULT_PARAM = {
            # Complete mofnet (encoder + decoder) from which we're going
            # to copy the decoder part
            'mofnet': None,
        }

        mofnet = get_value('mofnet', param, DEFAULT_PARAM)

        # Construct the conditional decoder
        self.mofnet_dec = ConditionalDecoder({'conditional_net': mofnet.mode_net})

    def decode(self, param):

        DEFAULT_PARAM = {
            # Absolute path of the bitstream
            'bitstream_path': None,
            # A scalar, indicating the type of the frame for
            # each of the B examples which are either: FRAME_I, FRAME_P or FRAME_B
            'frame_type': None,
            # 4-D previous reference frame x_{t-1}
            'prev': None,
            # 4-D next reference x_{t+1}
            'next': None,
            # Data dimension (as dictionnary of tuples), one for 'x', one for 'y'
            # and one for 'z'
            'data_dim': None,
            # Set to true to generate more stuff, useful for debug
            'flag_bitstream_debug': False,
            # For multi rate
            'idx_rate': 0.,
            # On which device the decoder runs
            'device': 'cpu',
        }

        bitstream_path = get_value('bitstream_path', param, DEFAULT_PARAM)
        frame_type = get_value('frame_type', param, DEFAULT_PARAM)
        prev_ref = get_value('prev', param, DEFAULT_PARAM)
        next_ref = get_value('next', param, DEFAULT_PARAM)
        data_dim = get_value('data_dim', param, DEFAULT_PARAM)
        flag_bitstream_debug = get_value('flag_bitstream_debug', param, DEFAULT_PARAM)
        idx_rate = get_value('idx_rate', param, DEFAULT_PARAM)
        device = get_value('device', param, DEFAULT_PARAM)        

        # Check if we need the shortcut and compute its input
        flag_shortcut = frame_type == FRAME_B
        if flag_shortcut:
            in_shortcut = torch.cat((prev_ref, next_ref), dim=1)
        else:
            in_shortcut = None

        # Perform MOFNet decoding
        mofnet_out = self.mofnet_dec.decode({
            'bitstream_path': bitstream_path,
            'frame_type': frame_type,
            'flag_shortcut': flag_shortcut,
            'in_shortcut': in_shortcut,
            'data_dim': data_dim,
            'flag_bitstream_debug': flag_bitstream_debug,
            'latent_name': 'mofnet',
            'idx_rate': idx_rate,
            'device': device,            
        })

        # Get alpha, beta, v_prev and v_next from MOFNet decodin
        h_x, w_x = data_dim.get('x')
        mofnet_out = mofnet_out[:, :, :h_x, :w_x]
        alpha = torch.clamp(mofnet_out[:, 0, :, :].view(1, 1, h_x, w_x) + 0.5, 0., 1.).repeat(1, 3, 1, 1)
        beta = torch.clamp(mofnet_out[:, 1, :, :].view(1, 1, h_x, w_x) + 0.5, 0., 1.).repeat(1, 3, 1, 1)
        v_prev = mofnet_out[:, 2:4, :, :]
        v_next = mofnet_out[:, 4:6, :, :]

        if frame_type == FRAME_P:
            beta[:, :, :, :] = 1.
            v_next[:, :, :, :] = 0


        net_out = {
            'alpha': alpha,
            'beta': beta,
            'v_prev': v_prev,
            'v_next': v_next,
        }

        return net_out


class ConditionalDecoder(Module):

    """
    Decoder part of a ConditionalNet, built from a complete
    ConditionalNet.
    """

    def __init__(self, param):
        super(ConditionalDecoder, self).__init__()

        DEFAULT_PARAM = {
            # Complete conditional coder (encoder + decoder) from which we're going
            # to copy the decoder part
            'conditional_net': None,
        }

        conditional_net = get_value('conditional_net', param, DEFAULT_PARAM)

        self.g_s = conditional_net.g_s
        self.h_s = conditional_net.h_s
        self.g_a_ref = conditional_net.g_a_ref
        self.pdf_y = conditional_net.pdf_y
        self.pdf_z = conditional_net.pdf_z
        self.pdf_parameterizer = conditional_net.pdf_parameterizer
        # Needed to generate dummy data when the shortcut is not available
        self.nb_ft_shortcut_out = conditional_net.out_c_shortcut_y
        # Number of bottleneck features
        self.nb_ft_y = conditional_net.nb_ft_y
        self.nb_ft_z = conditional_net.nb_ft_z

        self.gain_I = conditional_net.gain_I
        self.flag_gain_p_b = conditional_net.flag_gain_p_b

        if self.flag_gain_p_b:
            self.gain_P = conditional_net.gain_P
            self.gain_B = conditional_net.gain_B
        
        self.ac = conditional_net.ac


    def decode(self, param):
        DEFAULT_PARAM = {
            # Absolute path of the bitstream
            'bitstream_path': None,
            # Type of the frame, either FRAME_I, FRAME_P or FRAME_B
            'frame_type': None,
            # If True, we use the shortcut transform
            'flag_shortcut': False,
            # Input (4-D) of the shortcut transform
            'in_shortcut': None,
            # Data dimension (as dictionnary of tuples), one for 'x', one for 'y'
            # and one for 'z'
            'data_dim': None,
            # For multi rate
            'idx_rate': 0.,
            # Set to true to generate more stuff, useful for debug
            'flag_bitstream_debug': False,        
            # Specify what quantity we're entropy coding.
            # It can be <mofnet_y>, <mofnet_z>, <codecnet_y>, codecnet_z>.
            # This is needed for the decoding part, to know which part of the
            # bitstream we should read.
            # Bitstream structure for each frame is detailed above.
            'latent_name': '',
            # On which device it will run
            'device': '',
        }

        bitstream_path = get_value('bitstream_path', param, DEFAULT_PARAM)
        frame_type = get_value('frame_type', param, DEFAULT_PARAM)
        flag_shortcut = get_value('flag_shortcut', param, DEFAULT_PARAM)
        in_shortcut = get_value('in_shortcut', param, DEFAULT_PARAM)
        data_dim = get_value('data_dim', param, DEFAULT_PARAM)     
        idx_rate = get_value('idx_rate', param, DEFAULT_PARAM)     
        flag_bitstream_debug = get_value('flag_bitstream_debug', param, DEFAULT_PARAM)
        latent_name = get_value('latent_name', param, DEFAULT_PARAM)
        device = get_value('device', param, DEFAULT_PARAM)

        if flag_bitstream_debug:
            print('Decoder idx_rate: ' + str(idx_rate))


        # Read dimension of the input, of y and of z
        h_y, w_y = data_dim.get('y')
        h_z, w_z = data_dim.get('z')

        # Decode the z
        z_hat = self.ac.decode({
            'mode': 'pmf',
            'bitstream_path': bitstream_path,
            'data_dim': (1, self.nb_ft_z, h_z, w_z),
            'device': device,
            'latent_name': latent_name + '_z',
        })

        # Generate mu and sigma from z_hat
        pdf_param_y = self.pdf_parameterizer(self.h_s(z_hat)[:, :, :h_y, :w_y])
        mu = pdf_param_y[0].get('mu')
        sigma = pdf_param_y[0].get('sigma')

        # Arithmetic decoding of the y
        y_centered = self.ac.decode({
            'mode': 'laplace',
            'bitstream_path': bitstream_path,
            'data_dim': (1, self.nb_ft_y, h_y, w_y),
            'device': device,
            'latent_name': latent_name + '_y',  
            'sigma': sigma,          
        })

        y_hat = y_centered + mu

        # Input of the gain matrix
        gain_matrix_in = {
            'x': y_hat, 'idx_rate': idx_rate, 'mode': 'dec'
        }

        # Get the correct gain matrix
        if not(self.flag_gain_p_b):
            tmp_out = self.gain_I(gain_matrix_in)
        else:
            if frame_type == FRAME_I:
                tmp_out = self.gain_I(gain_matrix_in)
            elif frame_type == FRAME_P:
                tmp_out = self.gain_P(gain_matrix_in)
            elif frame_type == FRAME_B:
                tmp_out = self.gain_B(gain_matrix_in)
        
        y_hat = tmp_out.get('output')

        # Shortcut if required
        if flag_shortcut:
            y_shortcut = self.g_a_ref(in_shortcut)
        else:
            # Dummy input if shortcut is not available
            y_shortcut = torch.zeros((1, self.nb_ft_shortcut_out, h_y, w_y), device=device)

        # Decode both latents together
        y_to_decode = torch.cat((y_hat, y_shortcut), dim=1)
        x_hat = self.g_s(y_to_decode)

        return x_hat

