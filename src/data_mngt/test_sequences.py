def get_clic_sequences(mode='valid'):
    """
    From a .txt file describing the frames to code, return an appropriate
    list of dictionnaries describing the sequence to code.

    <mode> is either "valid" or "minivalid".
    """

    if mode == 'valid':
        DESCRIPTION_FILE = '/opt/GPU/Dataset/VideoCoding/CLIC_21_Video/TestSets/ChallengeValidation/video_targets_valid.txt'
    elif mode == 'minivalid':
        DESCRIPTION_FILE = '/opt/GPU/Dataset/VideoCoding/CLIC_21_Video/TestSets/MiniValidation/video_targets_valid.txt'

    f = open(DESCRIPTION_FILE, 'r')

    list_sequence = []

    seq_name = ''
    idx_starting_frame = -1
    cur_idx = -1
    idx_end_frame = -1

    ROOT_PATH_SEQ = '/opt/GPU/Dataset/VideoCoding/CLIC_21_Video/RawData/'
    
    for line in f.readlines():
        line = line.rstrip('\n')

        # We've reached the end of one sequence description. 
        # Keep the last current index <cur_idx> as the idx_end_frame
        # Log this sequence inside <list_sequence> and proceed to the
        # next one.
        # Remove the and seq_name!
        if line == '':
            idx_end_frame = cur_idx
            # # Store results
            # if not(seq_name == ''):
            list_sequence.append({
                'sequence_path': ROOT_PATH_SEQ + seq_name + '/',
                'idx_starting_frame': idx_starting_frame,
                'idx_end_frame': idx_end_frame,
            })
            # Re-init the variables
            seq_name = ''
            idx_starting_frame = -1
            cur_idx = -1
            idx_end_frame = -1
            # Proceed to next line
            continue

        if seq_name == '':
            seq_name = '_'.join(line.split('_')[:-2])
            idx_starting_frame = int(line.split('_')[-2])
        
            # Temporary:
            # remove !
            # if seq_name != 'Vlog_720P-0d79':
            #     seq_name = ''
            #     idx_starting_frame = -1
            #     continue

        cur_idx = int(line.split('_')[-2])

    f.close()
    return list_sequence
