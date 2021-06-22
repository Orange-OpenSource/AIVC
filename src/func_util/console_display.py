"""Utilitary function for a neat console display
"""

from func_util.cluster_mngt import COMPUTE_PARAM

LINE_LENGTH = 80
MSG_TYPE_LEN = 10
FN_NAME_LEN = 30
VAR_NAME_LEN = 30

def print_dic_content(dic_to_print, dic_name=''):
    """Print the content of a dictionnary:
            key: value
            key: value
    etc.
    """
    if COMPUTE_PARAM.get('flag_quiet'):
        return

    total_length = 120
    msg = '\n' + str(dic_name).center(total_length) + '\n'
    size_col_1 = 60
    size_col_2 = total_length - size_col_1 - 1
    msg += '+' + '-' * size_col_1 + '+' + '-' * size_col_2 + '+\n'
    for key, value in dic_to_print.items():
        # Recursive dic printing when we have a dic of dic
        if type(value) is dict:
            print_dic_content(value, dic_name=key)
        else:
            msg += '|' + key.center(size_col_1) + '|'
            msg += str(value).center(size_col_2) + '|\n'
    msg += '+' + '-' * size_col_1 + '+' + '-' * size_col_2 + '+\n'
    print(msg)


def print_log_msg(msg_type, fn_name, var_name, var):
    if COMPUTE_PARAM.get('flag_quiet'):
        return

    print(('[' + str(msg_type) + ']').ljust(MSG_TYPE_LEN, ' ')
          + (' | ' + str(fn_name)).ljust(FN_NAME_LEN, ' ')
          + (' | ' + str(var_name)).ljust(VAR_NAME_LEN, ' ')
          + ' | ' + str(var))

