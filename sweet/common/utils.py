from datetime import datetime as dt


def now_str(format="%d-%m-%Y_%H_%M_%S"):
    """
    Returns current time string with default
    format compatible for filename
    """
    return dt.now().strftime(format)


def list_to_dict(input_list):
    """
    Transform list to dictionnary
    Note: keys are incremental integer
    """
    res = dict()
    for i, e in enumerate(input_list):
        res[i] = e
    return res
