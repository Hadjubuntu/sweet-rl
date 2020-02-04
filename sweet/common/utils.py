from datetime import datetime as dt


def now_str(format="%d-%m-%Y_%H_%M_%S"):
    return dt.now().strftime(format)
