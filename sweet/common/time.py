import datetime as dt


def dt_to_str(dt_seconds):
    """
    Converts delta time into string "hh:mm:ss"
    """
    return str(dt.timedelta(seconds=dt_seconds))
