from datetime import datetime, timedelta

# converting UNIX to str and back
str_format = "%d-%m-%y %H.%M.%S"
unix_to_str = lambda unix: datetime.fromtimestamp(unix).strftime(str_format)
str_to_unix = lambda string: int(datetime.strptime(string, str_format).timestamp())


def create_delay_seconds(hour: int, minute: int):
    """
    Creates delay in seconds from now until specified hour and minute in the same 24 hour period.

    Parameters
    ----------
    -hour:
        Hour of the time to create delay for
    -minute:
        Minute of the time to create delay for

    Returns
    -------
    -now:
        The time now as a datetime object
    -delay:
        The delay until the specified times in seconds as an integer
    -delay_time:
        The time the delay is set to as a datetime object

    """

    now = datetime.now().replace(microsecond=0)
    hours_delta = hour - now.hour if now.hour <= hour else hour + (24 - now.hour)

    delay_time = now.replace(minute=minute, second=0, microsecond=0) + timedelta(
        hours=hours_delta
    )

    delay = (delay_time - now).total_seconds()

    return now, delay, delay_time


def strfdelta(tdelta: timedelta, fmt: str):
    """
    Formats datetime timedelta object containing days, hours, minutes and seconds in a specfied string format.
    """

    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)

    return fmt.format(**d)
