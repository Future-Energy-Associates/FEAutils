from datetime import datetime

# converting UNIX to str and back
str_format = "%d-%m-%y %H.%M.%S"
unix_to_str = lambda unix: datetime.fromtimestamp(unix).strftime(str_format)
str_to_unix = lambda string: int(datetime.strptime(string, str_format).timestamp())
