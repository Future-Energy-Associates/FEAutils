import pandas as pd
import requests
from IPython.display import JSON

# general functions

reverse_dict = lambda original_dict: {
    value: key for key, value in original_dict.items()
}

capfirst = lambda s: s[:1].upper() + s[1:]

keep_index = lambda df, func, *args, **kwargs: pd.DataFrame(
    func(df, *args, **kwargs), index=df.index
)

view_dict_head = lambda d, n=5: JSON(pd.Series(d).head(n).to_dict())

msg_2_data = lambda record_msg: f'{"{"}"text":"{record_msg}"{"}"}'


def send_slack_msg(
    record_msg, slack_channel_url="",
):
    headers = {"Content-type": "application/json"}
    data = msg_2_data(record_msg)

    try:
        requests.post(slack_channel_url, headers=headers, data=data)
    except:
        print(f"Slack message failed to send\n{record_msg}")
