import pandas as pd
import requests
from IPython.display import JSON

# General functions
from collections.abc import Iterable

cond_attr = lambda df, attr, cond='tuple()', col=None: getattr(df.query(cond), attr)() if col is None else getattr(df.query(cond)[col], attr)()
cond_func = lambda df, func, cond='tuple()', col=None: func(df.query(cond)) if col is None else func(df.query(cond)[col])

def summarise(df, agg_funcs):
    cleaned_agg_funcs = {
        k: (v if isinstance(v, Iterable) else (v, {}))
        for k, v 
        in agg_funcs.items()
    }
    
    df_summarised = (df
                     .apply(lambda df: pd.Series({
                         name: func(df, **kwargs) 
                         for name, (func, kwargs)
                         in cleaned_agg_funcs.items()
                     }))
                    )
    
    return df_summarised

reverse_dict = lambda original_dict: {
    value: key for key, value in original_dict.items()
}

capfirst = lambda s: s[:1].upper() + s[1:]

keep_index = lambda df, func, *args, **kwargs: pd.DataFrame(
    func(df, *args, **kwargs), index=df.index
)

view_dict_head = lambda d, n=5: JSON(pd.Series(d).head(n).to_dict())


# Slack
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
