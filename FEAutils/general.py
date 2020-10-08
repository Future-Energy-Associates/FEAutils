import pandas as pd
import requests
from IPython.display import JSON

# General functions
from collections.abc import Iterable

def summarise_wrapper(df, operation, cond='tuple()', col=None, wrapper_type='auto'):
    """
    Carries out the individual summary operations 

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe over which the summarisation will be carried out
    operation : str (for method) or function/method
        Name of the function/method, e.g. could pass max or 'max'
    cond : str
        Condition that will be passed to the dataframe's .query method
    col : str
        Column used for the summary statistic
    wrapper_type : str
        One of: 'auto', 'method', 'function'.
        For 'auto' the method will be used
        unless it does not exist

    Returns
    -------
    df_summary : pd.DataFrame
        Dataframe of summary statistics
        
    """
    
    cond_method = lambda df, method, cond='tuple()', col=None: getattr(df.query(cond), method)() if col is None else getattr(df.query(cond)[col], method)()
    cond_function = lambda df, function, cond='tuple()', col=None: function(df.query(cond)) if col is None else function(df.query(cond)[col])

    if '__name__' in dir(operation) and wrapper_type!='function':
        if operation.__name__ not in dir(df) and wrapper_type=='auto':
            pass
        else:
            operation = operation.__name__
        
    wrapper_type_to_process = {
        'auto': lambda df, operation, cond, col: cond_method(df, operation, cond, col) if operation in dir(df) else cond_function(df, operation, cond, col),
        'method': cond_method,
        'function': cond_function
    }
    
    return wrapper_type_to_process[wrapper_type](df, operation, cond, col)

def summarise(df_gb:pd.core.groupby.generic.DataFrameGroupBy, agg_ops:dict):
    """
    summarise accepts a dictionary of aggregation operations that 
    will be used to construct the returned summary dataframe. The
    dictionary must map from the name of the new summary column to
    either a function/method, or a tuple that contains a function/
    method as well as a dictionary of keyword arguments. 

    Parameters
    ----------
    df_gb : pd.core.groupby.generic.DataFrameGroupBy
        Groupby object over which the summarisation will be carried out
    agg_ops : dict
        Mapping from column names to aggregation operations

    Returns
    -------
    df_summary : pd.DataFrame
        Dataframe of summary statistics

    e.g. 

    ```
    agg_ops = {
        'max_customers': (max, dict(col='customers')),                   # By default the .max() method will be used but
        'max_sales': (max, dict(col='sales', wrapper_type='function')),  # it can also be used as a function if specified.
        'total_days': len
        'days_open': (len, dict(cond='store_open == True')),
        'sales_tues': ('sum', dict(col='sales', cond='day == 2'))        # When using methods they can be passed as a string
    }

    df2 = (df1
           .groupby('store')
           .pipe(summarise, agg_ops)
          )
    ```
    """
    
    # Handling agg_ops with no kwargs
    cleaned_agg_ops = {
        k: (v if isinstance(v, Iterable) else (v, {}))
        for k, v 
        in agg_ops.items()
    }
    
    # Calculating summary statistics
    df_summary = (df_gb
                  .apply(lambda df: pd.Series({
                      name: summarise_wrapper(df, operation, **kwargs) 
                      for name, (operation, kwargs)
                      in cleaned_agg_ops.items()
                  }))
                 )
    
    return df_summary

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
