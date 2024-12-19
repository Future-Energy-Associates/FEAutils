import pandas as pd
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d  # type: ignore
from scipy.stats import norm  # type: ignore
from scipy.stats._morestats import _calc_uniform_order_statistic_medians  # type: ignore

SEASON_MAP = {
    1: "Wtr",
    2: "Wtr",
    3: "Wtr",
    4: "Spr",
    5: "Spr",
    6: "Smr",
    7: "Smr",
    8: "Hsr",
    9: "Aut",
    10: "Aut",
    11: "Wtr",
    12: "Wtr",
}

WEEKDAY_MAP = {0: "Wd", 1: "Wd", 2: "Wd", 3: "Wd", 4: "Wd", 5: "Sat", 6: "Sun"}


def settle_period_from_dt_index(index):
    return (index.hour * 2 + index.minute // 30) + 1


def clean_df(df, unstack=True):
    assert "Settlement period" in df.columns, "Settlement period column not found in df"
    df.rename(columns={"Settlement period": "Timestamp"}, inplace=True)
    assert "Timestamp" in df.columns, "Timestamp column not found in df"
    df["Timestamp"] = df["Timestamp"].apply(
        lambda t: pd.Timestamp.combine(pd.Timestamp("2000-01-01"), t)
    )
    df.set_index("Timestamp", inplace=True)

    if unstack:
        # for each column in the df split the column name into season and weekday and assign to multiindex
        li = []
        for col in df.columns:
            season, weekday = col.split(" ")
            new_df = df[col].to_frame("kwh")
            new_df["season"] = season
            new_df["weekday"] = weekday
            li.append(new_df)

        df = pd.concat(li, axis=0)

    df["settlement_period"] = settle_period_from_dt_index(df.index)

    return df


def parse_raw_profile_classes(filepath: Path, unstack=True, return_long=False):
    """Reads the raw data from the excel file and returns a
    dictionary of profiles for each class

    Args:
        filepath (Path): Path to the excel file

        Returns:
        dict: Dictionary of profiles for each class

    """

    profile_classes = {}
    li = []
    for i in range(8):
        df = pd.read_excel(filepath, skiprows=2 + 52 * i)[:48]
        assert (
            "Settlement period" in df.columns
        ), "Settlement period column not found in df"
        print(df.columns)
        dfc = clean_df(df)
        profile_classes[i + 1] = clean_df(dfc, unstack=unstack)
        li.append(dfc)

    df_long = pd.concat(li)

    if return_long:
        return df_long

    return profile_classes


class CorrelatedSampler:
    def __init__(
        self, distribution: np.ndarray, summary: int = 100, seed: int | None = None
    ):
        x = np.sort(distribution)
        y = _calc_uniform_order_statistic_medians(len(distribution))

        # We want to normalise the distribution before taking autocorr
        y2 = norm.ppf(interp1d(x, y)(distribution))
        self._autocorr = np.corrcoef(y2[:-1], y2[1:])[0, 1]
        self._w2 = np.sqrt(1 - self._autocorr**2)
        self.rng = np.random.default_rng(seed)
        if summary:
            assert summary > 3, "Need more than 3 summary points"
            y2 = np.sort(np.append(norm.cdf(np.linspace(-3, 3, summary - 2)), [0, 1]))
            x = interp1d(y, x, bounds_error=False, fill_value=(x.min(), x.max()))(y2)
            y = y2

        # The way this works is to just translate from the normal copula
        # to our empirical distribution or its summary
        self.map_norm_to_dist = interp1d(
            norm.ppf(y), x, bounds_error=False, fill_value=(x.min(), x.max())
        )


def get_missing_periods(year_kwh):
    """
    Get the missing periods in the year_kwh
    series for each season and weekday
    """

    missing_periods = year_kwh[year_kwh.kwh.isna()].copy()
    missing_periods = (
        missing_periods.groupby(["season", "weekday"]).kwh.sum().index.to_list()
    )
    return missing_periods


def process_kwh_data(kwh, season_map=SEASON_MAP, weekday_map=WEEKDAY_MAP):
    """Process the kwh data to create a timeseries of the last year
    and reindex the kwh series to the last year
    map all periods to the profile class season, weekday and settelment period

    Args:
        kwh (pd.Series): Input kwh series
        season_map (dict): Map of month to elexon profile season
        weekday_map (dict): Map of weekday to elexon profile weekday

    Returns:
        pd.DataFrame: Yearly kwh dataframe
        float: Percentage of missing data
    """

    # get the last day of the previous month
    last_day = pd.Timestamp.now().replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    ) - pd.Timedelta(days=1)

    # create a timeseries of the last year and reindex the kwh series to the last year
    date_range = pd.date_range(last_day - pd.DateOffset(years=1), last_day, freq="30T")
    year_kwh = kwh.reindex(date_range).to_frame("kwh")

    # check that the input kwh series is at least one month long
    PERIODS_IN_MONTH = 48 * 30
    assert (
        len(year_kwh.dropna()) > PERIODS_IN_MONTH
    ), "Length of input kwh series was not at least one month"

    # map all periods to the profile class season, weekday and settelment period
    year_kwh["season"] = year_kwh.index.month.map(season_map)
    year_kwh["weekday"] = year_kwh.index.dayofweek.map(weekday_map)
    year_kwh["settlement_period"] = settle_period_from_dt_index(year_kwh.index)

    # get the percentage of missing data
    percentage_missing = year_kwh.kwh.isna().sum() / len(year_kwh)

    return year_kwh, percentage_missing


def check_existing_data_length(year_kwh, season, weekday, min_periods=48 * 2):
    """Check how much data exists for a given season and weekday

    Args:
        year_kwh (pd.DataFrame): Yearly kwh dataframe
        season (str): Season
        weekday (str): Weekday
        min_periods (int): Minimum number of settlement periods to consider (48 SPs per day)

        Returns:
        bool: True if enough data exists, False otherwise
    """

    existing_data = year_kwh[
        (year_kwh.season == season) & (year_kwh.weekday == weekday)
    ]

    return len(existing_data.dropna()) > min_periods


def fill_missing_season_weekday_mean(year_kwh, season, weekday):
    """
    Where there is enough data for a given season and weekday,
    fill missing values with the mean of the existing data for
    the same season and weekday

    Args:
        year_kwh (pd.DataFrame): Yearly kwh dataframe
        season (str): Season
        weekday (str): Weekday

        Returns:
        pd.DataFrame: Updated dataframe with filled values
    """

    filtered_data = year_kwh[
        (year_kwh.season == season) & (year_kwh.weekday == weekday)
    ].copy()

    # Calculate the mean of existing data for the filtered data
    group = ["season", "weekday", "settlement_period"]
    existing_data_means = filtered_data.groupby(group).kwh.mean()
    existing_data_means = existing_data_means.to_frame("mean_kwh")
    existing_data_means.reset_index(inplace=True)

    # Merge the mean values with the filtered data
    filtered_data = filtered_data.merge(existing_data_means, on=group, how="left")
    filtered_data.loc[filtered_data.mean_kwh.notna(), "estimate"] = True

    # Fill NaN values in 'kwh' with the corresponding mean values
    filtered_data["kwh"] = filtered_data["kwh"].fillna(filtered_data["mean_kwh"])

    # Update the original dataframe with the filled values
    year_kwh.loc[(year_kwh.season == season) & (year_kwh.weekday == weekday), "kwh"] = (
        filtered_data.kwh.values
    )

    year_kwh.loc[
        (year_kwh.season == season) & (year_kwh.weekday == weekday), "estimate"
    ] = filtered_data.estimate.values

    return year_kwh


def fill_missing_season_weekday_factor(year_kwh, season, weekday, profile_class):
    """Where there is not enough data for a given season and weekday,
    fill missing values with the average HH profile multiplied by the seasonal adjustment factor for the
    weekday and season

    Args:
        year_kwh (pd.DataFrame): Yearly kwh dataframe
        season (str): Season
        weekday (str): Weekday
        profile_class (pd.DataFrame): Profile class dataframe

        Returns:
        pd.DataFrame: Updated dataframe with filled values
    """

    elexon_season_weekday = f"{season} {weekday}"

    # get the adjustment factor from the profile class for the season and weekday
    profile_class["annual_mean"] = profile_class.drop(columns="settlement_period").mean(
        axis=1
    )

    profile_class["seasonal_adjustment_factor"] = (
        profile_class[elexon_season_weekday] / profile_class.annual_mean
    )

    profile_class.to_csv(f"profile_class_{season}_{weekday}.csv")

    # get the kwh for the average profile by settlement period
    average_profile = year_kwh.groupby("settlement_period").kwh.mean()

    # merge the average profile with the profile class to get the seasonal adjustment factor
    average_profile = average_profile.to_frame("average_profile").merge(
        profile_class[["settlement_period", "seasonal_adjustment_factor"]],
        on="settlement_period",
    )

    # apply the seasonal adjustment factor to the average profile
    average_profile["season_weekday_adjusted_profile"] = (
        average_profile.average_profile * average_profile.seasonal_adjustment_factor
    )

    average_profile.to_csv(f"average_profile_{season}_{weekday}.csv")

    season_weekday_adjusted_profile = average_profile.set_index(
        "settlement_period"
    ).season_weekday_adjusted_profile

    # now fill the missing values with the adjusted profile
    idx = (year_kwh.season == season) & (year_kwh.weekday == weekday)

    year_kwh.loc[idx, "kwh"] = year_kwh.loc[idx, "settlement_period"].map(
        season_weekday_adjusted_profile
    )

    year_kwh.loc[idx, "estimate"] = True

    return year_kwh


def infer_timeseries_with_profile_class(
    kwh: pd.Series,
    profile_classes: dict,
    class_number: int,
):
    """Infer the profile class of a kwh series

    1. from the kwh series, take any data from the last 12 months
    2. map all periods to the profile class season, weekday and settelment period
    4. for each missing period:
           i. if there is existing data for the equivalent, season, weekeday and settlement period take the mean and use that
          ii. if there is no existing data, take the average profile for the
              weekday and settlement period and apply the seasonal adjustment factor from the profile class

    Args:
        kwh (pd.Series): Input kwh series
        profile_classes (dict): Dictionary of profile classes
        class_number (int): Profile class number
        season_map (dict): Map of month to elexon profile season
        weekday_map (dict): Map of weekday to elexon profile weekday

    Returns:
        pd.Series: Inferrred kwh timeseries
    """

    profile_class = profile_classes[class_number]

    # assert that all settlement_period values are in range 1-48
    assert profile_class.settlement_period.min() == 1
    assert profile_class.settlement_period.max() == 48

    year_kwh, percentage_missing = process_kwh_data(kwh)

    assert year_kwh.settlement_period.min() == 1
    assert year_kwh.settlement_period.max() == 48

    # Add estimate flag
    year_kwh["estimate"] = False

    # get summary of elexon periods that are missing by season, weekday
    missing_periods = get_missing_periods(year_kwh)

    for season, weekday in missing_periods:
        # check if there is existing data for the equivalent, season, weekeday
        enough_existing_data = check_existing_data_length(year_kwh, season, weekday)

        if enough_existing_data:
            # fill missing values with the mean of the existing data
            year_kwh = fill_missing_season_weekday_mean(year_kwh, season, weekday)
        else:
            # Use an adjustment factor for season and weekday applied to the average HH profile
            year_kwh = fill_missing_season_weekday_factor(
                year_kwh, season, weekday, profile_class
            )

    # if any remaining missing values, use interpolation with a limit of 8 periods
    if year_kwh.kwh.isna().any():
        year_kwh.loc[year_kwh.kwh.isna(), "estimate"] = True
        year_kwh["kwh"] = year_kwh.kwh.interpolate(method="linear", limit=8)

    assert year_kwh.kwh.isna().sum() == 0

    return year_kwh, percentage_missing


def create_timeseries_files(filepath):

    # read in csv
    df = pd.read_csv(filepath)

    df["Timestamp"] = pd.to_datetime(df.Timestamp)
    df.set_index("Timestamp", inplace=True)

    # bring to current year
    month = df.index.max().month + 2
    delta = (
        df.index.max().replace(year=pd.Timestamp.now().year, month=month)
        - df.index.max()
    )
    df.index = df.index + delta

    # for each column in the df
    # choose length of timeseries in months as random int between 1 and 12 and random days between 0 and 15
    for col in df.columns:
        days_length = np.random.randint(1, 12) * 30
        days_offset = np.random.randint(0, 15) + np.random.randint(0, 3) * 30

        end = df.index.max() - pd.Timedelta(days=days_offset)
        time_idx = pd.date_range(end - pd.Timedelta(days=days_length), end, freq="30T")

        df_out = df[col].reindex(time_idx).dropna().to_frame("kwh")

        # now drop up to 20 individual rows from the df
        n = np.random.randint(0, 200)
        idx = np.random.randint(3, df_out.shape[0], n)

        # df_out.index[idx]
        df_out.iloc[idx, 0] = None

        df_out.columns = ["kwh"]

        name = str(col).replace(".csv", "")
        df_out.to_csv(Path("data", f"sme_{name}.csv"))
