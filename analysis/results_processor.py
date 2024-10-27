import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, kruskal

warnings.filterwarnings("ignore")


def preprocess_measurements_data(measurements):
    # create a data-time column
    date_time = zip(measurements["Date"].to_list(), measurements["Time"].to_list())
    measurements["timestamp"] = [
        date_val + " " + time_val for date_val, time_val in date_time
    ]
    measurements["timestamp"] = pd.to_datetime(
        measurements["timestamp"], format="%d.%m.%Y %H:%M:%S.%f"
    )

    # remove the columns that are not used
    measurements = measurements.drop(["Date", "Time"], axis=1)

    # convert columns to floats
    for column in measurements.columns:
        if column == "timestamp":
            continue

        measurements[column] = measurements[column].astype("float")

    return measurements


def load_data(experiment_name, features):
    # read the measurements recorded with HWiNFO and the stored timestamps
    results_path = f"../results/{experiment_name}/"
    file_names = os.listdir(results_path)

    timestamp_file_paths = [
        results_path + file_name
        for file_name in file_names
        if "timestamps" in file_name
    ]
    measurement_file_path = results_path + "measurements.csv"

    measurements = pd.read_csv(
        measurement_file_path,
        encoding="unicode_escape",
        usecols=["Date", "Time"] + features,
        low_memory=False,
    )[:-2]

    timestamps_data = {}

    for timestamp_file_path in timestamp_file_paths:
        run_timestamps = pd.read_csv(timestamp_file_path, index_col="run_number")
        timestamps_name = (
            timestamp_file_path.split("/")[-1]
            .replace("timestamps_", "")
            .replace(".csv", "")
        )
        timestamps_data[timestamps_name] = run_timestamps

    return preprocess_measurements_data(measurements), timestamps_data


def get_runs_measurements(measurements, run_timestamps, measurements_timestamps):
    # compute for each experiment run the duration and get the associated measurements
    start_timestamps = pd.to_datetime(run_timestamps["start_timestamp"]).to_list()
    end_timestamps = pd.to_datetime(run_timestamps["end_timestamp"]).to_list()

    experiment_intervals = zip(start_timestamps, end_timestamps)
    measurements_per_run = []
    durations_per_run = []

    for run_index, interval in enumerate(experiment_intervals):
        duration_seconds = (interval[1] - interval[0]).seconds
        duration_microseconds = (interval[1] - interval[0]).microseconds
        durations_per_run.append(duration_seconds + duration_microseconds / 1000000)

        considered_timestamps = measurements_timestamps[
            (interval[0] <= measurements_timestamps)
            & (measurements_timestamps <= interval[1])
        ]

        measurements_current_run = measurements[
            measurements["timestamp"].isin(considered_timestamps)
        ]
        measurements_current_run.loc[:, "run_index"] = run_index

        measurements_per_run.append(measurements_current_run)

    return durations_per_run, pd.concat(measurements_per_run).drop("timestamp", axis=1)


def get_experiments_data(experiment_name, features):
    """
    experiments_data with keys that are combinations of independent variables and the values are dictionaries, with 2 keys:
    - durations_per_run: contains a list of durations
    - measurements_per_run: contains of a list of pandas data frames with the measurements
    """

    measurements, timestamps_data = load_data(experiment_name, features)
    measurements_timestamps = measurements["timestamp"].to_numpy()

    experiments_data = {}

    for timestamps_name, runs_timestamps in timestamps_data.items():
        durations_per_run, measurements_per_run = get_runs_measurements(
            measurements, runs_timestamps, measurements_timestamps
        )

        experiments_data[timestamps_name] = {
            "durations_per_run": durations_per_run,
            "measurements_per_run": measurements_per_run,
        }

    return experiments_data


def compute_total_energy_per_run(experiments_data):
    # compute total energy consumption per run
    for experiment_name in experiments_data.keys():
        experiments_data[experiment_name]["measurements_per_run"] = (
            experiments_data[experiment_name]["measurements_per_run"]
            .groupby("run_index")
            .apply(lambda x: np.sum(x * 0.1))
            .drop("run_index", axis=1)
            .rename(
                {
                    "CPU Package Power [W]": "CPU Package Energy [J]",
                    "IA Cores Power [W]": "IA Cores Energy [J]",
                    "Total DRAM Power [W]": "DRAM Energy [J]",
                    "GPU Rail Powers (avg) [W]": "GPU Energy [J]",
                },
                axis=1,
            )
        )

    return experiments_data


def get_variation_runs_data(
    runs_data, small, medium, large, independent_variable, variable_values
):
    # combines measurements into one data frame, where observations are grouped
    runs_list = []

    for index, data_independent_variable in enumerate([small, medium, large]):
        runs = pd.concat(
            [
                runs_data[experiment_name]["measurements_per_run"]
                for experiment_name in data_independent_variable
            ]
        )
        durations = np.array(
            [
                runs_data[experiment_name]["durations_per_run"]
                for experiment_name in data_independent_variable
            ]
        ).flatten()
        runs.loc[:, independent_variable] = variable_values[index]
        runs.loc[:, "duration [s]"] = durations

        runs_list.append(runs)

    return pd.concat(runs_list)


def prepare_plotting_data(data, independent_variable, cols_to_drop):
    data = data.drop(cols_to_drop, axis=1)
    independent_variable_values = data[independent_variable].to_list()
    data = data.drop(independent_variable, axis=1)
    features_no = len(data.columns)
    data = data.melt()

    data[independent_variable] = independent_variable_values * features_no

    return data


def compute_kruskal_wallis(
    data, independent_var_name, independent_var_vals, measurement_name
):
    return kruskal(
        data[data[independent_var_name] == independent_var_vals[0]][
            measurement_name
        ].to_list(),
        data[data[independent_var_name] == independent_var_vals[1]][
            measurement_name
        ].to_list(),
        data[data[independent_var_name] == independent_var_vals[2]][
            measurement_name
        ].to_list(),
    )


def get_ci_deviation(values):
    return round(1.96 * values.std() / np.sqrt(len(values)), 2)

def compute_wilcoxon(
    data,
    first_embedding_model,
    second_embedding_model,
    independent_var,
    alternative="less",
):
    return wilcoxon(
        x=data[data["embedding_model"] == first_embedding_model][
            independent_var
        ].to_list(),
        y=data[data["embedding_model"] == second_embedding_model][
            independent_var
        ].to_list(),
        alternative=alternative,
    )