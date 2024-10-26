import os
import warnings
import numpy as np
import pandas as pd

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


def load_data(experiment_name):
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
        usecols=[
            "Date",
            "Time",
            "GPU Rail Powers (avg) [W]",
            "Total DRAM Power [W]",
            "CPU Package Power [W]",
            "IA Cores Power [W]",
        ],
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


def get_experiments_data(experiment_name):
    """
    experiments_data with keys that are combinations of independent variables and the values are dictionaries, with 2 keys:
    - durations_per_run: contains a list of durations
    - measurements_per_run: contains of a list of pandas data frames with the measurements
    """

    measurements, timestamps_data = load_data(experiment_name)
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