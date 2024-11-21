import numpy as np
import scipy.signal as signal
import h5py
import argparse
import pandas as pd
from sklearn.metrics import roc_curve, auc

def bandpass_filter(signal_data, fs, lowcut=0.5, highcut=40.0, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    # Apply zero-phase filtering to avoid phase distortion
    y = signal.filtfilt(b, a, signal_data)
    return y

def notch_filter(signal_data, fs, freq=50.0, Q=30.0):
    # Adjust frequency parameter to ensure correct w0
    w0 = freq / (0.5 * fs)
    b, a = signal.iirnotch(w0, Q)
    y = signal.filtfilt(b, a, signal_data)
    return y

def pan_tompkins_detector(ecg_signal, fs):
    diff = np.diff(ecg_signal)
    squared = diff ** 2
    window_size = int(0.150 * fs)
    integrated = np.convolve(squared, np.ones(window_size) / window_size, mode='same')
    threshold = np.mean(integrated) * 0.5
    distance = int(0.2 * fs)
    peaks, _ = signal.find_peaks(integrated, height=threshold, distance=distance)
    return peaks

def detect_qrs_onset_offset(ecg_signal, r_peaks, fs):
    qrs_onsets = []
    qrs_offsets = []
    for r_peak in r_peaks:
        onset = r_peak
        while onset > 0 and ecg_signal[onset] > ecg_signal[onset - 1]:
            onset -= 1
        qrs_onsets.append(onset)

        offset = r_peak
        while offset < len(ecg_signal) - 1 and ecg_signal[offset] > ecg_signal[offset + 1]:
            offset += 1
        qrs_offsets.append(offset)
    return qrs_onsets, qrs_offsets


def detect_p_waves(ecg_signal, qrs_onsets, fs):
    p_peaks = []
    for onset in qrs_onsets:
        search_region = ecg_signal[max(0, onset - int(0.2 * fs)):onset]
        if len(search_region) == 0:
            p_peaks.append(None)
            continue
        # Find P wave peak
        p_peak = np.argmax(search_region) + max(0, onset - int(0.2 * fs))
        p_peaks.append(p_peak)
    return p_peaks


def calculate_intervals(p_peaks, qrs_onsets, qrs_offsets, fs):
    pr_intervals = []
    qrs_durations = []
    for p_peak, qrs_onset, qrs_offset in zip(p_peaks, qrs_onsets, qrs_offsets):
        if p_peak is not None:
            pr_interval = (qrs_onset - p_peak) / fs  # 초 단위
            pr_intervals.append(pr_interval)
        else:
            pr_intervals.append(None)
        qrs_duration = (qrs_offset - qrs_onset) / fs  # 초 단위
        qrs_durations.append(qrs_duration)
    return pr_intervals, qrs_durations

def calculate_rr_intervals(r_peaks, fs):
    rr_intervals = np.diff(r_peaks) / fs  # 초 단위
    return rr_intervals

def calculate_hrv_metrics(rr_intervals):
    rr_mean = np.mean(rr_intervals) if len(rr_intervals) > 0 else None
    rr_sdnn = np.std(rr_intervals) if len(rr_intervals) > 0 else None
    rr_rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2)) if len(rr_intervals) > 1 else None
    return rr_mean, rr_sdnn, rr_rmssd


def analyze_ecg(ecg_signals, fs):
    analysis_results = []

    # for i in range(10):
    for i in range(ecg_signals.shape[0]):
        ecg_signal = ecg_signals[i]
        ecg_filtered = bandpass_filter(ecg_signal, fs, lowcut=0.5, highcut=40.0, order=2)
        ecg_filtered = notch_filter(ecg_filtered, fs, freq=50.0, Q=30.0)

        r_peaks = pan_tompkins_detector(ecg_filtered, fs)
        rr_intervals = calculate_rr_intervals(r_peaks, fs)
        rr_mean, rr_sdnn, rr_rmssd = calculate_hrv_metrics(rr_intervals)
        qrs_onsets, qrs_offsets = detect_qrs_onset_offset(ecg_filtered, r_peaks, fs)
        p_peaks = detect_p_waves(ecg_filtered, qrs_onsets, fs)
        pr_intervals, qrs_durations = calculate_intervals(p_peaks, qrs_onsets, qrs_offsets, fs)

        analysis_result = {
            'RR Intervals': rr_intervals,
            # 'HRV Metrics': {
            #     'Mean RR': rr_mean,
            #     'SDNN': rr_sdnn,
            #     'RMSSD': rr_rmssd
            # },
            'Mean RR': rr_mean,
            'SDNN': rr_sdnn,
            'RMSSD': rr_rmssd,
            'PR Intervals': pr_intervals,
            'QRS Durations': qrs_durations,
            'P Peaks': p_peaks
        }


        analysis_results.append(analysis_result)

    return analysis_results


def categorize_ecg_condition(analysis_results):
    categories = []

    for result in analysis_results:
        pr_intervals = result['PR Intervals']
        qrs_durations = result['QRS Durations']
        rr_intervals = result['RR Intervals']
        p_peaks = result.get('P Peaks', [None] * len(pr_intervals))
        rr_rmssd = result['HRV Metrics']['RMSSD'] if result['HRV Metrics']['RMSSD'] is not None else 0

        av_block_threshold = 0.2
        # bundle_branch_block_threshold = 0.12
        # bradycardia_threshold = 1.0
        # tachycardia_threshold = 0.6
        # threshold_rmssd = 0.1

        av_block_threshold = 0.1225
        bundle_branch_block_threshold = 0.0350
        bradycardia_threshold = 1.2450
        tachycardia_threshold = 1.3275
        threshold_rmssd = 0.1

        # 상태 조건
        av_block = any(pr is not None and pr > av_block_threshold for pr in pr_intervals)
        bundle_branch_block = any(qrs is not None and qrs >= bundle_branch_block_threshold for qrs in qrs_durations)
        bradycardia = any(rr is not None and rr > bradycardia_threshold for rr in rr_intervals)
        tachycardia = any(rr is not None and rr < tachycardia_threshold for rr in rr_intervals)
        # atrial_fibrillation = (all(p is None for p in p_peaks) or rr_rmssd > threshold_rmssd) if p_peaks else False

        conditions = {
            'AV Block': av_block,
            'Bundle Branch Block': bundle_branch_block,
            'Bradycardia': bradycardia,
            'Tachycardia': tachycardia,
            # 'Atrial Fibrillation': atrial_fibrillation
        }

        categories.append(conditions)

    return categories

def find_optimal_threshold(args):
    # Load data
    data = pd.read_csv(args.path_to_csv)
    data = data.dropna(subset=[data.columns[0], data.columns[1]])

    # Extract labels and PR Intervals
    y_true = data.iloc[:, 0]  # First column as labels
    y_scores = data.iloc[:, 1]  # Second column as variables

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores) # False Positive Rate and True Positive Rate
    roc_auc = auc(fpr, tpr)

    # Calculate Youden's Index to find the optimal threshold
    youden_index = tpr - fpr
    optimal_threshold_index = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_threshold_index]

    print(f"----------{data.columns[0]}----------")
    print(f"AUC: {roc_auc}")
    print(f"Optimal PR Interval Threshold: {optimal_threshold:.4f}")

    # Return the optimal threshold and AUC
    return optimal_threshold, roc_auc

def save_analysis_results(args):
    with h5py.File(args.path_to_hdf5, 'r') as f:
        ecg_signals = f['tracings'][:, :, 3]

    # convert signals into voltage
    fs = 400
    analysis_results = analyze_ecg(ecg_signals * 1e-4, fs)
    # sort_results = categorize_ecg_condition(analysis_results)
    # Y = pd.read_csv(args.path_to_csv, header=None).values
    Y = pd.read_csv(args.path_to_csv).values

    data_rows = []

    # Loop through each item in Y and analysis_results and combine them
    for i in range(len(Y)):
        rr_intervals = analysis_results[i].get('RR Intervals', np.array([]))

        row = {
            '1dAVb': Y[i][0],
            'RBBB': Y[i][1],
            'LBBB': Y[i][2],
            'SB': Y[i][3],
            'AF': Y[i][4],
            'ST': Y[i][5],
            'P Peaks (Max)': max(analysis_results[i].get('P Peaks', [np.nan])),
            'QRS Durations (Max)': max(analysis_results[i].get('QRS Durations', [np.nan])),
            'PR Intervals (Max)': max(analysis_results[i].get('PR Intervals', [np.nan])),
            'RMSSD': analysis_results[i].get('RMSSD', np.nan),
            'SDNN': analysis_results[i].get('SDNN', np.nan),
            'Mean RR': analysis_results[i].get('Mean RR', np.nan),
            'RR Intervals (Max)': max(rr_intervals) if len(rr_intervals) > 0 else np.nan,
            'RR Intervals (Min)': min(rr_intervals) if len(rr_intervals) > 0 else np.nan,
        }
        data_rows.append(row)

    # Create the DataFrame
    df = pd.DataFrame(data_rows)

    # Save the DataFrame to CSV
    df.to_csv("combined_max_output.csv", index=False)

    print("Data has been successfully exported to 'combined_output.csv'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('path_to_hdf5', type=str, help='Path to HDF5 file containing tracings')
    parser.add_argument('path_to_csv', type=str, help='Path to CSV file containing annotations')
    parser.add_argument('--dataset_name', type=str, default='tracings', help='Dataset name in HDF5 file')
    args = parser.parse_args()

    # save_analysis_results(args)
    find_optimal_threshold(args)