import numpy as np
import scipy.signal as signal
import h5py
import argparse
import pandas as pd
from sklearn.metrics import roc_curve, auc


def remove_padding(ecg_signal):
    """
    Remove zero padding from an ECG signal.
    """
    non_zero_indices = np.where(ecg_signal != 0)[0]
    if len(non_zero_indices) == 0:
        raise ValueError("Signal is entirely zero.")
    return ecg_signal[non_zero_indices[0]:non_zero_indices[-1] + 1]


def bandpass_filter(signal_data, fs, lowcut=0.5, highcut=40.0, order=2):
    """
    Bandpass filter the signal_data between lowcut and highcut frequencies.

    Parameters:
        signal_data (array-like): The input ECG signal.
        fs (float): Sampling frequency in Hz.
        lowcut (float): Low cutoff frequency in Hz.
        highcut (float): High cutoff frequency in Hz.
        order (int): Order of the Butterworth filter.

    Returns:
        y (ndarray): The filtered signal.
    """
    if not isinstance(signal_data, np.ndarray):
        signal_data = np.array(signal_data)
    if signal_data.ndim != 1:
        raise ValueError("signal_data must be a 1D array.")

    nyquist = 0.5 * fs
    if lowcut >= nyquist or highcut >= nyquist:
        raise ValueError("Cutoff frequencies must be less than Nyquist frequency.")
    if lowcut >= highcut:
        raise ValueError("lowcut must be less than highcut.")

    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, signal_data)
    return y


def notch_filter(signal_data, fs, freq=50.0, Q=30.0):
    """
    Apply a notch filter to remove a specific frequency from the signal.

    Parameters:
        signal_data (array-like): The input ECG signal.
        fs (float): Sampling frequency in Hz.
        freq (float): Frequency to remove in Hz.
        Q (float): Quality factor.

    Returns:
        y (ndarray): The filtered signal.
    """
    if not isinstance(signal_data, np.ndarray):
        signal_data = np.array(signal_data)
    if signal_data.ndim != 1:
        raise ValueError("signal_data must be a 1D array.")

    nyquist = 0.5 * fs
    if freq >= nyquist:
        raise ValueError("Notch frequency must be less than Nyquist frequency.")

    w0 = freq / nyquist
    b, a = signal.iirnotch(w0, Q)
    y = signal.filtfilt(b, a, signal_data)
    return y


def pan_tompkins_detector(ecg_signal, fs):
    """
    Detect R-peaks in an ECG signal using the Pan-Tompkins algorithm.

    Parameters:
        ecg_signal (array-like): The input ECG signal.
        fs (float): Sampling frequency in Hz.

    Returns:
        peaks (ndarray): Indices of detected R-peaks in the ECG signal.
    """
    if not isinstance(ecg_signal, np.ndarray):
        ecg_signal = np.array(ecg_signal)
    if ecg_signal.ndim != 1:
        raise ValueError("ecg_signal must be a 1D array.")

    # 1. Differentiate the signal
    diff = np.diff(ecg_signal)

    # 2. Square the signal to emphasize larger values
    squared = diff ** 2

    # 3. Moving window integration
    window_size = int(0.150 * fs)  # 150 ms window
    if window_size < 1:
        raise ValueError("Window size for integration must be at least 1 sample.")

    integrated = np.convolve(squared, np.ones(window_size) / window_size, mode='same')

    # 4. Thresholding
    threshold = np.mean(integrated) + 0.5 * np.std(integrated)

    # 5. Peak detection with minimum distance between peaks
    distance = int(0.2 * fs)  # Minimum distance between R-peaks
    peaks, _ = signal.find_peaks(integrated, height=threshold, distance=distance)

    return peaks


# def bandpass_filter(signal_data, fs, lowcut=1.0, highcut=100.0, order=4):
#     """
#     Bandpass filter the signal_data between lowcut and highcut frequencies.
#     """
#     nyquist = 0.5 * fs
#     low = lowcut / nyquist
#     high = highcut / nyquist
#     b, a = signal.butter(order, [low, high], btype='band')
#     return signal.filtfilt(b, a, signal_data)

# def notch_filter(signal_data, fs, freq=50.0, Q=35.0):
#     """
#     Apply a notch filter to remove specific frequency.
#     """
#     nyquist = 0.5 * fs
#     w0 = freq / nyquist
#     b, a = signal.iirnotch(w0, Q)
#     return signal.filtfilt(b, a, signal_data)

# def pan_tompkins_detector(ecg_signal, fs):
#     """
#     Detect R-peaks in an ECG signal using Pan-Tompkins algorithm.
#     """
#     diff = np.diff(ecg_signal)
#     squared = diff ** 2
#     window_size = int(0.150 * fs)  # 150 ms window
#     integrated = np.convolve(squared, np.ones(window_size) / window_size, mode='same')
#     threshold = np.mean(integrated) + 0.5 * np.std(integrated)
#     distance = int(0.2 * fs)  # Minimum distance between R-peaks
#     peaks, _ = signal.find_peaks(integrated, height=threshold, distance=distance)
#     return peaks


def detect_qrs_onset_offset(ecg_signal, r_peaks, fs):
    """
    Detect QRS onset and offset indices based on R-peaks.

    Parameters:
        ecg_signal (array-like): The input ECG signal.
        r_peaks (array-like): Indices of detected R-peaks.
        fs (float): Sampling frequency in Hz.

    Returns:
        qrs_onsets (list): Indices of QRS onset points.
        qrs_offsets (list): Indices of QRS offset points.
    """
    if not isinstance(ecg_signal, np.ndarray):
        ecg_signal = np.array(ecg_signal)
    if ecg_signal.ndim != 1:
        raise ValueError("ecg_signal must be a 1D array.")

    qrs_onsets = []
    qrs_offsets = []

    for r_peak in r_peaks:
        # Detect QRS onset
        onset = r_peak
        while onset > 0 and ecg_signal[onset] > ecg_signal[onset - 1]:
            onset -= 1
        qrs_onsets.append(max(0, onset))  # Prevent index out of bounds

        # Detect QRS offset
        offset = r_peak
        while offset < len(ecg_signal) - 1 and ecg_signal[offset] > ecg_signal[offset + 1]:
            offset += 1
        qrs_offsets.append(min(len(ecg_signal) - 1, offset))  # Prevent index out of bounds

    return qrs_onsets, qrs_offsets


def detect_p_waves(ecg_signal, qrs_onsets, fs):
    """
    Detect P-wave peaks preceding QRS onsets.

    Parameters:
        ecg_signal (array-like): The input ECG signal.
        qrs_onsets (array-like): Indices of QRS onset points.
        fs (float): Sampling frequency in Hz.

    Returns:
        p_peaks (list): Indices of detected P-wave peaks. None if not found.
    """
    if not isinstance(ecg_signal, np.ndarray):
        ecg_signal = np.array(ecg_signal)
    if ecg_signal.ndim != 1:
        raise ValueError("ecg_signal must be a 1D array.")

    p_peaks = []
    for onset in qrs_onsets:
        # Define a search region for the P-wave (200 ms before the QRS onset)
        search_start = max(0, onset - int(0.2 * fs))
        search_region = ecg_signal[search_start:onset]

        if len(search_region) == 0:
            p_peaks.append(None)
            continue

        # Find the peak in the search region
        p_peak_relative = np.argmax(search_region)
        p_peak = search_start + p_peak_relative
        p_peaks.append(p_peak)

    return p_peaks


# def detect_p_waves(ecg_signal, qrs_onsets, fs, max_p_wave_distance=0.2):
#     """
#     Detect P-wave peaks preceding QRS onsets.

#     Parameters:
#         ecg_signal (array-like): The input ECG signal.
#         qrs_onsets (array-like): Indices of QRS onset points.
#         fs (float): Sampling frequency in Hz.
#         max_p_wave_distance (float): Maximum distance (in seconds) to search for P-wave before QRS onset.

#     Returns:
#         p_peaks (list): Indices of detected P-wave peaks. None if not found.
#     """
#     p_peaks = []
#     for onset in qrs_onsets:
#         # Define a dynamic search region based on maximum P-wave distance
#         search_start = max(0, onset - int(max_p_wave_distance * fs))
#         search_region = ecg_signal[search_start:onset]

#         if len(search_region) == 0:
#             p_peaks.append(None)
#             continue

#         # Find the peak in the search region
#         p_peak_relative = np.argmax(search_region)
#         p_peak = search_start + p_peak_relative
#         p_peaks.append(p_peak)

#     return p_peaks


def calculate_intervals(p_peaks, qrs_onsets, qrs_offsets, fs):
    """
    Calculate PR intervals and QRS durations.

    Parameters:
        p_peaks (array-like): Indices of P-wave peaks.
        qrs_onsets (array-like): Indices of QRS onset points.
        qrs_offsets (array-like): Indices of QRS offset points.
        fs (float): Sampling frequency in Hz.

    Returns:
        pr_intervals (list): PR intervals in seconds. None if P-wave is missing.
        qrs_durations (list): QRS durations in seconds.
    """
    pr_intervals = []
    qrs_durations = []

    for p_peak, qrs_onset, qrs_offset in zip(p_peaks, qrs_onsets, qrs_offsets):
        # Calculate PR interval
        if p_peak is not None and p_peak < qrs_onset:
            pr_interval = (qrs_onset - p_peak) / fs  # Convert to seconds
        else:
            pr_interval = None
        pr_intervals.append(pr_interval)

        # Calculate QRS duration
        if qrs_offset > qrs_onset:
            qrs_duration = (qrs_offset - qrs_onset) / fs  # Convert to seconds
        else:
            qrs_duration = None
        qrs_durations.append(qrs_duration)

    return pr_intervals, qrs_durations


# def calculate_intervals(p_peaks, qrs_onsets, qrs_offsets, fs):
#     pr_intervals = []
#     qrs_durations = []

#     for p_peak, qrs_onset, qrs_offset in zip(p_peaks, qrs_onsets, qrs_offsets):
#         # Calculate PR interval
#         if p_peak is not None and p_peak < qrs_onset:
#             pr_interval = (qrs_onset - p_peak) / fs  # Convert to seconds
#             # Filter out unrealistic PR intervals (e.g., too short or too long)
#             if 0.08 <= pr_interval <= 0.2:  # Typical PR interval range
#                 pr_intervals.append(pr_interval)
#             else:
#                 pr_intervals.append(None)
#         else:
#             pr_intervals.append(None)

#         # Calculate QRS duration
#         if qrs_offset > qrs_onset:
#             qrs_duration = (qrs_offset - qrs_onset) / fs  # Convert to seconds
#             qrs_durations.append(qrs_duration)
#         else:
#             qrs_durations.append(None)

#     return pr_intervals, qrs_durations


def calculate_rr_intervals(r_peaks, fs):
    rr_intervals = np.diff(r_peaks) / fs  # 초 단위
    return rr_intervals


def calculate_hrv_metrics(rr_intervals):
    rr_mean = np.mean(rr_intervals) if len(rr_intervals) > 0 else None
    rr_sdnn = np.std(rr_intervals) if len(rr_intervals) > 0 else None
    rr_rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2)) if len(rr_intervals) > 1 else None
    return rr_mean, rr_sdnn, rr_rmssd


def analyze_ecg(ecg_signals, fs):
    """
    Analyze ECG signals with zero-padding handling, R-peak detection, and RR interval calculation.
    """
    analysis_results = []
    for signal_idx, ecg_signal in enumerate(ecg_signals):
        # Remove zero padding
        ecg_signal = remove_padding(ecg_signal)

        # Filter the signal
        ecg_filtered = bandpass_filter(ecg_signal, fs, lowcut=1.0, highcut=40.0, order=4)
        ecg_filtered = notch_filter(ecg_filtered, fs, freq=50.0, Q=35.0)

        # Detect R-peaks
        r_peaks = pan_tompkins_detector(ecg_filtered, fs)

        # Calculate RR intervals
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


def determine(result):
    pr_intervals = result['PR Intervals']
    qrs_durations = result['QRS Durations']
    rr_intervals = result['RR Intervals']
    p_peaks = result.get('P Peaks', [None] * len(pr_intervals))
    rr_rmssd = result['HRV Metrics']['RMSSD'] if result['HRV Metrics']['RMSSD'] is not None else 0

    av_block_threshold = 0.1225
    bundle_branch_block_threshold = 0.0350
    bradycardia_threshold = 1.2450
    tachycardia_threshold = 1.3275
    threshold_rmssd = 0.1

    av_block = any(pr is not None and pr > av_block_threshold for pr in pr_intervals)
    bundle_branch_block = any(qrs is not None and qrs >= bundle_branch_block_threshold for qrs in qrs_durations)
    bradycardia = any(rr is not None and rr > bradycardia_threshold for rr in rr_intervals)
    tachycardia = any(rr is not None and rr < tachycardia_threshold for rr in rr_intervals)
    atrial_fibrillation = (all(p is None for p in p_peaks) or rr_rmssd > threshold_rmssd) if p_peaks else False

    conditions = {
        'AV Block': av_block,
        'Bundle Branch Block': bundle_branch_block,
        'Bradycardia': bradycardia,
        'Tachycardia': tachycardia,
        'Atrial Fibrillation': atrial_fibrillation
    }

    return conditions


def categorize_ecg_condition(analysis_results):
    categories = []

    for result in analysis_results:
        conditions = determine(result)

        categories.append(conditions)

    return categories


def find_optimal_threshold(args, lead=3):
    print(f"\n------------------------------ Lead {lead} ------------------------------")
    save_analysis_results(args, lead)

    # Load data
    data = pd.read_csv('./combined_max_output.csv')

    roc_auc_mean = 0
    for i in range(0, len(data.columns) // 2):
        data = data.dropna(subset=[data.columns[2 * i], data.columns[2 * i + 1]])

        y_true = data.iloc[:, 2 * i]
        y_scores = data.iloc[:, 2 * i + 1]

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)  # False Positive Rate and True Positive Rate
        roc_auc = auc(fpr, tpr)
        roc_auc_mean += roc_auc

        # Calculate Youden's Index to find the optimal threshold
        # Youden's Index = Sensitivity + Specificity − 1 = TPR - FPR
        # Sensitivity (TPR) Specificity (1−FPR)
        youden_index = tpr - fpr
        optimal_threshold_index = np.argmax(youden_index)
        optimal_threshold = thresholds[optimal_threshold_index]

        print(f"----------{data.columns[2 * i]}----------")
        print(f"AUC: {roc_auc}")
        print(f"Optimal  Threshold: {optimal_threshold:.4f}")

        # Return the optimal threshold and AUC
    roc_auc_mean = roc_auc_mean / (len(data.columns) // 2)
    print(f"Mean AUC: {roc_auc_mean}")

    return optimal_threshold, roc_auc_mean


def save_analysis_results(args, lead=3):
    # todo: i -> n
    with h5py.File(args.path_to_hdf5, 'r') as f:
        ecg_signals = f['tracings'][:, :, lead]

    # convert signals into voltage
    fs = 400
    analysis_results = analyze_ecg(ecg_signals * 1e-4, fs)
    Y = pd.read_csv(args.path_to_csv).values

    data_rows = []

    # Loop through each item in Y and analysis_results and combine them
    for i in range(len(Y)):
        rr_intervals = analysis_results[i].get('RR Intervals', np.array([]))

        # row = {
        #     '1dAVb': Y[i][0],
        #     'RBBB': Y[i][1],
        #     'LBBB': Y[i][2],
        #     'SB': Y[i][3],
        #     'AF': Y[i][4],
        #     'ST': Y[i][5],
        #     'P Peaks (Max)': max(analysis_results[i].get('P Peaks', [np.nan])),
        #     'QRS Durations (Max)': max(analysis_results[i].get('QRS Durations', [np.nan])),
        #     'PR Intervals (Max)': max(analysis_results[i].get('PR Intervals', [np.nan])),
        #     'RMSSD': analysis_results[i].get('RMSSD', np.nan),
        #     'SDNN': analysis_results[i].get('SDNN', np.nan),
        #     'Mean RR': analysis_results[i].get('Mean RR', np.nan),
        #     'RR Intervals (Max)': max(rr_intervals) if len(rr_intervals) > 0 else np.nan,
        #     'RR Intervals (Min)': min(rr_intervals) if len(rr_intervals) > 0 else np.nan,
        # }

        bbb_value = 1 if Y[i][1] == 1 or Y[i][2] == 1 else 0
        qrs_durations = analysis_results[i].get('QRS Durations', [np.nan])
        filtered_qrs_durations = [val for val in qrs_durations if val is not None]
        qrs_max = max(filtered_qrs_durations) if filtered_qrs_durations else np.nan

        pr_intervals = analysis_results[i].get('PR Intervals', [np.nan])
        filtered_pr_intervals = [val for val in pr_intervals if val is not None]
        pr_max = max(filtered_pr_intervals) if filtered_pr_intervals else np.nan

        row = {
            '1dAVb': Y[i][0],
            # 'PR Intervals (Max)': max(analysis_results[i].get('PR Intervals', [np.nan])),
            'PR Intervals (Max)': pr_max,

            'BBB': Y[i][1] + Y[i][2],
            # 'QRS Durations (Max)': max(analysis_results[i].get('QRS Durations', [np.nan])),
            'QRS Durations (Max)': qrs_max,

            'SB': bbb_value,
            'RR Intervals (Max)': max(rr_intervals) if len(rr_intervals) > 0 else np.nan,

            'ST': Y[i][5],
            'RR Intervals (Min)': min(rr_intervals) if len(rr_intervals) > 0 else np.nan,

            'AF': Y[i][4],
            'RMSSD': analysis_results[i].get('RMSSD', np.nan),
        }

        # pr_intervals = analysis_results[i].get('PR Intervals', [np.nan])
        # pr_intervals = [float(val) for val in pr_intervals]  # np.float64 → float
        # row = {
        #     'SB': bbb_value,
        #     'ST': Y[i][5],
        #     'PR_intervals': pr_intervals
        # }

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
    # exit()

    ''' Check which lead is better '''
    # menas = []
    # for i in range(0, 12):
    #     _, mean = find_optimal_threshold(args, i)
    #     menas.append(mean)
    # for i in range(0, 12):
    #     print(f"Mean ACU at Lead {i}: {menas[i]}")

    optimal_threshold, mean = find_optimal_threshold(args)
