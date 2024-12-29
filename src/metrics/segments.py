import numpy as np
import neurokit2 as nk

"""
This module extracts key ECG segment durations (QT, QRS, PR, and ST) from a given ECG signal.

Parameters:
- ecg_signal: Flattened ECG signal as a 1D array.
- sampling_rate: Sampling rate of the ECG signal in Hz.

Returns:
- QT: Array of QT segment durations (seconds).
- QRS: Array of QRS durations (seconds).
- PR: Array of PR segment durations (seconds).
- ST: Array of ST segment durations (seconds).
"""

def extract_stats(ecg_signal, sampling_rate):
    """
    Process an ECG signal to extract durations of key segments (QT, QRS, PR, and ST).

    Arguments:
    ecg_signal -- 1D numpy array of the ECG signal.
    sampling_rate -- Sampling rate of the ECG signal in Hz.

    Returns:
    QT, QRS, PR, ST -- Arrays of segment durations in seconds.
    """

    # Process the ECG signal to clean and extract relevant features
    signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)

    # Delineate the ECG waveforms to identify key points
    df, _ = nk.ecg_delineate(
        signals["ECG_Clean"],
        info['ECG_R_Peaks'],
        sampling_rate=sampling_rate,
        method="cwt",
        show=False,
        show_type='all'
    )

    # QT Segment: Duration from Q to T offset
    Q = np.array(df['ECG_Q_Peaks'])
    T = np.array(df['ECG_T_Offsets'])
    min_QT = min(len(Q), len(T))
    Q, T = Q[:min_QT], T[:min_QT]
    QT = (T - Q) / sampling_rate
    QT = np.abs(QT[~np.isnan(QT)])  # Remove NaN values

    # QRS Duration: Duration from Q to S
    Q = np.array(df['ECG_Q_Peaks'])
    S = np.array(df['ECG_S_Peaks'])
    min_QS = min(len(Q), len(S))
    Q, S = Q[:min_QS], S[:min_QS]
    QRS = (S - Q) / sampling_rate
    QRS = np.abs(QRS[~np.isnan(QRS)])  # Remove NaN values

    # PR Segment: Duration from P offset to R onset
    P = np.array(df['ECG_P_Offsets'])
    R = np.array(df['ECG_R_Onsets'])
    min_PR = min(len(P), len(R))
    P, R = P[:min_PR], R[:min_PR]
    PR = (P - R) / sampling_rate
    PR = np.abs(PR[~np.isnan(PR)])  # Remove NaN values

    # ST Segment: Duration from R offset to T onset
    R = np.array(df['ECG_R_Offsets'])
    T = np.array(df['ECG_T_Onsets'])
    min_RT = min(len(R), len(T))
    R, T = R[:min_RT], T[:min_RT]
    ST = (T - R) / sampling_rate
    ST = np.abs(ST[~np.isnan(ST)])  # Remove NaN values

    return QT, QRS, PR, ST
