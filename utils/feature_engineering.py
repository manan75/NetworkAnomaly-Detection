"""
utils/feature_engineering.py
-----------------------------
Shared flow-based feature extraction logic.
Used identically by 03_feature_engineering.ipynb and 05_inference_pipeline.ipynb
to guarantee consistency between training and inference.
"""

import re
import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Info column parsers
# ---------------------------------------------------------------------------

def extract_src_port(info: str) -> Optional[int]:
    """Extract source port from Info string like '12345 > 80 ...'"""
    m = re.match(r'\s*(\d+)\s*>', str(info))
    return int(m.group(1)) if m else None


def extract_dst_port(info: str) -> Optional[int]:
    """Extract destination port from Info string like '12345 > 80 ...'"""
    m = re.search(r'>\s*(\d+)', str(info))
    return int(m.group(1)) if m else None


def count_flag(info: str, flag: str) -> int:
    """Return 1 if a TCP flag keyword appears in the Info string, else 0."""
    return 1 if flag in str(info) else 0


def count_http_requests(info: str) -> int:
    """Return 1 if Info contains an HTTP request method."""
    methods = ('GET ', 'POST ', 'PUT ', 'DELETE ', 'HEAD ', 'PATCH ', 'OPTIONS ')
    return 1 if any(m in str(info) for m in methods) else 0


def count_http_responses(info: str) -> int:
    """Return 1 if Info contains an HTTP response."""
    return 1 if ('HTTP/' in str(info) and ('200' in str(info) or '404' in str(info)
                  or '301' in str(info) or '500' in str(info) or 'OK' in str(info))) else 0


# ---------------------------------------------------------------------------
# 2. Packet-level preprocessing
# ---------------------------------------------------------------------------

def preprocess_packets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and enrich a raw packet DataFrame.
    Expects columns: No., Time, Source, Destination, Protocol, Length, Info
    Returns the same DataFrame with added helper columns.
    """
    df = df.copy()

    # Standardise column names
    df.columns = [c.strip().replace('.', '').replace(' ', '_').lower() for c in df.columns]

    # Rename 'no_' back to 'no' for clarity
    if 'no_' in df.columns:
        df.rename(columns={'no_': 'no'}, inplace=True)

    # Coerce types
    df['time']   = pd.to_numeric(df['time'],   errors='coerce')
    df['length'] = pd.to_numeric(df['length'], errors='coerce')
    df['info']   = df['info'].fillna('').astype(str)
    df['protocol'] = df['protocol'].fillna('UNKNOWN').astype(str).str.upper().str.strip()
    df['source']      = df['source'].fillna('0.0.0.0').astype(str).str.strip()
    df['destination'] = df['destination'].fillna('0.0.0.0').astype(str).str.strip()

    # Drop rows where critical fields are missing
    df.dropna(subset=['time', 'length'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Parse per-packet helper columns
    df['dst_port']          = df['info'].apply(extract_dst_port)
    df['syn']               = df['info'].apply(lambda x: count_flag(x, '[SYN]'))
    df['ack']               = df['info'].apply(lambda x: count_flag(x, '[ACK]'))
    df['fin']               = df['info'].apply(lambda x: count_flag(x, '[FIN]'))
    df['rst']               = df['info'].apply(lambda x: count_flag(x, '[RST]'))
    df['http_req']          = df['info'].apply(count_http_requests)
    df['http_resp']         = df['info'].apply(count_http_responses)

    return df


# ---------------------------------------------------------------------------
# 3. Flow grouping & feature extraction
# ---------------------------------------------------------------------------

TIME_WINDOW_SECONDS = 5  # group packets within this sliding window into a flow


def assign_flow_id(df: pd.DataFrame, window: float = TIME_WINDOW_SECONDS) -> pd.DataFrame:
    """
    Assign a flow_id to each packet.
    A flow is defined by (Source, Destination, Protocol) within a time window.
    Packets are sorted by time first.
    """
    df = df.sort_values('time').reset_index(drop=True)

    # Create a group key
    df['_flow_key'] = df['source'] + '|' + df['destination'] + '|' + df['protocol']

    # Assign window bucket: floor(time / window)
    df['_time_bucket'] = (df['time'] // window).astype(int)

    # Flow id = hash of key + bucket (unique integer per flow)
    df['flow_id'] = (df['_flow_key'] + '|' + df['_time_bucket'].astype(str)).apply(
        lambda x: abs(hash(x)) % (10**9)
    )

    df.drop(columns=['_flow_key', '_time_bucket'], inplace=True)
    return df


def extract_flow_features(df: pd.DataFrame, label_col: Optional[str] = None) -> pd.DataFrame:
    """
    Given a preprocessed packet DataFrame (output of preprocess_packets),
    group into flows and compute flow-level features.

    If label_col is provided (training mode), the majority label per flow is kept.
    Returns a DataFrame where each row is one flow.
    """
    df = assign_flow_id(df)

    def flow_agg(grp):
        times   = grp['time'].sort_values().values
        lengths = grp['length'].values
        iats    = np.diff(times) if len(times) > 1 else np.array([0.0])
        duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0

        total_bytes = lengths.sum()
        bps = total_bytes / duration if duration > 0 else 0.0
        pps = len(grp)  / duration if duration > 0 else 0.0

        row = {
            'packet_count':        len(grp),
            'avg_length':          float(np.mean(lengths)),
            'std_length':          float(np.std(lengths)),
            'min_length':          float(np.min(lengths)),
            'max_length':          float(np.max(lengths)),
            'avg_iat':             float(np.mean(iats)),
            'std_iat':             float(np.std(iats)),
            'duration':            duration,
            'bytes_per_second':    bps,
            'packets_per_second':  pps,
            'syn_count':           int(grp['syn'].sum()),
            'ack_count':           int(grp['ack'].sum()),
            'fin_count':           int(grp['fin'].sum()),
            'rst_count':           int(grp['rst'].sum()),
            'http_request_count':  int(grp['http_req'].sum()),
            'http_response_count': int(grp['http_resp'].sum()),
            'unique_dst_ports':    int(grp['dst_port'].nunique()),
            'protocol':            grp['protocol'].mode().iloc[0],  # dominant protocol
        }
        return pd.Series(row)

    agg = df.groupby('flow_id').apply(flow_agg).reset_index()

    # Encode protocol as integer category
    agg['protocol_encoded'] = agg['protocol'].astype('category').cat.codes
    agg.drop(columns=['protocol'], inplace=True)

    # Attach label if available (training mode)
    if label_col and label_col in df.columns:
        flow_labels = (
            df.groupby('flow_id')[label_col]
            .agg(lambda x: x.mode().iloc[0])
            .reset_index()
            .rename(columns={label_col: 'label'})
        )
        agg = agg.merge(flow_labels, on='flow_id', how='left')

    # Replace inf/nan that can arise from edge-case flows
    agg.replace([np.inf, -np.inf], 0, inplace=True)
    agg.fillna(0, inplace=True)

    return agg


# ---------------------------------------------------------------------------
# 4. Feature columns (used by model training & inference)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    'packet_count', 'avg_length', 'std_length', 'min_length', 'max_length',
    'avg_iat', 'std_iat', 'duration', 'bytes_per_second', 'packets_per_second',
    'syn_count', 'ack_count', 'fin_count', 'rst_count',
    'http_request_count', 'http_response_count',
    'unique_dst_ports', 'protocol_encoded'
]
