#!/usr/bin/env python3
"""
Run Test Tool - Browser Version (Pyodide Compatible)
Based on Reliability_Test_p1_hotfix.py
- Fast "contains 24h subwindow >= floor" checks using prefix sums on ACTIVE bins
- Alarm log filtering support
- Full parameter support matching CLI version
"""

import io
import json
import time
import re
import bisect
import gc
import heapq
import numpy as np
import pandas as pd

# Global storage for Excel generation on-demand
output_excel_bytes = None
cached_analysis_data = None  # Stores {summaries, data_sheets, alarm_sheets} for Excel generation


class MissingAirDensityError(Exception):
    def __init__(self, wtg: str, available_columns: list):
        self.wtg = wtg
        self.available_columns = available_columns
        super().__init__(f"Air density column missing for {wtg}")

# Detection candidates
TIMESTAMP_CANDIDATES = ['PCTimeStamp', 'Timestamp', 'TimeStamp', 'DateTime', 'Datetime', 'Time', 'Date']
POWER_SUFFIX_CANDIDATES = ['Power, Average', 'Grid Production Power Avg.', 'Total Active power']
WIND_SPEED_SUFFIX_CANDIDATES = ['Wind speed, Average', 'Ambient WindSpeed Avg.']
AIR_DENSITY_SUFFIX_CANDIDATES = ['Ambient Airdensity AirDensityAvg Avg']
STATE_SUFFIX = 'System States TurbineState'
CAT_SUFFIX = '1_Report Category'

# Alarm column candidates
ALARM_TS_CANDIDATES = [
    'Detected', 'Event time', 'Occured', 'Occurred', 'Alarm time', 'Start time', 'Time', 'Date', 'Datetime', 'Timestamp',
    'PCtimeStamp', 'PC timestamp', 'Device ack.'
]
ALARM_UNIT_CANDIDATES = ['Unit', 'WTG', 'Turbine', 'Turbine ID', 'Unit ID', 'Wind turbine', 'Unit name']
ALARM_EVENTTYPE_CANDIDATES = ['Event type', 'Type', 'EventType']
ALARM_SEVERITY_CANDIDATES = ['Severity', 'Level']

# Global output holder
output_excel_bytes = None


def _clean_colname(name: str) -> str:
    s = str(name).strip()
    s = re.sub(r"\s*\(\d+\)\s*$", "", s)
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def _is_number(val: str) -> bool:
    try:
        float(val)
        return True
    except Exception:
        return False


def parse_power_curve_text(text: str) -> pd.DataFrame:
    """Parse a pasted power curve table (wind speed rows, air density columns)."""
    if not text or not text.strip():
        raise ValueError('Power curve text is empty.')
    raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not raw_lines:
        raise ValueError('Power curve text is empty after trimming.')

    def _tokens(line: str) -> list:
        return [t for t in re.split(r"[\s,]+", line.strip()) if t]

    header_idx = None
    for idx, ln in enumerate(raw_lines):
        toks = _tokens(ln)
        joined = ' '.join(toks).lower()
        if 'ws' in joined and any(_is_number(t) for t in toks):
            header_idx = idx
            header_tokens = toks
            break
    if header_idx is None:
        raise ValueError('Could not locate header row containing WS and air density values.')

    num_start = next((i for i, t in enumerate(header_tokens) if _is_number(t)), None)
    if num_start is None:
        raise ValueError('Header row missing numeric air density values.')
    first_col = ' '.join(header_tokens[:num_start]).strip() or 'WS'
    density_values = [float(t) for t in header_tokens[num_start:]]
    expected_cols = 1 + len(density_values)

    rows = []
    i = header_idx + 1
    while i < len(raw_lines):
        toks = _tokens(raw_lines[i])
        if not toks:
            i += 1
            continue
        if len(toks) == 1 and _is_number(toks[0]) and i + 1 < len(raw_lines):
            nxt = _tokens(raw_lines[i + 1])
            if len(nxt) == len(density_values) and all(_is_number(t) for t in nxt):
                row = [float(toks[0])] + [float(x) for x in nxt]
                rows.append(row)
                i += 2
                continue
        if _is_number(toks[0]) and len(toks) == expected_cols and all(_is_number(t) for t in toks[1:]):
            row = [float(toks[0])] + [float(x) for x in toks[1:]]
            rows.append(row)
        i += 1

    if not rows:
        raise ValueError('No power curve rows parsed; check pasted format.')

    col_names = [first_col] + [str(d) for d in density_values]
    return pd.DataFrame(rows, columns=col_names)


def prepare_power_curve(curve_df: pd.DataFrame):
    ws_vals = pd.to_numeric(curve_df.iloc[:, 0], errors='coerce').to_numpy()
    ad_vals = np.array([float(c) for c in curve_df.columns[1:]], dtype=float)
    grid = curve_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    if ws_vals.size == 0 or ad_vals.size == 0:
        raise ValueError('Power curve must include wind speed and air density columns.')
    return ws_vals, ad_vals, grid


def interpolate_power_curve(curve_arrays, wind_speed: float, air_density: float) -> float:
    ws_vals, ad_vals, grid = curve_arrays
    if not np.isfinite(wind_speed) or not np.isfinite(air_density):
        return float('nan')
    ws_clamped = float(np.clip(wind_speed, ws_vals.min(), ws_vals.max()))
    ad_clamped = float(np.clip(air_density, ad_vals.min(), ad_vals.max()))

    i = np.searchsorted(ws_vals, ws_clamped)
    j = np.searchsorted(ad_vals, ad_clamped)
    i0 = max(i - 1, 0); i1 = min(i, len(ws_vals) - 1)
    j0 = max(j - 1, 0); j1 = min(j, len(ad_vals) - 1)
    ws0, ws1 = ws_vals[i0], ws_vals[i1]
    ad0, ad1 = ad_vals[j0], ad_vals[j1]
    q11 = grid[i0, j0]; q12 = grid[i0, j1]; q21 = grid[i1, j0]; q22 = grid[i1, j1]

    if ws1 == ws0 and ad1 == ad0:
        return float(q11)
    if ws1 == ws0:
        denom = (ad1 - ad0) if ad1 != ad0 else 1e-9
        t = (ad_clamped - ad0) / denom
        return float(q11 + (q12 - q11) * t)
    if ad1 == ad0:
        denom = (ws1 - ws0) if ws1 != ws0 else 1e-9
        t = (ws_clamped - ws0) / denom
        return float(q11 + (q21 - q11) * t)

    t = (ws_clamped - ws0) / (ws1 - ws0)
    u = (ad_clamped - ad0) / (ad1 - ad0)
    return float((1 - t) * (1 - u) * q11 + (1 - t) * u * q12 + t * (1 - u) * q21 + t * u * q22)


def compute_expected_power(ws_series: pd.Series, ad_series: pd.Series, curve_arrays) -> np.ndarray:
    """Vectorized expected power calculation using pure numpy bilinear interpolation."""
    ws_grid, ad_grid, grid = curve_arrays
    ws_vals = pd.to_numeric(ws_series, errors='coerce').to_numpy()
    ad_vals = pd.to_numeric(ad_series, errors='coerce').to_numpy()
    
    # Clamp values to grid bounds
    ws_clamped = np.clip(ws_vals, ws_grid.min(), ws_grid.max())
    ad_clamped = np.clip(ad_vals, ad_grid.min(), ad_grid.max())
    
    # Handle NaN values
    valid_mask = np.isfinite(ws_clamped) & np.isfinite(ad_clamped)
    out = np.full(len(ws_vals), np.nan, dtype=float)
    
    if not valid_mask.any():
        return out
    
    # Get valid values only
    ws_v = ws_clamped[valid_mask]
    ad_v = ad_clamped[valid_mask]
    
    # Find indices for interpolation (vectorized)
    # np.searchsorted finds where values would be inserted
    i_hi = np.searchsorted(ws_grid, ws_v).clip(1, len(ws_grid) - 1)
    j_hi = np.searchsorted(ad_grid, ad_v).clip(1, len(ad_grid) - 1)
    i_lo = i_hi - 1
    j_lo = j_hi - 1
    
    # Get grid values at corners
    ws_lo = ws_grid[i_lo]
    ws_hi = ws_grid[i_hi]
    ad_lo = ad_grid[j_lo]
    ad_hi = ad_grid[j_hi]
    
    # Get power values at 4 corners of each cell
    q11 = grid[i_lo, j_lo]  # (ws_lo, ad_lo)
    q12 = grid[i_lo, j_hi]  # (ws_lo, ad_hi)
    q21 = grid[i_hi, j_lo]  # (ws_hi, ad_lo)
    q22 = grid[i_hi, j_hi]  # (ws_hi, ad_hi)
    
    # Compute interpolation weights
    # Handle edge case where hi == lo (at grid boundaries)
    ws_denom = np.where(ws_hi != ws_lo, ws_hi - ws_lo, 1.0)
    ad_denom = np.where(ad_hi != ad_lo, ad_hi - ad_lo, 1.0)
    
    t = np.where(ws_hi != ws_lo, (ws_v - ws_lo) / ws_denom, 0.0)
    u = np.where(ad_hi != ad_lo, (ad_v - ad_lo) / ad_denom, 0.0)
    
    # Bilinear interpolation
    result = (1 - t) * (1 - u) * q11 + (1 - t) * u * q12 + t * (1 - u) * q21 + t * u * q22
    
    out[valid_mask] = result
    return out


def normalize_headers(df: pd.DataFrame, header_row_1based=None) -> pd.DataFrame:
    d = df.copy()
    if header_row_1based is not None:
        idx = max(0, header_row_1based - 1)
        header_values = d.iloc[idx].tolist()
        d = d.iloc[idx+1:].reset_index(drop=True)
        d.columns = [_clean_colname(x) for x in header_values]
    else:
        cols_clean = [_clean_colname(c) for c in d.columns]
        if any('timestamp' in c.lower() for c in cols_clean) or any(c in cols_clean for c in TIMESTAMP_CANDIDATES):
            d.columns = cols_clean
        else:
            tokens = [t.lower() for t in TIMESTAMP_CANDIDATES] + ['timestamp']
            header_idx = None
            scan_rows = min(100, len(d))
            for i in range(scan_rows):
                row = d.iloc[i].astype(str).str.strip().str.lower()
                if any(tok in row.values for tok in tokens):
                    header_idx = i
                    break
            if header_idx is not None:
                header_values = d.iloc[header_idx].tolist()
                d = d.iloc[header_idx+1:].reset_index(drop=True)
                d.columns = [_clean_colname(x) for x in header_values]
            else:
                d.columns = cols_clean
    d = d.dropna(axis=1, how='all')
    d = d.loc[:, ~d.columns.duplicated()].copy()
    return d


def detect_timestamp_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c in TIMESTAMP_CANDIDATES:
            return c
    for c in df.columns:
        if 'timestamp' in str(c).lower():
            return c
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], errors='coerce')
            if parsed.notna().mean() > 0.5:
                return c
        except Exception:
            pass
    raise ValueError('Could not find a timestamp column.')


def infer_bin_minutes(ts: pd.Series) -> float:
    ts = ts.sort_values()
    diffs = ts.diff().dropna().dt.total_seconds() / 60.0
    if diffs.empty:
        return 10.0
    return float(np.median(diffs.values))


def _parse_separators(arg: str) -> list:
    parts = [p.strip() for p in (arg or '').split(',') if p.strip()]
    out = []
    for p in parts:
        out.append(' ' if p.lower() == 'space' else p)
    if not out:
        out = ['_', ' ', '-']
    return out


def find_col_with_suffix(columns: list, wtg: str, suffixes: list, seps: list):
    for sep in seps:
        for suf in suffixes:
            name = f"{wtg}{sep}{suf}"
            if name in columns:
                return name
    for c in columns:
        for suf in suffixes:
            if c.endswith(suf):
                head = c[:-len(suf)].rstrip()
                for sep in seps:
                    if head.endswith(sep):
                        pref = head[:-len(sep)].rstrip()
                        if pref == wtg:
                            return c
    return None


def extract_wtgs_flexible(columns: list, seps: list) -> list:
    meta = {}
    for c in columns:
        for suf in POWER_SUFFIX_CANDIDATES:
            if c.endswith(suf):
                head = c[:-len(suf)].rstrip()
                for sep in seps:
                    if head.endswith(sep):
                        pref = head[:-len(sep)].rstrip()
                        if pref:
                            meta.setdefault(pref, {'power': False, 'state': False, 'cat': False})
                            meta[pref]['power'] = True
                        break
        if c.endswith(STATE_SUFFIX):
            head = c[:-len(STATE_SUFFIX)].rstrip()
            for sep in seps:
                if head.endswith(sep):
                    pref = head[:-len(sep)].rstrip()
                    if pref:
                        meta.setdefault(pref, {'power': False, 'state': False, 'cat': False})
                        meta[pref]['state'] = True
        if c.endswith(CAT_SUFFIX):
            head = c[:-len(CAT_SUFFIX)].rstrip()
            for sep in seps:
                if head.endswith(sep):
                    pref = head[:-len(sep)].rstrip()
                    if pref:
                        meta.setdefault(pref, {'power': False, 'state': False, 'cat': False})
                        meta[pref]['cat'] = True
    return sorted([p for p, flags in meta.items() if flags['power'] and flags['state'] and flags['cat']])


def _normalize_category_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.strip()
    s2 = s2.str.replace("'", "'", regex=False).str.replace("'", "'", regex=False)
    s2 = s2.str.replace('\xa0', ' ', regex=False)
    return s2.str.lower()


# ---------------- Alarm helpers ----------------

def detect_alarm_columns(df: pd.DataFrame, ts_hint=None, unit_hint=None, type_hint=None, severity_hint=None):
    """Auto-detect alarm log columns."""
    cols = list(df.columns)
    
    ts_col = ts_hint if (ts_hint and ts_hint in cols) else next((c for c in cols if str(c).strip() in ALARM_TS_CANDIDATES), None)
    if ts_col is None:
        for c in cols:
            try:
                parsed = pd.to_datetime(df[c], errors='coerce')
                if parsed.notna().mean() > 0.5:
                    ts_col = c
                    break
            except Exception:
                pass
    
    unit_col = unit_hint if (unit_hint and unit_hint in cols) else next((c for c in cols if str(c).strip() in ALARM_UNIT_CANDIDATES), None)
    type_col = type_hint if (type_hint and type_hint in cols) else next((c for c in cols if str(c).strip() in ALARM_EVENTTYPE_CANDIDATES), None)
    sev_col = severity_hint if (severity_hint and severity_hint in cols) else next((c for c in cols if str(c).strip() in ALARM_SEVERITY_CANDIDATES), None)
    
    return ts_col, unit_col, type_col, sev_col


def preprocess_alarm_df(a_df: pd.DataFrame, ts_col: str, unit_col, type_col) -> pd.DataFrame:
    """Preprocess alarm dataframe for filtering."""
    d = a_df.copy()
    if ts_col not in d.columns:
        raise KeyError(f"Timestamp column '{ts_col}' not present in alarm log.")
    
    d['_ts_'] = pd.to_datetime(d[ts_col], errors='coerce')
    
    if unit_col and unit_col in d.columns:
        d['_unit_norm'] = (d[unit_col].astype(str)
                          .str.replace('\xa0', ' ', regex=False)
                          .str.strip().str.upper())
    
    if type_col and type_col in d.columns:
        d['_etype_norm'] = d[type_col].astype(str).str.strip().str.lower()
    
    return d.sort_values('_ts_').reset_index(drop=True)


def filter_alarm_log_for_window(alarm_df: pd.DataFrame, unit_value: str, win_start, win_end,
                                 ts_col: str, unit_col, type_col, keep_event_types,
                                 sev_col, min_severity) -> pd.DataFrame:
    """Filter alarm log for a specific WTG window."""
    d = alarm_df
    mask = (d['_ts_'] >= win_start) & (d['_ts_'] <= win_end)
    
    if unit_col and '_unit_norm' in d.columns:
        mask &= (d['_unit_norm'] == str(unit_value).replace('\xa0', ' ').strip().upper())
    
    if type_col and keep_event_types and '_etype_norm' in d.columns:
        allowed = {s.strip().lower() for s in keep_event_types}
        mask &= d['_etype_norm'].apply(lambda x: any(tok in x for tok in allowed))
    
    out = d.loc[mask].copy()
    
    if sev_col and min_severity is not None and sev_col in out.columns:
        sev = pd.to_numeric(out[sev_col], errors='coerce')
        out = out.loc[sev >= min_severity]
    
    if ts_col not in out.columns:
        out[ts_col] = out['_ts_']
    
    return out.drop(columns=['_ts_', '_unit_norm', '_etype_norm'], errors='ignore')


def _evaluate_and_summarize(d: pd.DataFrame, ts_col: str, wtg: str,
                            power_col: str, state_col: str, cat_col: str,
                            s_idx: int, e_idx: int,
                            rated_power_kw: float, bin_minutes: float,
                            nominal_threshold_pct: float,
                            blocked_set: set, active_base_set: set,
                            energy_power_col: str,
                            pr_threshold: float,
                            air_density_source: str,
                            air_density_missing_count: int = 0) -> tuple:
    """Evaluate a window and build summary."""
    slice_df = d.iloc[s_idx:e_idx+1].copy()
    cats_norm = _normalize_category_series(slice_df[cat_col].astype(str))
    
    paused_mask = (~cats_norm.isin(blocked_set) & (~cats_norm.isin(active_base_set))).to_numpy()
    active_mask = ~paused_mask
    unavail_mask = cats_norm.isin(blocked_set).to_numpy()
    active_idx = np.where(active_mask)[0]
    
    availability_pct = (100.0 * np.mean(~unavail_mask[active_idx])) if len(active_idx) else float('nan')
    
    if '_NominalFlag' in slice_df.columns:
        nominal_mask = slice_df['_NominalFlag'].to_numpy().astype(bool)
    else:
        power = pd.to_numeric(slice_df[power_col], errors='coerce').to_numpy()
        threshold_kw = rated_power_kw * (nominal_threshold_pct / 100.0)
        nominal_mask = np.isfinite(power) & (power >= threshold_kw)
    nominal_valid = nominal_mask.copy()
    nominal_valid[~active_mask] = False
    total_nominal_hours = nominal_valid.sum() * (bin_minutes / 60.0)
    
    energy_series = pd.to_numeric(slice_df[energy_power_col], errors='coerce').to_numpy()
    total_energy_kwh = float(np.nansum(energy_series[active_idx]) * (bin_minutes / 60.0))
    
    if len(active_idx):
        avail_valid = (~unavail_mask[active_idx]).astype(int)
        inv = 1 - avail_valid
        interruptions = int(inv[0] == 1) + int(np.sum((inv[1:] == 1) & (inv[:-1] == 0)))
    else:
        interruptions = 0
    
    test_start_dt = pd.to_datetime(slice_df[ts_col].iloc[0])
    # Add bin_minutes to the last bin's timestamp to get the actual end time
    test_end_dt = pd.to_datetime(slice_df[ts_col].iloc[-1]) + pd.Timedelta(minutes=bin_minutes)
    test_start_str = test_start_dt.strftime('%Y-%m-%d %H:%M')
    test_end_str = test_end_dt.strftime('%Y-%m-%d %H:%M')
    active_hours = len(active_idx) * bin_minutes / 60.0
    span_hours = (e_idx - s_idx + 1) * bin_minutes / 60.0
    paused_hours = int((~active_mask).sum()) * bin_minutes / 60.0
    
    avg_pr = float(np.nanmean(slice_df['_PerformanceRatio'])) if '_PerformanceRatio' in slice_df.columns else float('nan')

    # Filter out nan/empty from categories list
    cats_list = [c for c in sorted(cats_norm.unique().tolist()) if c and c.lower() != 'nan']
    
    # Find the first and last nominal power bins in the entire window
    nominal_active_mask = nominal_valid  # Already filtered to active bins only
    nominal_indices = np.where(nominal_active_mask)[0]
    
    nominal_start_str = ''
    nominal_end_str = ''
    if len(nominal_indices) > 0:
        # First nominal bin
        first_nominal_idx = nominal_indices[0]
        nominal_start_str = pd.to_datetime(slice_df[ts_col].iloc[first_nominal_idx]).strftime('%Y-%m-%d %H:%M')
        # Last nominal bin - add bin_minutes to get the actual end of that bin
        last_nominal_idx = nominal_indices[-1]
        nominal_end_dt = pd.to_datetime(slice_df[ts_col].iloc[last_nominal_idx]) + pd.Timedelta(minutes=bin_minutes)
        nominal_end_str = nominal_end_dt.strftime('%Y-%m-%d %H:%M')
    
    # Convert energy to MWh
    total_energy_mwh = total_energy_kwh / 1000.0

    # Build summary with specified column order
    # User-specified columns first, then remaining columns
    
    # Build notes list
    notes = []
    if air_density_missing_count > 0:
        notes.append(f"Air density missing for {air_density_missing_count} slots - used 90% rated power threshold for nominal power check")
    notes_str = '; '.join(notes)
    
    summary = {
        'WTG': wtg,
        'Availability (%)': round(availability_pct, 2),
        'Date of Test Start': test_start_dt.strftime('%Y-%m-%d'),
        'Time of Test Start': test_start_dt.strftime('%H:%M'),
        'Date of Test End': test_end_dt.strftime('%Y-%m-%d'),
        'Time of Test End': test_end_dt.strftime('%H:%M'),
        'Cumulative Time of Testing (h)': round(span_hours, 2),
        # Events in Window and Event Details will be added later during alarm processing
        'First Nominal Power': nominal_start_str,
        'Last Nominal Power': nominal_end_str,
        'Cumulative Nominal Hours (h)': round(total_nominal_hours, 2),
        'Total Energy in Window (MWh)': round(total_energy_mwh, 2),
        # Remaining columns
        'Rated Power (kW)': rated_power_kw,
        'Bin Size (min)': bin_minutes,
        'Active Hours (h)': round(active_hours, 2),
        'Paused Hours (h)': round(paused_hours, 2),
        'Interruptions (count)': interruptions,
        'Test Start': test_start_str,
        'Test End': test_end_str,
        'Power Column Used': power_col,
        'Energy Power Column Used': energy_power_col,
        'State Column': state_col,
        'Category Column': cat_col,
        'Window Categories Seen': ','.join(cats_list),
        'Average Performance Ratio': round(avg_pr, 3) if np.isfinite(avg_pr) else float('nan'),
        'PR Threshold Used': pr_threshold,
        'Air Density Source': air_density_source,
        'Notes': notes_str,
    }
    return slice_df, summary


def process_wtg_fast(d: pd.DataFrame, ts_col: str, wtg: str,
                     rated_power_kw: float, bin_minutes: float,
                     test_hours: int, extension_hours: int,
                     min_availability_pct: float, nominal_threshold_pct: float,
                     allowed_categories: list, disallowed_categories: list,
                     active_base_categories: list, energy_source: str,
                     require_nominal: bool,
                     require_energy: bool,
                     energy_threshold_mwh: float,
                     allowed_window_categories: list,
                     disqualifying_window_categories: list,
                     power_curve_arrays,
                     pr_threshold: float,
                     wind_col_hint: str = None,
                     air_density_col_hint: str = None) -> dict:
    """Process a single WTG using optimized algorithm from hotfix."""
    
    def log_proc(msg):
        print(f"[Python] [{wtg}] {msg}")
    
    # Resolve columns (minimal logging - only log errors)
    seps = ['_', ' ', '-']
    power_col = find_col_with_suffix(d.columns.tolist(), wtg, POWER_SUFFIX_CANDIDATES, seps)
    if power_col is None:
        log_proc(f"FAILED: No power column found")
        return {'wtg': wtg, 'summary': {'WTG': wtg, 'Status': 'FAILED',
                                        'Reason': f'No power column found for {wtg}'}, 'data_slice': d.head(0)}
    
    total_active_col = find_col_with_suffix(d.columns.tolist(), wtg, ['Total Active power'], seps)
    energy_power_col = total_active_col if (energy_source == 'total_active' and total_active_col) else power_col
    state_col = find_col_with_suffix(d.columns.tolist(), wtg, [STATE_SUFFIX], seps) or f"{wtg}_{STATE_SUFFIX}"
    cat_col = find_col_with_suffix(d.columns.tolist(), wtg, [CAT_SUFFIX], seps) or f"{wtg}_{CAT_SUFFIX}"
    
    if cat_col not in d.columns:
        log_proc(f"FAILED: No category column found")
        return {'wtg': wtg, 'summary': {'WTG': wtg, 'Status': 'FAILED',
                                        'Reason': f'No category column found for {wtg}'}, 'data_slice': d.head(0)}

    wind_col = wind_col_hint if (wind_col_hint and wind_col_hint in d.columns) else find_col_with_suffix(d.columns.tolist(), wtg, WIND_SPEED_SUFFIX_CANDIDATES, seps)
    if wind_col is None:
        log_proc(f"FAILED: No wind speed column found")
        return {'wtg': wtg, 'summary': {'WTG': wtg, 'Status': 'FAILED',
                                        'Reason': f'No wind speed column found for {wtg}'}, 'data_slice': d.head(0)}

    # Handle air density - if missing, use 90% rated power for those slots and track for notes
    air_col = air_density_col_hint if (air_density_col_hint and air_density_col_hint in d.columns) else find_col_with_suffix(d.columns.tolist(), wtg, AIR_DENSITY_SUFFIX_CANDIDATES, seps)
    air_density_source = 'column'
    air_density_missing_count = 0
    if air_col is None:
        # No air density column at all - all rows will use 90% fallback for nominal calculation
        air_density_source = 'missing_all'
        air_series = pd.Series([np.nan] * len(d), index=d.index)
        air_density_missing_count = len(d)
    else:
        air_series = pd.to_numeric(d[air_col], errors='coerce')
        # Track missing values within the column
        air_density_missing_count = int(air_series.isna().sum())
        if air_density_missing_count > 0:
            air_density_source = f'column_with_{air_density_missing_count}_missing'

    wind_series = pd.to_numeric(d[wind_col], errors='coerce')
    expected_power_raw = compute_expected_power(wind_series, air_series, power_curve_arrays)
    # Apply nominal threshold percentage to expected power
    expected_power = expected_power_raw * (nominal_threshold_pct / 100.0)
    
    # Track which rows have missing air density for using 90% rated power fallback
    air_density_missing_mask = air_series.isna().to_numpy()
    
    # Normalize sets & arrays
    cats_norm_full = _normalize_category_series(d[cat_col].astype(str))
    blocked_set = {s.strip().lower() for s in disallowed_categories}
    active_base_set = {s.strip().lower() for s in active_base_categories}
    allow_window_set = {s.strip().lower() for s in (allowed_window_categories or [])}
    disq_window_set = {s.strip().lower() for s in (disqualifying_window_categories or [])}
    
    unavail_full = cats_norm_full.isin(blocked_set).to_numpy()
    paused_full = (~cats_norm_full.isin(blocked_set) & (~cats_norm_full.isin(active_base_set))).to_numpy()
    active_full = ~paused_full
    
    # Precompute prefix sums and performance ratio based nominal flags
    pwr = pd.to_numeric(d[power_col], errors='coerce').to_numpy()
    perf_ratio = np.divide(pwr, expected_power, out=np.full_like(expected_power, np.nan), where=np.isfinite(expected_power))
    
    # For rows with air density, use PR-based nominal; for missing air density, use 90% rated power
    threshold_kw_normal = rated_power_kw * (nominal_threshold_pct / 100.0)
    threshold_kw_missing_air = rated_power_kw * 0.90  # 90% rated power for missing air density
    
    # Normal nominal check (PR-based with fallback to threshold)
    normal_nominal = np.where(np.isfinite(perf_ratio), perf_ratio >= pr_threshold, np.isfinite(pwr) & (pwr >= threshold_kw_normal))
    # Missing air density uses 90% rated power threshold
    missing_air_nominal = np.isfinite(pwr) & (pwr >= threshold_kw_missing_air)
    
    # Combine: use missing_air logic where air density is missing, otherwise use normal logic
    nominal_full = np.where(air_density_missing_mask, missing_air_nominal, normal_nominal)
    
    # Calculate data summary for final log (but don't log here - reduce noise)
    total_active_bins = int(active_full.sum())
    total_active_hours = total_active_bins * bin_minutes / 60.0
    
    d['_WindSpeed'] = wind_series
    d['_AirDensityUsed'] = air_series
    d['_ExpectedPower'] = expected_power
    d['_PerformanceRatio'] = perf_ratio
    d['_NominalFlag'] = nominal_full
    d['_AirDensityMissing'] = air_density_missing_mask

    energy_full_kw = pd.to_numeric(d[energy_power_col], errors='coerce').fillna(0.0).to_numpy()
    e_kwh_per_row = energy_full_kw * (bin_minutes / 60.0)
    
    ps_active = np.cumsum(active_full.astype(np.int32))
    ps_nominal_active = np.cumsum((nominal_full & active_full).astype(np.int32))
    ps_energy_active = np.cumsum((e_kwh_per_row * active_full).astype(np.float64))
    ps_unavail_active = np.cumsum((unavail_full & active_full).astype(np.int32))
    
    # Bad mask: bins that are disqualifying or not in allowed window categories
    bad_mask = (cats_norm_full.isin(disq_window_set) | ~cats_norm_full.isin(allow_window_set)).to_numpy()
    ps_bad = np.cumsum(bad_mask.astype(np.int32))
    
    def span_sum(ps, s, e):
        return ps[e] - (ps[s-1] if s > 0 else 0)
    
    # Track the best "near-miss" candidate across all target sizes for failure diagnostics
    best_near_miss = None
    
    def best_by_priority_active(target_active_bins: int, req_mwh_1x: float, req_mwh_3x: float, log_func=None):
        """
        Find the BEST window with exactly target_active_bins active samples.
        Evaluates ALL valid candidates and returns the one with:
        1. Best PR tier (3x > 1x > none)
        2. Shortest wall-clock duration (fewer total bins)
        3. Earliest start time (as tiebreaker)
        """
        nonlocal best_near_miss
        N = len(d)
        s = 0
        e = -1
        active_acc = 0
        
        # Collect candidates - use list and sort at end (simpler, more reliable)
        MAX_CANDIDATES = 200  # Increased to find all potential 0-alarm windows
        all_candidates = []
        candidates_found = 0
        
        # Track rejected candidates for "near-miss" diagnostics
        near_miss_candidates = []
        
        # Use smaller step size to find more candidates
        step = max(1, int(2 * 60 / bin_minutes))  # 2-hour steps
        
        while s < N:
            while e + 1 < N and active_acc < target_active_bins:
                e += 1
                if active_full[e]:
                    active_acc += 1
            
            if active_acc >= target_active_bins:
                has_bad = span_sum(ps_bad, s, e) > 0
                active_bins = span_sum(ps_active, s, e)
                unavail_count = span_sum(ps_unavail_active, s, e)
                availability_pct = 100.0 * ((active_bins - unavail_count) / max(1, active_bins))
                nominal_hours = span_sum(ps_nominal_active, s, e) * (bin_minutes / 60.0)
                energy_kwh = span_sum(ps_energy_active, s, e)
                has_3x = energy_kwh >= req_mwh_3x
                has_1x = energy_kwh >= req_mwh_1x
                wall_clock_bins = e - s + 1
                
                # Track why this window was rejected (for near-miss diagnostics)
                rejection_reasons = []
                if has_bad:
                    rejection_reasons.append('Contains disqualifying categories')
                if availability_pct + 1e-9 < min_availability_pct:
                    rejection_reasons.append(f'Availability {availability_pct:.1f}% < {min_availability_pct}% required')
                if nominal_hours < 24.0:
                    rejection_reasons.append(f'Nominal {nominal_hours:.1f}h < 24h required')
                
                # Build candidate dict for both valid and near-miss tracking
                cand = {
                    'start': int(s), 'end': int(e),
                    'active_bins': int(active_bins),
                    'wall_clock_bins': int(wall_clock_bins),
                    'availability_pct': float(availability_pct),
                    'nominal_hours': float(nominal_hours),
                    'energy_kwh': float(energy_kwh),
                    'interruptions': 1 if unavail_count > 0 else 0,
                    'contains_3x': bool(has_3x),
                    'contains_1x': bool(has_1x),
                }
                
                if not has_bad and availability_pct + 1e-9 >= min_availability_pct:
                    # Valid window (may still fail nominal check later)
                    candidates_found += 1
                    
                    # Score tuple: (meets_nominal, pr_tier, wall_clock, start) - LOWER is BETTER
                    meets_nom = 0 if nominal_hours >= 24.0 else 1
                    pr_tier = 0 if has_3x else (1 if has_1x else 2)
                    score = (meets_nom, pr_tier, wall_clock_bins, s)
                    cand['_score'] = score
                    
                    all_candidates.append(cand)
                    
                    # Cap list size periodically
                    if len(all_candidates) > MAX_CANDIDATES * 2:
                        all_candidates.sort(key=lambda c: c['_score'])
                        all_candidates = all_candidates[:MAX_CANDIDATES]
                
                # Track near-miss regardless (for failure diagnostics)
                if rejection_reasons or nominal_hours < 24.0:
                    cand['rejection_reasons'] = rejection_reasons if rejection_reasons else [f'Nominal {nominal_hours:.1f}h < 24h required']
                    # Score near-misses: prefer highest nominal hours, then highest availability
                    near_miss_score = (-nominal_hours, -availability_pct, wall_clock_bins)
                    cand['_near_miss_score'] = near_miss_score
                    near_miss_candidates.append(cand)
                    if len(near_miss_candidates) > 20:
                        near_miss_candidates.sort(key=lambda c: c['_near_miss_score'])
                        near_miss_candidates = near_miss_candidates[:10]
            
            # Skip ahead by step, but update active_acc properly
            skip = min(step, N - s)
            for _ in range(skip):
                if s < N and active_full[s]:
                    active_acc -= 1
                s += 1
                if e < s - 1:
                    e = s - 1
                if s >= N:
                    break
        
        # Update global best near-miss if this target had better ones
        if near_miss_candidates:
            near_miss_candidates.sort(key=lambda c: c['_near_miss_score'])
            best_nm = near_miss_candidates[0]
            if best_near_miss is None or best_nm['_near_miss_score'] < best_near_miss['_near_miss_score']:
                best_near_miss = best_nm
        
        if not all_candidates:
            if log_func:
                log_func(f"    target={target_active_bins} bins: No valid candidates found")
            # Still return near-miss candidates for the explorer
            if near_miss_candidates:
                return {'valid': [], 'near_miss': near_miss_candidates}
            return None
        
        # Sort by score and return best
        MIN_NOMINAL = 24.0
        def score(c):
            meets_nominal = 0 if c['nominal_hours'] >= MIN_NOMINAL else 1
            pr_tier = 0 if c['contains_3x'] else (1 if c['contains_1x'] else 2)
            return (meets_nominal, pr_tier, c['wall_clock_bins'], c['start'])
        
        all_candidates.sort(key=score)
        best = all_candidates[0]
        
        # Don't log individual target searches - too verbose
        
        # Return ALL valid candidates (sorted), plus near-miss candidates for explorer
        return {'valid': all_candidates, 'near_miss': near_miss_candidates}
    
    # Targets and floors
    target72 = int(round(test_hours * 60 / bin_minutes))
    max_target = int(round((test_hours + extension_hours) * 60 / bin_minutes)) if extension_hours > 0 else target72
    req_mwh_1x = 0.5 * (rated_power_kw / 1000.0) * 24.0  # Energy floor in MWh
    req_mwh_3x = 3.0 * req_mwh_1x
    
    MIN_NOMINAL_HOURS_REQUIRED = 24.0
    
    def log_wtg(msg):
        print(f"[Python] [{wtg}] {msg}")
    
    # Search ALL target sizes and collect best candidates from each
    # Don't break early - we want to find the globally best window
    step_bins = int(round(2 * 60 / bin_minutes))  # 2-hour increments
    MAX_VALID_CANDIDATES = 100  # Keep enough candidates for alarm filtering
    MAX_EXPLORER_CANDIDATES = 50  # Keep more for Window Explorer (includes non-viable)
    all_valid_candidates = []
    all_explorer_candidates = []  # For Window Explorer - includes non-viable windows
    
    all_near_miss_candidates = []  # Track all near-miss candidates for explorer
    
    for target_bins in range(target72, max_target + 1, step_bins):
        result = best_by_priority_active(target_bins, req_mwh_1x, req_mwh_3x, log_func=None)  # No per-target logging
        if result is None:
            continue
        
        valid_cands = result.get('valid', [])
        near_miss_cands = result.get('near_miss', [])
        
        # Process valid candidates
        for cand in valid_cands:
            # Add timestamp info for all candidates
            cand['test_start'] = d.iloc[cand['start']][ts_col]
            cand['test_end'] = d.iloc[cand['end']][ts_col]
            cand['target_bins'] = target_bins
            
            # Determine viability and issues
            issues = []
            if cand['nominal_hours'] < MIN_NOMINAL_HOURS_REQUIRED:
                issues.append(f"Nominal {cand['nominal_hours']:.1f}h < 24h required")
            if cand['availability_pct'] < min_availability_pct:
                issues.append(f"Availability {cand['availability_pct']:.1f}% < {min_availability_pct}% required")
            cand['viable'] = len(issues) == 0
            cand['issues'] = issues
            
            # Add to explorer list (all candidates)
            all_explorer_candidates.append(cand)
            
            # Add to valid list only if meets requirements
            if cand['nominal_hours'] >= MIN_NOMINAL_HOURS_REQUIRED:
                all_valid_candidates.append(cand)
        
        # Also track near-miss candidates for explorer (even when they failed due to bad categories)
        for cand in near_miss_cands:
            if 'test_start' not in cand:
                cand['test_start'] = d.iloc[cand['start']][ts_col]
                cand['test_end'] = d.iloc[cand['end']][ts_col]
                cand['target_bins'] = target_bins
            # Mark as non-viable and include rejection reasons
            cand['viable'] = False
            cand['issues'] = cand.get('rejection_reasons', ['Unknown rejection reason'])
            all_near_miss_candidates.append(cand)
    
    # Sort all candidates by score and keep top MAX_VALID_CANDIDATES
    def final_score(c):
        meets_nominal = 0 if c['nominal_hours'] >= MIN_NOMINAL_HOURS_REQUIRED else 1
        pr_tier = 0 if c['contains_3x'] else (1 if c['contains_1x'] else 2)
        return (meets_nominal, pr_tier, c['wall_clock_bins'], c['start'])
    
    all_valid_candidates.sort(key=final_score)
    valid_candidates = all_valid_candidates[:MAX_VALID_CANDIDATES]
    
    # Sort explorer candidates: viable first, then by score
    def explorer_score(c):
        viable_priority = 0 if c.get('viable', False) else 1
        meets_nominal = 0 if c['nominal_hours'] >= MIN_NOMINAL_HOURS_REQUIRED else 1
        pr_tier = 0 if c['contains_3x'] else (1 if c['contains_1x'] else 2)
        return (viable_priority, meets_nominal, pr_tier, c['wall_clock_bins'], c['start'])
    
    # If no explorer candidates from valid windows, use near-miss candidates
    if not all_explorer_candidates and all_near_miss_candidates:
        # Add near-miss candidates to explorer so users can see why they failed
        all_explorer_candidates = all_near_miss_candidates
    
    all_explorer_candidates.sort(key=explorer_score)
    explorer_candidates = all_explorer_candidates[:MAX_EXPLORER_CANDIDATES]
    
    # If no window meets requirement, try the max extension as fallback
    fallback = None
    if not valid_candidates:
        result = best_by_priority_active(max_target, req_mwh_1x, req_mwh_3x, log_func=None)
        if result and result.get('valid'):
            cand = result['valid'][0]
            cand['test_start'] = d.iloc[cand['start']][ts_col]
            cand['test_end'] = d.iloc[cand['end']][ts_col]
            fallback = cand
            log_wtg(f"FALLBACK: Using window with only {cand['nominal_hours']:.1f}h nominal (< 24h required)")
        else:
            # Log near-miss info for debugging with detailed rejection reasons
            if best_near_miss:
                nm = best_near_miss
                rejection = ', '.join(nm.get('rejection_reasons', ['unknown']))
                log_wtg(f"FAILED: No valid window. Best candidate had nominal={nm['nominal_hours']:.1f}h/24h, avail={nm['availability_pct']:.1f}%, rejected due to: {rejection}")

    # Return candidates for alarm-based selection in main function
    # Diagnostic info for failures
    def build_diagnostic_info(best_cand=None, failure_reason=None):
        cats_unique = [c for c in sorted(cats_norm_full.unique().tolist()) if c and c.lower() != 'nan']
        total_rows = len(d)
        total_active = int(active_full.sum())
        total_paused = int(paused_full.sum())
        total_unavail = int(unavail_full.sum())
        total_hours = total_rows * bin_minutes / 60.0
        active_hours = total_active * bin_minutes / 60.0
        paused_hours = total_paused * bin_minutes / 60.0
        unavail_hours = total_unavail * bin_minutes / 60.0
        
        # Calculate why we failed
        required_active_bins = target72
        required_active_hours = required_active_bins * bin_minutes / 60.0
        
        # Use near-miss if no best_cand provided
        near_miss = best_near_miss if best_cand is None else best_cand
        
        # Determine specific failure reason
        if failure_reason is None:
            if total_active < required_active_bins:
                failure_reason = f"Not enough active bins: have {total_active} ({active_hours:.1f}h), need {required_active_bins} ({required_active_hours:.1f}h)"
            elif near_miss and near_miss.get('nominal_hours', 0) < MIN_NOMINAL_HOURS_REQUIRED:
                failure_reason = f"Insufficient nominal power hours: best candidate has {near_miss.get('nominal_hours', 0):.1f}h, need {MIN_NOMINAL_HOURS_REQUIRED}h"
            elif near_miss and near_miss.get('availability_pct', 0) < min_availability_pct:
                failure_reason = f"Availability too low: best candidate has {near_miss.get('availability_pct', 0):.1f}%, need {min_availability_pct}%"
            else:
                failure_reason = "No contiguous window of sufficient quality found"
        
        diag = {
            'WTG': wtg,
            'Status': 'FAILED',
            'Failure Reason': failure_reason,
            'Total Rows': total_rows,
            'Total Hours of Data': round(total_hours, 1),
            'Active Bins': total_active,
            'Active Hours': round(active_hours, 1),
            'Paused Hours': round(paused_hours, 1),
            'Unavailable Hours': round(unavail_hours, 1),
            'Required Active Hours': required_active_hours,
            'Categories Found': ', '.join(cats_unique[:10]) + ('...' if len(cats_unique) > 10 else ''),
            'Category Breakdown': ', '.join([f"{cat}: {count}" for cat, count in cats_norm_full.value_counts().head(5).items()]),
            'Allowed Window Categories': ', '.join(allowed_window_categories),
            'Disqualifying Categories': ', '.join(disqualifying_window_categories),
        }
        # Add near-miss details
        if near_miss:
            diag['Nearest Candidate Nominal Hours'] = round(near_miss.get('nominal_hours', 0), 2)
            diag['Nearest Candidate Availability (%)'] = round(near_miss.get('availability_pct', 0), 2)
            diag['Nearest Candidate Active Hours'] = round(near_miss.get('active_bins', 0) * bin_minutes / 60.0, 2)
            diag['Nearest Candidate Wall-Clock Hours'] = round(near_miss.get('wall_clock_bins', 0) * bin_minutes / 60.0, 2)
            diag['Nearest Candidate Energy (kWh)'] = round(near_miss.get('energy_kwh', 0), 1)
            diag['Nearest Candidate Rejection Reasons'] = ', '.join(near_miss.get('rejection_reasons', ['unknown']))
        return diag
    
    def finalize_result(chosen):
        """Build final result from chosen candidate."""
        s_idx, e_idx = chosen['start'], chosen['end']
        slice_df, summ = _evaluate_and_summarize(d, ts_col, wtg, power_col, state_col, cat_col,
                                                 s_idx, e_idx, rated_power_kw, bin_minutes,
                                                 nominal_threshold_pct, blocked_set, active_base_set,
                                                 energy_power_col, pr_threshold, air_density_source,
                                                 air_density_missing_count)
        
        # Calculate end timestamp for exactly target72 active bins (for alarm counting)
        # This ensures alarms are only counted for the required test hours, not extensions
        active_count = 0
        test_end_72h_idx = e_idx  # Default to full window end
        for idx in range(s_idx, e_idx + 1):
            if active_full[idx]:
                active_count += 1
                if active_count >= target72:
                    test_end_72h_idx = idx
                    break
        test_end_72h_str = pd.to_datetime(d.iloc[test_end_72h_idx][ts_col]).strftime('%Y-%m-%d %H:%M')
        summ['Test End (72h Active)'] = test_end_72h_str
        
        # Determine status based on nominal hours and energy
        if not require_nominal or chosen['nominal_hours'] >= MIN_NOMINAL_HOURS_REQUIRED:
            if chosen['contains_3x']:
                status = 'PASSED (3x floor achieved)'
            elif chosen['contains_1x']:
                status = 'PASSED (1x floor achieved)'
            else:
                status = 'PASSED'
            summ.update({'Status': status,
                         'Nominal 24h Achieved': bool(chosen['contains_3x'] or chosen['contains_1x']),
                         'Contains 24h subwindow >= 3x floor': bool(chosen['contains_3x']),
                         'Contains 24h subwindow >= 1x floor': bool(chosen['contains_1x'])})
        else:
            summ.update({'Status': 'completed_due_to_climatic_conditions', 'Nominal 24h Achieved': False})
        
        full_df = d.copy()
        full_df['_InWindow'] = False
        full_df.iloc[s_idx:e_idx+1, full_df.columns.get_loc('_InWindow')] = True
        return {'wtg': wtg, 'summary': summ, 'data_slice': full_df, 'window_start': s_idx, 'window_end': e_idx}
    
    # If we have valid candidates, return them for alarm-based selection
    if valid_candidates:
        return {
            'wtg': wtg,
            'candidates': valid_candidates,
            'explorer_candidates': explorer_candidates,  # All candidates for Window Explorer
            'finalize': finalize_result,
            'full_data': d,
            'ts_col': ts_col
        }
    
    # No valid candidates - use fallback or fail
    if fallback is None:
        # Determine why we failed
        total_active = int(active_full.sum())
        required_active_bins = target72
        
        if total_active < required_active_bins:
            reason = f"Insufficient active time: only {total_active * bin_minutes / 60.0:.1f}h active, need {required_active_bins * bin_minutes / 60.0:.1f}h"
        else:
            reason = "No contiguous window meeting all requirements found (check nominal power threshold and availability)"
        
        diag = build_diagnostic_info(failure_reason=reason)
        full_df = d.copy()
        full_df['_InWindow'] = False
        log_wtg(f"FAILED: {reason}")
        # Still include explorer_candidates so failed WTGs show up in Window Explorer
        return {'wtg': wtg, 'summary': diag, 'data_slice': full_df, 'window_start': None, 'window_end': None,
                'explorer_candidates': explorer_candidates}
    
    # Fallback doesn't meet 24h nominal but use it anyway
    return finalize_result(fallback)


def run_browser_analysis(file_bytes: bytes, file_name: str, params: dict,
                         alarm_file_bytes: bytes = None, alarm_file_name: str = None) -> str:
    """
    Main entry point for browser-based analysis.
    Returns JSON string with results.
    
    Parameters:
    - file_bytes: SCADA data file bytes
    - file_name: SCADA data file name
    - params: Dictionary of analysis parameters
    - alarm_file_bytes: Optional alarm log file bytes
    - alarm_file_name: Optional alarm log file name
    """
    global output_excel_bytes, cached_analysis_data
    
    t0 = time.time()
    
    def log(msg):
        """Print log message for browser console"""
        print(f"[Python] {msg}")
    
    try:
        # Initialize these early so they exist even if exceptions occur
        summaries = []
        data_sheets = {}
        alarm_sheets = {}
        
        log(f"Starting analysis: {file_name} ({len(file_bytes):,} bytes)")
        
        # Read SCADA file
        file_ext = file_name.lower().split('.')[-1]
        file_buffer = io.BytesIO(bytes(file_bytes))
        
        if file_ext in ['xlsx', 'xlsm', 'xltx', 'xltm']:
            raw_df = pd.read_excel(file_buffer, engine='openpyxl')
        elif file_ext == 'xls':
            raw_df = pd.read_excel(file_buffer)
        elif file_ext == 'csv':
            raw_df = pd.read_csv(file_buffer)
        elif file_ext == 'tsv':
            raw_df = pd.read_csv(file_buffer, sep='\t')
        else:
            return json.dumps({'success': False, 'error': f'Unsupported file type: {file_ext}'})
        
        # Normalize headers and detect structure
        df = normalize_headers(raw_df)
        ts_col = detect_timestamp_column(df)
        df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
        df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
        
        # Parse separators and detect WTGs
        seps = _parse_separators(params.get('wtg_separators', '_, -, space'))
        wtgs = extract_wtgs_flexible(df.columns.tolist(), seps)
        log(f"Loaded {len(df):,} rows, {len(wtgs)} WTGs: {', '.join(wtgs[:8])}{'...' if len(wtgs) > 8 else ''}")
        
        if not wtgs:
            return json.dumps({'success': False, 'error': 'No WTGs detected. Check column naming.'})
        
        # Infer bin size
        inferred_bin = int(round(infer_bin_minutes(df[ts_col])))
        bin_minutes = float(params.get('bin_minutes') or inferred_bin)
        
        # Parse category lists
        allowed_categories = [x.strip() for x in params.get('allowed_categories', 'Normal operation').split(',') if x.strip()]
        disallowed_categories = [x.strip() for x in params.get('disallowed_categories', 'Manufacturer,Unscheduled maintenance').split(',') if x.strip()]
        active_base_categories = [x.strip() for x in params.get('active_base_categories', 'Normal operation').split(',') if x.strip()]
        allowed_window_categories = [x.strip() for x in params.get('allowed_window_categories', 'Normal operation,Scheduled maintenance,Owner,Environmental,Utility').split(',') if x.strip()]
        disqualifying_window_categories = [x.strip() for x in params.get('disqualifying_window_categories', 'Manufacturer,Unscheduled maintenance').split(',') if x.strip()]

        pr_threshold = float(params.get('pr_threshold', 0.97))
        wind_col_hint = params.get('wind_speed_column_hint')
        air_density_col_hint = params.get('air_density_column_hint')
        power_curve_text = params.get('power_curve_text', '') or ''
        if not power_curve_text.strip():
            return json.dumps({'success': False, 'error': 'Power curve text is required for expected power calculation.'})
        power_curve_df = parse_power_curve_text(power_curve_text)
        power_curve_arrays = prepare_power_curve(power_curve_df)
        
        # Energy source
        energy_source = params.get('energy_source', 'power')
        
        # Toggle-based parameters (replaces mode)
        require_availability = params.get('require_availability', True)
        require_nominal = params.get('require_nominal', True)
        require_energy = params.get('require_energy', True)
        energy_threshold_mwh = params.get('energy_threshold_mwh', 54.0)
        
        # Set min_availability based on toggle
        if require_availability:
            min_availability_pct = float(params.get('min_availability_pct', 96))
        else:
            min_availability_pct = 0
        
        # Load alarm file if provided
        alarm_df = None
        alarm_cols = (None, None, None, None)
        if alarm_file_bytes and alarm_file_name:
            log(f"Alarm file loaded: {alarm_file_name}")
            alarm_ext = alarm_file_name.lower().split('.')[-1]
            alarm_buffer = io.BytesIO(bytes(alarm_file_bytes))
            
            if alarm_ext in ['xlsx', 'xlsm', 'xltx', 'xltm']:
                a_raw = pd.read_excel(alarm_buffer, engine='openpyxl')
            elif alarm_ext == 'xls':
                a_raw = pd.read_excel(alarm_buffer)
            elif alarm_ext == 'csv':
                a_raw = pd.read_csv(alarm_buffer)
            elif alarm_ext == 'tsv':
                a_raw = pd.read_csv(alarm_buffer, sep='\t')
            else:
                a_raw = None
            
            if a_raw is not None:
                a_df = normalize_headers(a_raw)
                ts_a, unit_a, type_a, sev_a = detect_alarm_columns(
                    a_df,
                    ts_hint=params.get('alarms_ts_col'),
                    unit_hint=params.get('alarms_unit_col'),
                    type_hint=params.get('alarms_type_col'),
                    severity_hint=params.get('alarms_severity_col')
                )
                if ts_a:
                    alarm_df = preprocess_alarm_df(a_df, ts_a, unit_a, type_a)
                    alarm_cols = (ts_a, unit_a, type_a, sev_a)
        
        # Parse alarm params
        keep_event_types = [x.strip() for x in params.get('alarms_keep_event_types', 'Alarm,Warning').split(',') if x.strip()]
        min_severity = params.get('alarms_min_severity')
        if min_severity is not None:
            min_severity = int(min_severity)
        
        # Process each WTG
        log(f"Processing {len(wtgs)} WTGs...")
        summaries = []
        data_sheets = {}
        wtg_candidates = {}  # Store top candidates per WTG for window explorer
        pending_selections = []  # WTGs that need user to choose between tied windows
        
        for i, wtg in enumerate(wtgs):
            # Slice dataframe for this WTG - columns are always WTG_ColumnName format
            # WTG names are unique (e.g., A01, A02) so no overlap concerns
            prefix = wtg + '_'
            keep_cols = [c for c in df.columns if c == ts_col or c.startswith(prefix)]
            
            # Validate we have the minimum required columns
            wtg_specific_cols = [c for c in keep_cols if c != ts_col]
            if len(wtg_specific_cols) == 0:
                log(f"  {wtg}: ERROR - No WTG-specific columns found!")
                summaries.append({'WTG': wtg, 'Status': 'FAILED', 'Reason': f'No columns found with prefix {prefix}'})
                continue
            
            # Create isolated copy with fresh index
            wtg_df = df.loc[:, keep_cols].copy().reset_index(drop=True)
            
            result = process_wtg_fast(
                d=wtg_df,
                ts_col=ts_col,
                wtg=wtg,
                rated_power_kw=float(params.get('rated_power_kw', 4500)),
                bin_minutes=bin_minutes,
                test_hours=int(params.get('test_hours', 72)),
                extension_hours=int(params.get('extension_hours', 24)),
                min_availability_pct=min_availability_pct,
                nominal_threshold_pct=float(params.get('nominal_threshold_pct', 99)),
                allowed_categories=allowed_categories,
                disallowed_categories=disallowed_categories,
                active_base_categories=active_base_categories,
                energy_source=energy_source,
                require_nominal=require_nominal,
                require_energy=require_energy,
                energy_threshold_mwh=energy_threshold_mwh,
                allowed_window_categories=allowed_window_categories,
                disqualifying_window_categories=disqualifying_window_categories,
                power_curve_arrays=power_curve_arrays,
                pr_threshold=pr_threshold,
                wind_col_hint=wind_col_hint,
                air_density_col_hint=air_density_col_hint
            )
            
            # Handle results with multiple candidates (for alarm-based selection)
            if 'candidates' in result:
                candidates = result['candidates']
                explorer_cands = result.get('explorer_candidates', candidates)  # All candidates for explorer
                finalize_func = result['finalize']
                
                # If we have alarm data, count alarms for each candidate and prefer windows with 0 alarms
                if alarm_df is not None and alarm_cols[0] is not None:
                    ts_a, unit_a, type_a, sev_a = alarm_cols
                    
                    # Count alarms for ALL candidates (both valid and explorer)
                    # Use a set to track which candidates we've processed
                    all_cands_to_process = list(explorer_cands)
                    processed_ids = set(id(c) for c in all_cands_to_process)
                    for c in candidates:
                        if id(c) not in processed_ids:
                            all_cands_to_process.append(c)
                            processed_ids.add(id(c))
                    
                    for cand in all_cands_to_process:
                        start = pd.to_datetime(cand['test_start'])
                        end = pd.to_datetime(cand['test_end'])
                        filtered = filter_alarm_log_for_window(
                            alarm_df, unit_value=wtg, win_start=start, win_end=end,
                            ts_col=ts_a, unit_col=unit_a, type_col=type_a,
                            keep_event_types=keep_event_types, sev_col=sev_a,
                            min_severity=min_severity
                        )
                        
                        # Count alarms vs warnings separately
                        alarm_count = 0
                        warning_count = 0
                        if type_a and type_a in filtered.columns:
                            for _, row in filtered.iterrows():
                                etype = str(row.get(type_a, '')).lower()
                                if 'alarm' in etype:
                                    alarm_count += 1
                                elif 'warning' in etype:
                                    warning_count += 1
                                else:
                                    # Default to alarm if unclear
                                    alarm_count += 1
                        else:
                            # No type column - count all as alarms
                            alarm_count = len(filtered)
                        
                        cand['alarm_count'] = alarm_count
                        cand['warning_count'] = warning_count
                        cand['event_count'] = alarm_count + warning_count
                        
                        # Store event details for each candidate
                        cand['alarm_details'] = []
                        if not filtered.empty:
                            for _, row in filtered.head(20).iterrows():  # Limit to 20 for display
                                alarm_ts = pd.to_datetime(row[ts_a]).strftime('%Y-%m-%d %H:%M') if pd.notna(row[ts_a]) else 'Unknown'
                                desc = str(row.get('Description', row.get('Message', row.get(type_a, 'Unknown'))))[:100]
                                etype = str(row.get(type_a, 'Unknown')) if type_a else 'Unknown'
                                cand['alarm_details'].append({'timestamp': alarm_ts, 'description': desc, 'event_type': etype})
                    
                    # Re-sort candidates: prioritize fewer events (alarms first, then warnings)
                    def alarm_aware_score(c):
                        meets_nominal = 0 if c['nominal_hours'] >= 24.0 else 1
                        pr_tier = 0 if c['contains_3x'] else (1 if c['contains_1x'] else 2)
                        # Sort by alarms first, then warnings
                        return (meets_nominal, pr_tier, c['alarm_count'], c['warning_count'], c['wall_clock_bins'], c['start'])
                    
                    candidates.sort(key=alarm_aware_score)
                else:
                    # No alarm data - set counts to 0 for all candidates (both valid and explorer)
                    all_cands = list(explorer_cands)
                    processed_ids = set(id(c) for c in all_cands)
                    for c in candidates:
                        if id(c) not in processed_ids:
                            all_cands.append(c)
                    for cand in all_cands:
                        cand['alarm_count'] = 0
                        cand['warning_count'] = 0
                        cand['event_count'] = 0
                        cand['alarm_details'] = []
                
                # Store top 10 candidates for window explorer (JSON serializable)
                # Use explorer_cands which includes non-viable windows
                top_candidates = []
                for idx, cand in enumerate(explorer_cands[:10]):
                    start_ts = pd.to_datetime(cand['test_start'])
                    end_ts = pd.to_datetime(cand['test_end'])
                    top_candidates.append({
                        'index': idx,
                        'start': start_ts.strftime('%Y-%m-%d %H:%M'),
                        'end': end_ts.strftime('%Y-%m-%d %H:%M'),
                        'active_hours': round(cand['active_bins'] * bin_minutes / 60.0, 1),
                        'wall_hours': round(cand['wall_clock_bins'] * bin_minutes / 60.0, 1),
                        'nominal_hours': round(cand['nominal_hours'], 1),
                        'availability_pct': round(cand['availability_pct'], 1),
                        'energy_mwh': round(cand['energy_kwh'] / 1000.0, 2),
                        'alarm_count': cand.get('alarm_count', 0),
                        'warning_count': cand.get('warning_count', 0),
                        'event_count': cand.get('event_count', 0),
                        'alarm_details': cand.get('alarm_details', []),
                        'energy_floor': '3x' if cand['contains_3x'] else ('1x' if cand['contains_1x'] else 'none'),
                        'viable': cand.get('viable', True),
                        'issues': cand.get('issues', [])
                    })
                wtg_candidates[wtg] = top_candidates
                
                # Check for ties on the minimum alarm count
                if len(candidates) > 1:
                    min_alarm_count = candidates[0]['alarm_count']
                    tied_candidates = [c for c in candidates if c['alarm_count'] == min_alarm_count]
                    if len(tied_candidates) > 1:
                        # Multiple windows tied on alarm count - flag for user selection
                        pending_selections.append({
                            'wtg': wtg,
                            'tied_count': len(tied_candidates),
                            'alarm_count': min_alarm_count,
                            'candidates': top_candidates[:len(tied_candidates)]  # Include tied candidates
                        })
                
                # Select the best candidate (first after sorting) and log final result
                best_cand = candidates[0]
                c_active = best_cand['active_bins'] * bin_minutes / 60.0
                c_wall = best_cand['wall_clock_bins'] * bin_minutes / 60.0
                energy_floor = "3x" if best_cand['contains_3x'] else ("1x" if best_cand['contains_1x'] else "none")
                alarm_info = f", alarms={best_cand.get('alarm_count', 0)}, warnings={best_cand.get('warning_count', 0)}" if alarm_df is not None else ""
                
                # Check min energy requirement at the end
                energy_mwh = best_cand['energy_kwh'] / 1000.0
                energy_pass = True
                if require_energy and energy_threshold_mwh is not None:
                    energy_pass = energy_mwh >= energy_threshold_mwh
                
                status_msg = "PASSED" if energy_pass else "ENERGY_FAIL"
                log(f"  {wtg}: {status_msg} | active={c_active:.1f}h, nominal={best_cand['nominal_hours']:.1f}h, avail={best_cand['availability_pct']:.1f}%, energy={energy_mwh:.1f}MWh, floor={energy_floor}{alarm_info}")
                
                result = finalize_func(best_cand)
            elif 'explorer_candidates' in result:
                # Failed WTG but has explorer candidates - process them for Window Explorer
                explorer_cands = result['explorer_candidates']
                
                # Count alarms for explorer candidates if alarm data available
                if alarm_df is not None and alarm_cols[0] is not None:
                    ts_a, unit_a, type_a, sev_a = alarm_cols
                    for cand in explorer_cands:
                        start = pd.to_datetime(cand['test_start'])
                        end = pd.to_datetime(cand['test_end'])
                        filtered = filter_alarm_log_for_window(
                            alarm_df, unit_value=wtg, win_start=start, win_end=end,
                            ts_col=ts_a, unit_col=unit_a, type_col=type_a,
                            keep_event_types=keep_event_types, sev_col=sev_a,
                            min_severity=min_severity
                        )
                        
                        # Count alarms vs warnings separately
                        alarm_count = 0
                        warning_count = 0
                        if type_a and type_a in filtered.columns:
                            for _, row in filtered.iterrows():
                                etype = str(row.get(type_a, '')).lower()
                                if 'alarm' in etype:
                                    alarm_count += 1
                                elif 'warning' in etype:
                                    warning_count += 1
                                else:
                                    alarm_count += 1
                        else:
                            alarm_count = len(filtered)
                        
                        cand['alarm_count'] = alarm_count
                        cand['warning_count'] = warning_count
                        cand['event_count'] = alarm_count + warning_count
                        
                        # Store event details
                        cand['alarm_details'] = []
                        if not filtered.empty:
                            for _, row in filtered.head(20).iterrows():
                                alarm_ts = pd.to_datetime(row[ts_a]).strftime('%Y-%m-%d %H:%M') if pd.notna(row[ts_a]) else 'Unknown'
                                desc = str(row.get('Description', row.get('Message', row.get(type_a, 'Unknown'))))[:100]
                                etype = str(row.get(type_a, 'Unknown')) if type_a else 'Unknown'
                                cand['alarm_details'].append({'timestamp': alarm_ts, 'description': desc, 'event_type': etype})
                else:
                    for cand in explorer_cands:
                        cand['alarm_count'] = 0
                        cand['warning_count'] = 0
                        cand['event_count'] = 0
                        cand['alarm_details'] = []
                
                # Store top 10 candidates for Window Explorer (all non-viable in this case)
                top_candidates = []
                for idx, cand in enumerate(explorer_cands[:10]):
                    start_ts = pd.to_datetime(cand['test_start'])
                    end_ts = pd.to_datetime(cand['test_end'])
                    top_candidates.append({
                        'index': idx,
                        'start': start_ts.strftime('%Y-%m-%d %H:%M'),
                        'end': end_ts.strftime('%Y-%m-%d %H:%M'),
                        'active_hours': round(cand['active_bins'] * bin_minutes / 60.0, 1),
                        'wall_hours': round(cand['wall_clock_bins'] * bin_minutes / 60.0, 1),
                        'nominal_hours': round(cand['nominal_hours'], 1),
                        'availability_pct': round(cand['availability_pct'], 1),
                        'energy_mwh': round(cand['energy_kwh'] / 1000.0, 2),
                        'alarm_count': cand.get('alarm_count', 0),
                        'warning_count': cand.get('warning_count', 0),
                        'event_count': cand.get('event_count', 0),
                        'alarm_details': cand.get('alarm_details', []),
                        'energy_floor': '3x' if cand['contains_3x'] else ('1x' if cand['contains_1x'] else 'none'),
                        'viable': cand.get('viable', False),
                        'issues': cand.get('issues', [])
                    })
                wtg_candidates[wtg] = top_candidates
                log(f"  {wtg}: FAILED - {len(top_candidates)} candidate windows available in explorer")
            
            if result.get('summary'):
                summaries.append(result['summary'])
            # Store only essential columns for charting (to keep Excel file size manageable)
            if isinstance(result.get('data_slice'), pd.DataFrame) and not result['data_slice'].empty:
                full_df = result['data_slice']
                # Keep only columns needed for charts: timestamp, power, expected power, wind, air density, in-window flag
                chart_cols = [ts_col]
                power_col_name = f"{wtg}_Grid Production Power Avg."
                if power_col_name in full_df.columns:
                    chart_cols.append(power_col_name)
                for col in ['_WindSpeed', '_AirDensityUsed', '_ExpectedPower', '_PerformanceRatio', '_InWindow']:
                    if col in full_df.columns:
                        chart_cols.append(col)
                chart_df = full_df[[c for c in chart_cols if c in full_df.columns]].copy()
                data_sheets[f"{wtg}_Data"] = chart_df
                
                # Clear heavy data immediately after extracting chart columns
                del full_df
                if 'data_slice' in result:
                    del result['data_slice']
                if 'full_data' in result:
                    del result['full_data']
            
            # Clear WTG dataframe to free memory before next iteration
            del wtg_df
        
        # Attach alarms to summaries
        # Use 'Test End (72h Active)' for alarm counting - only count alarms in required test hours
        # Window extensions for nominal power should not affect alarm count
        alarm_sheets = {}
        if alarm_df is not None:
            ts_a, unit_a, type_a, sev_a = alarm_cols
            for s in summaries:
                if 'Test Start' in s and 'Test End' in s:
                    wtg = s['WTG']
                    start = pd.to_datetime(s['Test Start'])
                    # Use 72h active end if available, otherwise fall back to full window end
                    end_key = 'Test End (72h Active)' if 'Test End (72h Active)' in s else 'Test End'
                    end = pd.to_datetime(s[end_key])
                    filtered = filter_alarm_log_for_window(
                        alarm_df, unit_value=wtg, win_start=start, win_end=end,
                        ts_col=ts_a, unit_col=unit_a, type_col=type_a,
                        keep_event_types=keep_event_types, sev_col=sev_a,
                        min_severity=min_severity
                    )
                    
                    # Count alarms vs warnings separately
                    alarm_count = 0
                    warning_count = 0
                    if type_a and type_a in filtered.columns:
                        for _, row in filtered.iterrows():
                            etype = str(row.get(type_a, '')).lower()
                            if 'alarm' in etype:
                                alarm_count += 1
                            elif 'warning' in etype:
                                warning_count += 1
                            else:
                                alarm_count += 1  # Default to alarm
                    else:
                        alarm_count = len(filtered)
                    
                    s['Alarms in Window'] = alarm_count
                    s['Warnings in Window'] = warning_count
                    s['Total Events'] = alarm_count + warning_count
                    
                    # Build alarm details string: timestamp - description for each event
                    if not filtered.empty:
                        alarm_sheets[f'{wtg}_Alarms'] = filtered
                        # Find description column (try common names)
                        desc_col = None
                        for cand in ['Description', 'description', 'Message', 'message', 'Event Description', 
                                     'Event Message', 'EventDescription', 'EventMessage', 'Alarm Description',
                                     'Alarm Message', 'Text', 'text', 'Details', 'details']:
                            if cand in filtered.columns:
                                desc_col = cand
                                break
                        
                        # Format each alarm as "[TYPE] YYYY-MM-DD HH:MM - Description"
                        ts_strs = pd.to_datetime(filtered[ts_a], errors='coerce').dt.strftime('%Y-%m-%d %H:%M').fillna('Unknown Time')
                        
                        # Get event types for labeling
                        if type_a and type_a in filtered.columns:
                            etypes = filtered[type_a].astype(str).str.strip().apply(
                                lambda x: '[ALARM]' if 'alarm' in x.lower() else ('[WARNING]' if 'warning' in x.lower() else '[EVENT]')
                            )
                        else:
                            etypes = pd.Series(['[EVENT]'] * len(filtered), index=filtered.index)
                        
                        if desc_col and desc_col in filtered.columns:
                            descs = filtered[desc_col].astype(str).str.strip()
                            alarm_details = (etypes + ' ' + ts_strs + ' - ' + descs).tolist()
                        elif type_a and type_a in filtered.columns:
                            type_descs = filtered[type_a].astype(str).str.strip()
                            alarm_details = (etypes + ' ' + ts_strs + ' - ' + type_descs).tolist()
                        else:
                            alarm_details = (etypes + ' ' + ts_strs).tolist()
                        
                        s['Event Details'] = '; '.join(alarm_details)
                    else:
                        s['Event Details'] = ''
        else:
            # No alarm file - set empty event details for all summaries
            for s in summaries:
                s['Alarms in Window'] = 0
                s['Warnings in Window'] = 0
                s['Total Events'] = 0
                s['Event Details'] = ''
        
        elapsed = round(time.time() - t0, 2)
        
        # Count passed/failed for summary
        passed_count = sum(1 for s in summaries if 'PASSED' in str(s.get('Status', '')))
        failed_count = len(summaries) - passed_count
        log(f"Analysis complete in {elapsed}s: {passed_count} passed, {failed_count} failed")
        
        # Reorder summary columns to user's specified order
        # Priority columns first, then any remaining columns
        priority_columns = [
            'WTG',
            'Energy Pass/Fail',
            'Availability (%)',
            'Date of Test Start',
            'Time of Test Start',
            'Date of Test End',
            'Time of Test End',
            'Cumulative Time of Testing (h)',
            'Alarms in Window',
            'Warnings in Window',
            'Total Events',
            'Event Details',
            'First Nominal Power',
            'Last Nominal Power',
            'Cumulative Nominal Hours (h)',
            'Total Energy in Window (MWh)',
        ]
        
        # Columns to remove
        remove_columns = {'Min Availability Required'}
        
        for s in summaries:
            # Add Energy Pass/Fail column based on energy threshold
            if require_energy and energy_threshold_mwh is not None:
                total_energy_mwh = s.get('Total Energy in Window (MWh)', 0) or 0
                s['Energy Pass/Fail'] = 'PASS' if total_energy_mwh >= energy_threshold_mwh else 'FAIL'
            else:
                s['Energy Pass/Fail'] = 'N/A (threshold disabled)'
            
            # Build reordered summary
            reordered = {}
            # Add priority columns first (if they exist in summary)
            for col in priority_columns:
                if col in s:
                    reordered[col] = s[col]
            # Add remaining columns (except removed ones)
            for col, val in s.items():
                if col not in reordered and col not in remove_columns:
                    reordered[col] = val
            # Replace the summary dict
            s.clear()
            s.update(reordered)
        
        # Cache data for Excel generation
        cached_analysis_data = {
            'summaries': summaries,
            'data_sheets': data_sheets,
            'alarm_sheets': alarm_sheets
        }
        output_excel_bytes = None
        
        # Convert summaries to JSON-serializable format
        summaries_json = []
        for s in summaries:
            s_copy = {}
            for k, v in s.items():
                if pd.isna(v):
                    s_copy[k] = None
                elif hasattr(v, 'isoformat'):
                    s_copy[k] = str(v)
                elif isinstance(v, (np.integer, np.floating)):
                    s_copy[k] = float(v) if isinstance(v, np.floating) else int(v)
                elif isinstance(v, (int, float, str, bool, type(None))):
                    s_copy[k] = v
                else:
                    s_copy[k] = str(v)
            summaries_json.append(s_copy)
        
        # Convert chart data to JSON-serializable format using vectorized operations
        chart_data = {}
        for sheet_name, sheet_df in data_sheets.items():
            if isinstance(sheet_df, pd.DataFrame) and not sheet_df.empty:
                # Sample if too large (every Nth row for charts)
                if len(sheet_df) > 2000:
                    sample_df = sheet_df.iloc[::len(sheet_df)//2000].copy()
                else:
                    sample_df = sheet_df.copy()
                
                # Convert datetime columns to strings
                for col in sample_df.columns:
                    if pd.api.types.is_datetime64_any_dtype(sample_df[col]):
                        sample_df[col] = sample_df[col].astype(str)
                
                # Vectorized conversion: replace NaN with None, convert numpy types
                sample_df = sample_df.replace({np.nan: None})
                
                # Use to_dict for fast serialization, then convert to row format
                records = sample_df.to_dict('split')
                rows_list = []
                for row in records['data']:
                    row_data = []
                    for val in row:
                        if isinstance(val, (np.integer,)):
                            row_data.append(int(val))
                        elif isinstance(val, (np.floating,)):
                            row_data.append(float(val) if np.isfinite(val) else None)
                        else:
                            row_data.append(val)
                    rows_list.append(row_data)
                
                chart_data[sheet_name] = {
                    'columns': records['columns'],
                    'rows': rows_list
                }
                
                # Clear sample_df after serialization
                del sample_df
        
        # Count alarm events before cleanup
        alarm_events_total = sum(len(adf) for adf in alarm_sheets.values()) if alarm_sheets else 0
        alarms_were_processed = alarm_df is not None
        
        return json.dumps({
            'success': True,
            'wtg_count': len(wtgs),
            'windows_found': len(summaries),
            'bin_minutes': bin_minutes,
            'processing_time': elapsed,
            'wtgs': wtgs,
            'alarms_processed': alarms_were_processed,
            'alarm_events_total': alarm_events_total,
            'summaries': summaries_json,
            'chart_data': chart_data,
            'wtg_candidates': wtg_candidates,  # Top 10 candidates per WTG for window explorer
            'pending_selections': pending_selections  # WTGs with tied alarm counts needing user input
        })
        
    except Exception as e:
        import traceback
        return json.dumps({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })


def generate_excel_on_demand() -> str:
    """
    Generate Excel file from cached analysis data.
    Called when user clicks Download button.
    Returns JSON with success status.
    
    NOTE: Memory optimization means cache is cleared after analysis.
    Excel export requires re-running the analysis.
    """
    global output_excel_bytes, cached_analysis_data
    
    def log(msg):
        print(f"[Python] {msg}")
    
    log(f"generate_excel_on_demand called. cached_analysis_data is {'None' if cached_analysis_data is None else 'set'}")
    
    try:
        if cached_analysis_data is None:
            log("ERROR: cached_analysis_data is None!")
            return json.dumps({'success': False, 'error': 'Excel export not available. Data was cleared to free memory. Please re-run the analysis if you need Excel export.'})
        
        summaries = cached_analysis_data['summaries']
        data_sheets = cached_analysis_data['data_sheets']
        alarm_sheets = cached_analysis_data['alarm_sheets']
        
        log(f"Generating Excel with {len(summaries)} summaries, {len(data_sheets)} data sheets...")
        t_excel = time.time()
        output_buffer = io.BytesIO()
        
        with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
            log("Writing Summary sheet...")
            if summaries:
                pd.DataFrame(summaries).sort_values('WTG').to_excel(writer, sheet_name='Summary', index=False)
            else:
                pd.DataFrame({'Message': ['No valid windows found']}).to_excel(writer, sheet_name='Summary', index=False)
            
            log(f"Writing {len(data_sheets)} data sheets...")
            for sheet_name, sheet_df in data_sheets.items():
                sheet_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            
            if alarm_sheets:
                log(f"Writing {len(alarm_sheets)} alarm sheets...")
                for nm, adf in alarm_sheets.items():
                    adf.to_excel(writer, sheet_name=nm[:31], index=False)
                all_alarms = pd.concat([adf.assign(_WTG=nm.split('_')[0]) for nm, adf in alarm_sheets.items()], ignore_index=True)
                all_alarms.to_excel(writer, sheet_name='All_Alarms_Filtered', index=False)
        
        log(f"Excel generation complete: {round(time.time() - t_excel, 2)}s")
        
        output_buffer.seek(0)
        output_excel_bytes = output_buffer.read()
        
        return json.dumps({'success': True, 'size_bytes': len(output_excel_bytes)})
        
    except Exception as e:
        import traceback
        return json.dumps({'success': False, 'error': str(e), 'traceback': traceback.format_exc()})


def clear_analysis_cache() -> str:
    """
    Clear cached analysis data to free memory.
    Note: Memory is now automatically cleared after analysis completes.
    This function is kept for compatibility but may not be needed.
    """
    global output_excel_bytes, cached_analysis_data
    
    output_excel_bytes = None
    cached_analysis_data = None
    
    # Force garbage collection
    gc.collect()
    
    print("[Python] Analysis cache cleared, memory freed")
    return json.dumps({'success': True})


def run_browser_analysis_from_text(data_text: str, file_name: str, params: dict,
                                   alarm_text: str = None, alarm_file_name: str = None) -> str:
    """
    Entry point for browser-based analysis using pasted text data.
    Converts pasted text to bytes and delegates to run_browser_analysis.
    
    Parameters:
    - data_text: SCADA data as pasted text (tab or comma separated)
    - file_name: Virtual file name (used for format detection, should be .csv or .tsv)
    - params: Dictionary of analysis parameters
    - alarm_text: Optional alarm log as pasted text
    - alarm_file_name: Optional alarm log virtual file name
    """
    try:
        # Detect delimiter from the pasted text
        first_line = data_text.split('\n')[0] if data_text else ''
        if '\t' in first_line:
            # Tab-separated (from Excel copy)
            file_name = 'pasted_data.tsv'
            data_bytes = data_text.encode('utf-8')
        else:
            # Comma-separated
            file_name = 'pasted_data.csv'
            data_bytes = data_text.encode('utf-8')
        
        # Handle alarm text similarly
        alarm_bytes = None
        if alarm_text and alarm_text.strip():
            alarm_first_line = alarm_text.split('\n')[0] if alarm_text else ''
            if '\t' in alarm_first_line:
                alarm_file_name = 'pasted_alarms.tsv'
            else:
                alarm_file_name = 'pasted_alarms.csv'
            alarm_bytes = alarm_text.encode('utf-8')
        
        # Delegate to the existing function
        return run_browser_analysis(data_bytes, file_name, params, alarm_bytes, alarm_file_name)
        
    except Exception as e:
        import traceback
        return json.dumps({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })
