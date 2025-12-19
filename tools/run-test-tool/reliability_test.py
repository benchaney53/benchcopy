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
import numpy as np
import pandas as pd

# Detection candidates
TIMESTAMP_CANDIDATES = ['PCTimeStamp', 'Timestamp', 'TimeStamp', 'DateTime', 'Datetime', 'Time', 'Date']
POWER_SUFFIX_CANDIDATES = ['Power, Average', 'Grid Production Power Avg.', 'Total Active power']
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


# ---------------- Fast 24h subwindow check ----------------

def _span_contains_active_energy_fast(active_mask: np.ndarray,
                                      energy_active_kwh_prefix: np.ndarray,
                                      active_positions: np.ndarray,
                                      s: int, e: int,
                                      target_active_bins: int,
                                      target_mwh: float) -> bool:
    """Check if within [s,e] there exists a contiguous subwindow of exactly
    target_active_bins ACTIVE samples whose energy >= target_mwh."""
    if target_active_bins <= 0:
        return True
    
    # Locate active indices within [s,e]
    left = bisect.bisect_left(active_positions.tolist(), s)
    right = bisect.bisect_right(active_positions.tolist(), e)
    count = right - left
    
    if count < target_active_bins:
        return False
    
    # Slide a window of size target_active_bins over this range
    for i in range(left, right - target_active_bins + 1):
        j = i + target_active_bins - 1
        e_kwh = energy_active_kwh_prefix[j] - (energy_active_kwh_prefix[i-1] if i > 0 else 0.0)
        if (e_kwh / 1000.0) + 1e-9 >= target_mwh:
            return True
    return False


def _evaluate_and_summarize(d: pd.DataFrame, ts_col: str, wtg: str,
                            power_col: str, state_col: str, cat_col: str,
                            s_idx: int, e_idx: int,
                            rated_power_kw: float, bin_minutes: float,
                            nominal_threshold_pct: float,
                            blocked_set: set, active_base_set: set,
                            energy_power_col: str) -> tuple:
    """Evaluate a window and build summary."""
    slice_df = d.iloc[s_idx:e_idx+1].copy()
    cats_norm = _normalize_category_series(slice_df[cat_col].astype(str))
    
    paused_mask = (~cats_norm.isin(blocked_set) & (~cats_norm.isin(active_base_set))).to_numpy()
    active_mask = ~paused_mask
    unavail_mask = cats_norm.isin(blocked_set).to_numpy()
    active_idx = np.where(active_mask)[0]
    
    availability_pct = (100.0 * np.mean(~unavail_mask[active_idx])) if len(active_idx) else float('nan')
    
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
    
    test_start_str = pd.to_datetime(slice_df[ts_col].iloc[0]).strftime('%Y-%m-%d %H:%M')
    test_end_str = pd.to_datetime(slice_df[ts_col].iloc[-1]).strftime('%Y-%m-%d %H:%M')
    active_hours = len(active_idx) * bin_minutes / 60.0
    span_hours = (e_idx - s_idx + 1) * bin_minutes / 60.0
    paused_hours = int((~active_mask).sum()) * bin_minutes / 60.0
    
    summary = {
        'WTG': wtg,
        'Rated Power (kW)': rated_power_kw,
        'Bin Size (min)': bin_minutes,
        'Active Hours (h)': round(active_hours, 2),
        'Wall-clock Span (h)': round(span_hours, 2),
        'Paused Hours (h)': round(paused_hours, 2),
        'Total Energy in Window (kWh)': round(total_energy_kwh, 2),
        'Availability (time-based, %)': round(availability_pct, 2),
        'Cumulative Nominal Hours (h)': round(total_nominal_hours, 2),
        'Interruptions (count)': interruptions,
        'Test Start': test_start_str,
        'Test End': test_end_str,
        'Power Column Used': power_col,
        'Energy Power Column Used': energy_power_col,
        'State Column': state_col,
        'Category Column': cat_col,
        'Window Categories Seen': ','.join(sorted(cats_norm.unique().tolist())),
    }
    return slice_df, summary


def process_wtg_fast(d: pd.DataFrame, ts_col: str, wtg: str,
                     rated_power_kw: float, bin_minutes: float,
                     test_hours: int, extension_hours: int,
                     min_availability_pct: float, nominal_threshold_pct: float,
                     allowed_categories: list, disallowed_categories: list,
                     active_base_categories: list, energy_source: str,
                     mode: str,
                     allowed_window_categories: list,
                     disqualifying_window_categories: list,
                     enforce_allowed_window_categories: bool) -> dict:
    """Process a single WTG using optimized algorithm from hotfix."""
    
    # Resolve columns
    seps = ['_', ' ', '-']
    power_col = find_col_with_suffix(d.columns.tolist(), wtg, POWER_SUFFIX_CANDIDATES, seps)
    if power_col is None:
        return {'wtg': wtg, 'summary': {'WTG': wtg, 'Mode': mode, 'Status': 'FAILED',
                                        'Reason': f'No power column found for {wtg}'}, 'data_slice': d.head(0)}
    
    total_active_col = find_col_with_suffix(d.columns.tolist(), wtg, ['Total Active power'], seps)
    energy_power_col = total_active_col if (energy_source == 'total_active' and total_active_col) else power_col
    state_col = find_col_with_suffix(d.columns.tolist(), wtg, [STATE_SUFFIX], seps) or f"{wtg}_{STATE_SUFFIX}"
    cat_col = find_col_with_suffix(d.columns.tolist(), wtg, [CAT_SUFFIX], seps) or f"{wtg}_{CAT_SUFFIX}"
    
    if cat_col not in d.columns:
        return {'wtg': wtg, 'summary': {'WTG': wtg, 'Mode': mode, 'Status': 'FAILED',
                                        'Reason': f'No category column found for {wtg}'}, 'data_slice': d.head(0)}
    
    # Normalize sets & arrays
    cats_norm_full = _normalize_category_series(d[cat_col].astype(str))
    blocked_set = {s.strip().lower() for s in disallowed_categories}
    active_base_set = {s.strip().lower() for s in active_base_categories}
    allow_window_set = {s.strip().lower() for s in (allowed_window_categories or [])}
    disq_window_set = {s.strip().lower() for s in (disqualifying_window_categories or [])}
    
    unavail_full = cats_norm_full.isin(blocked_set).to_numpy()
    paused_full = (~cats_norm_full.isin(blocked_set) & (~cats_norm_full.isin(active_base_set))).to_numpy()
    active_full = ~paused_full
    
    # Precompute prefix sums
    pwr = pd.to_numeric(d[power_col], errors='coerce').to_numpy()
    energy_full_kw = pd.to_numeric(d[energy_power_col], errors='coerce').fillna(0.0).to_numpy()
    e_kwh_per_row = energy_full_kw * (bin_minutes / 60.0)
    
    threshold_kw = rated_power_kw * (nominal_threshold_pct / 100.0)
    nominal_full = np.isfinite(pwr) & (pwr >= threshold_kw)
    
    ps_active = np.cumsum(active_full.astype(np.int32))
    ps_nominal_active = np.cumsum((nominal_full & active_full).astype(np.int32))
    ps_energy_active = np.cumsum((e_kwh_per_row * active_full).astype(np.float64))
    ps_unavail_active = np.cumsum((unavail_full & active_full).astype(np.int32))
    
    if enforce_allowed_window_categories:
        bad_mask = (cats_norm_full.isin(disq_window_set) | ~cats_norm_full.isin(allow_window_set)).to_numpy()
    else:
        bad_mask = cats_norm_full.isin(disq_window_set).to_numpy()
    ps_bad = np.cumsum(bad_mask.astype(np.int32))
    
    # Build active-only index and its energy prefix for fast 24h subwindow checks
    active_positions = np.where(active_full)[0]
    energy_on_active = e_kwh_per_row[active_full]
    ps_energy_on_active = np.cumsum(energy_on_active)
    
    def span_sum(ps, s, e):
        return ps[e] - (ps[s-1] if s > 0 else 0)
    
    def best_by_priority_active(target_active_bins: int, req_mwh_1x: float, req_mwh_3x: float):
        N = len(d)
        s = 0
        e = -1
        active_acc = 0
        best = None
        
        while s < N:
            while e + 1 < N and active_acc < target_active_bins:
                e += 1
                if active_full[e]:
                    active_acc += 1
            
            if active_acc >= target_active_bins:
                active_bins = span_sum(ps_active, s, e)
                nominal_hours = span_sum(ps_nominal_active, s, e) * (bin_minutes / 60.0)
                energy_kwh = span_sum(ps_energy_active, s, e)
                unavail_count = span_sum(ps_unavail_active, s, e)
                availability_pct = 100.0 * ((active_bins - unavail_count) / max(1, active_bins))
                has_bad = span_sum(ps_bad, s, e) > 0
                
                if availability_pct + 1e-9 >= min_availability_pct and not has_bad:
                    # Fast 24h active subwindow energy checks
                    target_bins_24h = int(round(24 * 60 / bin_minutes))
                    has_3x = _span_contains_active_energy_fast(active_full, ps_energy_on_active, active_positions,
                                                               s, e, target_bins_24h, req_mwh_3x)
                    has_1x = False if has_3x else _span_contains_active_energy_fast(active_full, ps_energy_on_active, active_positions,
                                                                                     s, e, target_bins_24h, req_mwh_1x)
                    interrupts = 1 if unavail_count > 0 else 0
                    cand = {
                        'start': int(s), 'end': int(e),
                        'active_bins': int(active_bins),
                        'availability_pct': float(availability_pct),
                        'nominal_hours': float(nominal_hours),
                        'energy_kwh': float(energy_kwh),
                        'interruptions': float(interrupts),
                        'contains_3x': bool(has_3x),
                        'contains_1x': bool(has_1x),
                    }
                    rank = (
                        0 if cand['contains_3x'] else (1 if cand['contains_1x'] else 2),
                        cand['start'],
                        -cand['energy_kwh'],
                        cand['interruptions'],
                    )
                    if best is None or rank < best[0]:
                        best = (rank, cand)
            
            if active_full[s]:
                active_acc -= 1
            s += 1
            if e < s - 1:
                e = s - 1
        
        return None if best is None else best[1]
    
    # Targets and floors
    target72 = int(round(test_hours * 60 / bin_minutes))
    target96 = int(round((test_hours + extension_hours) * 60 / bin_minutes)) if extension_hours > 0 else None
    req_mwh_1x = 0.5 * (rated_power_kw / 1000.0) * 24.0
    req_mwh_3x = 3.0 * req_mwh_1x
    
    # Evaluate candidates
    cand72 = best_by_priority_active(target72, req_mwh_1x, req_mwh_3x)
    cand96 = best_by_priority_active(target96, req_mwh_1x, req_mwh_3x) if target96 else None
    
    MIN_NOMINAL_HOURS_REQUIRED = 24.0
    
    def valid(c):
        return c and c['nominal_hours'] >= MIN_NOMINAL_HOURS_REQUIRED
    
    def choose(a, b):
        if a is None:
            return b
        if b is None:
            return a
        ar = (0 if a['contains_3x'] else (1 if a['contains_1x'] else 2), a['start'], -a['energy_kwh'], a['interruptions'])
        br = (0 if b['contains_3x'] else (1 if b['contains_1x'] else 2), b['start'], -b['energy_kwh'], b['interruptions'])
        return a if ar < br else b
    
    if valid(cand72) and valid(cand96):
        chosen = choose(cand72, cand96)
    elif valid(cand96):
        chosen = cand96
    elif valid(cand72):
        chosen = cand72
    else:
        fallback = cand96 if cand96 is not None else cand72
        if fallback is None:
            return {'wtg': wtg, 'summary': {'WTG': wtg, 'Mode': mode, 'Status': 'FAILED',
                                            'Reason': 'No feasible 96h/72h window found.'}, 'data_slice': d.head(0)}
        s_idx, e_idx = fallback['start'], fallback['end']
        slice_df, summ = _evaluate_and_summarize(d, ts_col, wtg, power_col, state_col, cat_col,
                                                 s_idx, e_idx, rated_power_kw, bin_minutes,
                                                 nominal_threshold_pct, blocked_set, active_base_set,
                                                 energy_power_col)
        summ.update({'Mode': mode, 'Status': 'completed_due_to_climatic_conditions', 'Nominal 24h Achieved': False})
        return {'wtg': wtg, 'summary': summ, 'data_slice': slice_df}
    
    s_idx, e_idx = chosen['start'], chosen['end']
    slice_df, summ = _evaluate_and_summarize(d, ts_col, wtg, power_col, state_col, cat_col,
                                             s_idx, e_idx, rated_power_kw, bin_minutes,
                                             nominal_threshold_pct, blocked_set, active_base_set,
                                             energy_power_col)
    summ.update({'Mode': mode,
                 'Contains 24h subwindow >= 3x floor': bool(chosen['contains_3x']),
                 'Contains 24h subwindow >= 1x floor': bool(chosen['contains_1x'])})
    return {'wtg': wtg, 'summary': summ, 'data_slice': slice_df}


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
    global output_excel_bytes
    
    t0 = time.time()
    
    try:
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
        
        # Normalize headers
        df = normalize_headers(raw_df)
        
        # Detect timestamp
        ts_col = detect_timestamp_column(df)
        df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
        df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
        
        # Parse separators and detect WTGs
        seps = _parse_separators(params.get('wtg_separators', '_, -, space'))
        wtgs = extract_wtgs_flexible(df.columns.tolist(), seps)
        
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
        enforce_allowed = not params.get('no_enforce_allowed_window', False)
        
        # Energy source
        energy_source = params.get('energy_source', 'power')
        
        # Mode adjustments
        mode = params.get('mode', 'm01')
        if mode == 'reliability':
            min_availability_pct = 0
        else:
            min_availability_pct = float(params.get('min_availability_pct', 96))
        
        # Load alarm file if provided
        alarm_df = None
        alarm_cols = (None, None, None, None)
        if alarm_file_bytes and alarm_file_name:
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
        summaries = []
        data_sheets = {}
        
        for wtg in wtgs:
            # Slice dataframe for this WTG
            prefix = wtg + '_'
            keep_cols = [c for c in df.columns if c == ts_col or c.startswith(prefix)]
            wtg_df = df.loc[:, keep_cols].copy()
            
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
                mode=mode,
                allowed_window_categories=allowed_window_categories,
                disqualifying_window_categories=disqualifying_window_categories,
                enforce_allowed_window_categories=enforce_allowed
            )
            
            if result.get('summary'):
                summaries.append(result['summary'])
            if isinstance(result.get('data_slice'), pd.DataFrame) and not result['data_slice'].empty:
                data_sheets[f"{wtg}_Data"] = result['data_slice']
        
        # Attach alarms to summaries
        alarm_sheets = {}
        if alarm_df is not None:
            ts_a, unit_a, type_a, sev_a = alarm_cols
            for s in summaries:
                if 'Test Start' in s and 'Test End' in s:
                    wtg = s['WTG']
                    start = pd.to_datetime(s['Test Start'])
                    end = pd.to_datetime(s['Test End'])
                    filtered = filter_alarm_log_for_window(
                        alarm_df, unit_value=wtg, win_start=start, win_end=end,
                        ts_col=ts_a, unit_col=unit_a, type_col=type_a,
                        keep_event_types=keep_event_types, sev_col=sev_a,
                        min_severity=min_severity
                    )
                    s['Events in Window (Alarm/Warning)'] = len(filtered)
                    if not filtered.empty:
                        alarm_sheets[f'{wtg}_Alarms'] = filtered
        
        # Write output Excel
        output_buffer = io.BytesIO()
        with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
            if summaries:
                pd.DataFrame(summaries).sort_values('WTG').to_excel(writer, sheet_name='Summary', index=False)
            else:
                pd.DataFrame({'Message': ['No valid windows found']}).to_excel(writer, sheet_name='Summary', index=False)
            
            for sheet_name, sheet_df in data_sheets.items():
                sheet_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            
            # Add alarm sheets
            if alarm_sheets:
                for nm, adf in alarm_sheets.items():
                    adf.to_excel(writer, sheet_name=nm[:31], index=False)
                # Combined alarms sheet
                all_alarms = pd.concat([adf.assign(_WTG=nm.split('_')[0]) for nm, adf in alarm_sheets.items()], ignore_index=True)
                all_alarms.to_excel(writer, sheet_name='All_Alarms_Filtered', index=False)
        
        output_buffer.seek(0)
        output_excel_bytes = output_buffer.read()
        
        elapsed = round(time.time() - t0, 2)
        
        return json.dumps({
            'success': True,
            'wtg_count': len(wtgs),
            'windows_found': len(summaries),
            'bin_minutes': bin_minutes,
            'processing_time': elapsed,
            'wtgs': wtgs[:10],
            'alarms_processed': alarm_df is not None,
            'alarm_events_total': sum(len(adf) for adf in alarm_sheets.values()) if alarm_sheets else 0
        })
        
    except Exception as e:
        import traceback
        return json.dumps({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })
