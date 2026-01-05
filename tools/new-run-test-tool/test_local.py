"""
Local test script for new-run-test-tool analysis.
Uses Sheet1.csv (alarms) and Sheet2.csv (SCADA) from this folder.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Import the analysis module
import analysis

# ============================================================================
# LOAD DATA
# ============================================================================

DATA_DIR = Path(__file__).parent

print("Loading SCADA data...")
scada_df = pd.read_csv(DATA_DIR / "Sheet2.csv")
print(f"  SCADA: {len(scada_df)} rows, {len(scada_df.columns)} columns")

print("Loading alarm data...")
alarm_df = pd.read_csv(DATA_DIR / "Sheet1.csv")
print(f"  Alarms: {len(alarm_df)} rows")

# Convert to dict format like the browser does
scada_data = {col: scada_df[col].tolist() for col in scada_df.columns}
scada_columns = list(scada_df.columns)

alarm_data = {col: alarm_df[col].tolist() for col in alarm_df.columns}
alarm_columns = list(alarm_df.columns)

# ============================================================================
# POWER CURVE (same as saved in browser localStorage)
# ============================================================================

# Vestas V150 4.5MW power curve (from browser)
power_curve_text = """WS,1.225,0.950,0.975,1.000,1.025,1.050,1.075,1.100,1.125,1.150,1.175,1.200,1.250,1.275
3.0,81,51,54,56,59,62,65,67,70,73,76,78,84,86
3.5,172,122,127,131,136,140,145,149,154,158,163,167,176,181
4.0,285,210,217,223,230,237,244,251,258,264,271,278,291,298
4.5,424,317,327,337,346,356,366,375,385,395,404,414,433,443
5.0,596,451,464,477,491,504,517,530,543,557,570,583,609,622
5.5,808,615,632,650,668,685,703,720,738,755,773,790,825,843
6.0,1061,811,834,857,880,902,925,948,970,993,1016,1039,1084,1106
6.5,1360,1044,1072,1101,1130,1159,1188,1217,1245,1274,1303,1331,1389,1417
7.0,1710,1317,1353,1389,1425,1461,1496,1532,1568,1604,1639,1675,1746,1781
7.5,2106,1629,1673,1717,1761,1805,1848,1891,1935,1978,2021,2064,2149,2191
8.0,2549,1982,2034,2086,2138,2190,2242,2293,2345,2396,2447,2498,2599,2649
8.5,3021,2376,2437,2498,2559,2620,2678,2737,2795,2854,2909,2965,3075,3129
9.0,3471,2794,2860,2926,2992,3058,3119,3180,3241,3302,3358,3415,3524,3578
9.5,3861,3187,3254,3321,3388,3455,3516,3576,3637,3698,3752,3807,3910,3959
10.0,4180,3559,3626,3693,3760,3827,3883,3938,3994,4050,4093,4136,4214,4249
10.5,4372,3901,3959,4018,4076,4135,4175,4215,4255,4295,4321,4347,4389,4407
11.0,4470,4189,4232,4274,4316,4359,4379,4400,4421,4442,4451,4460,4475,4480
11.5,4494,4374,4394,4414,4435,4455,4462,4469,4477,4484,4487,4490,4495,4497
12.0,4500,4457,4464,4472,4480,4487,4490,4492,4495,4498,4498,4499,4500,4500
12.5,4500,4486,4489,4492,4495,4498,4499,4499,4500,4500,4500,4500,4500,4500
13.0,4500,4497,4498,4498,4499,4500,4500,4500,4500,4500,4500,4500,4500,4500
13.5,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500
14.0,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500
14.5,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500
15.0,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500
15.5,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500
16.0,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500
16.5,4498,4498,4498,4498,4498,4498,4498,4498,4498,4498,4498,4498,4498,4498
17.0,4473,4473,4473,4473,4473,4473,4473,4473,4473,4473,4473,4473,4473,4473
17.5,4394,4394,4394,4394,4394,4394,4394,4394,4394,4394,4394,4394,4394,4394
18.0,4268,4268,4268,4268,4268,4268,4268,4268,4268,4268,4268,4268,4268,4268
18.5,4139,4139,4139,4139,4139,4139,4139,4139,4139,4139,4139,4139,4139,4139
19.0,4031,4031,4031,4031,4031,4031,4031,4031,4031,4031,4031,4031,4031,4031
19.5,3909,3909,3909,3909,3909,3909,3909,3909,3909,3909,3909,3909,3909,3909
20.0,3771,3771,3771,3771,3771,3771,3771,3771,3771,3771,3771,3771,3771,3771
20.5,3607,3606,3606,3606,3606,3606,3606,3606,3606,3607,3607,3607,3607,3607
21.0,3408,3407,3407,3407,3408,3408,3408,3408,3408,3408,3408,3408,3408,3408
21.5,3180,3179,3179,3179,3179,3179,3179,3179,3179,3179,3179,3179,3180,3180
22.0,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917,2917
22.5,2645,2645,2645,2645,2645,2645,2645,2645,2645,2645,2645,2645,2645,2645"""

power_curve_df = analysis.parse_power_curve_text(power_curve_text)
power_curve_arrays = analysis.prepare_power_curve(power_curve_df)
print(f"Power curve loaded: WS range {power_curve_arrays[0].min()}-{power_curve_arrays[0].max()} m/s, "
      f"AD range {power_curve_arrays[1].min()}-{power_curve_arrays[1].max()} kg/m³")

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================

params = {
    'rated_power_kw': 4500,
    'min_active_hours': 72,
    'min_nominal_hours': 24,
    'min_availability_pct': 96,
    'extension_hours': 24,
    'require_nominal': True,
    'require_three_strike': True,
    'pr_threshold': 0.90,  # 90% - user's typical setting
    'bin_minutes': 10
}

# ============================================================================
# RUN ANALYSIS FOR A01
# ============================================================================

wtg = 'A01'
print(f"\n{'='*60}")
print(f"Processing {wtg}")
print(f"{'='*60}")

# Extract columns for this WTG
wtg_cols = analysis.extract_wtg_columns(scada_data, scada_columns, wtg, 'PCTimeStamp')
print(f"Found columns: cat={wtg_cols['cat_col']}, power={wtg_cols['power_col']}, wind={wtg_cols['wind_col']}, air={wtg_cols['air_col']}")

# Create DataFrame
df = pd.DataFrame({
    'timestamp': pd.to_datetime(wtg_cols['timestamp'], errors='coerce'),
    'category': [str(c).strip().lower() if pd.notna(c) else '' for c in wtg_cols['category']],
    'power': pd.to_numeric(wtg_cols['power'], errors='coerce'),
    'wind_speed': pd.to_numeric(wtg_cols['wind_speed'], errors='coerce'),
    'air_density': pd.to_numeric(wtg_cols['air_density'], errors='coerce')
})

# Drop NaT timestamps
df = df.dropna(subset=['timestamp']).reset_index(drop=True)
print(f"Data rows: {len(df)}")

# Check categories
cats = df['category'].value_counts()
print(f"\nCategory distribution:")
for cat, count in cats.head(10).items():
    print(f"  {cat}: {count}")

# Check power range
pwr = df['power']
print(f"\nPower: min={pwr.min():.1f}, max={pwr.max():.1f}, mean={pwr.mean():.1f} kW")

# Check wind speed range
ws = df['wind_speed']
print(f"Wind: min={ws.min():.1f}, max={ws.max():.1f}, mean={ws.mean():.1f} m/s")

# Check air density range
ad = df['air_density']
print(f"Air density: min={ad.min():.3f}, max={ad.max():.3f}, mean={ad.mean():.3f} kg/m³")

# ============================================================================
# COMPUTE EXPECTED POWER AND NOMINAL FLAGS
# ============================================================================

expected_power_raw = analysis.compute_expected_power(df['wind_speed'], df['air_density'], power_curve_arrays)
expected_power = expected_power_raw * 0.99  # Scale by 0.99 like old tool

valid_expected = np.isfinite(expected_power).sum()
print(f"\nExpected power: valid={valid_expected}/{len(df)}, min={np.nanmin(expected_power):.0f}, max={np.nanmax(expected_power):.0f}, mean={np.nanmean(expected_power):.0f} kW")

# Performance ratio
perf_ratio = np.divide(df['power'].values, expected_power, 
                       out=np.full(len(df), np.nan), 
                       where=np.isfinite(expected_power) & (expected_power > 0))

pr_comparison_threshold = 0.90  # Match params
pr_nominal = perf_ratio >= pr_comparison_threshold

print(f"PR-based nominal bins: {pr_nominal.sum()} ({pr_nominal.sum() * 10 / 60:.1f}h)")

# Check PR distribution
pr_valid = perf_ratio[np.isfinite(perf_ratio)]
print(f"\nPerformance ratio distribution:")
print(f"  Valid: {len(pr_valid)}/{len(df)}")
print(f"  Min: {pr_valid.min():.3f}")
print(f"  Max: {pr_valid.max():.3f}")
print(f"  Mean: {pr_valid.mean():.3f}")
print(f"  Median: {np.median(pr_valid):.3f}")
print(f"  >= 0.97: {(pr_valid >= 0.97).sum()} ({100*(pr_valid >= 0.97).mean():.1f}%)")
print(f"  >= 0.99: {(pr_valid >= 0.99).sum()} ({100*(pr_valid >= 0.99).mean():.1f}%)")
print(f"  >= 1.00: {(pr_valid >= 1.00).sum()} ({100*(pr_valid >= 1.00).mean():.1f}%)")

# Active mask
active = np.array([c == 'normal operation' for c in df['category']])
print(f"\nActive bins: {active.sum()} ({active.sum() * 10 / 60:.1f}h)")

# Combined nominal (active AND nominal)
nominal_active = pr_nominal & active
print(f"Nominal active bins: {nominal_active.sum()} ({nominal_active.sum() * 10 / 60:.1f}h)")

# ============================================================================
# CHECK THE SPECIFIC WINDOW FROM OLD TOOL
# Old tool found: 2025-12-06 18:00 to 2025-12-09 17:50, 35h nominal
# ============================================================================

print(f"\n{'='*60}")
print("Checking specific window from old tool")
print(f"{'='*60}")

start_dt = pd.Timestamp('2025-12-06 18:00:00')
end_dt = pd.Timestamp('2025-12-09 17:50:00')

mask = (df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)
window_df = df[mask]
window_pr_nominal = pr_nominal[mask]
window_active = active[mask]

print(f"Window: {start_dt} to {end_dt}")
print(f"  Bins: {mask.sum()}")
print(f"  Active bins: {window_active.sum()} ({window_active.sum() * 10 / 60:.1f}h)")
print(f"  PR-nominal bins: {window_pr_nominal.sum()} ({window_pr_nominal.sum() * 10 / 60:.1f}h)")
print(f"  Nominal active bins: {(window_pr_nominal & window_active).sum()} ({(window_pr_nominal & window_active).sum() * 10 / 60:.1f}h)")

# Check PR distribution in this window
window_pr = perf_ratio[mask]
window_pr_valid = window_pr[np.isfinite(window_pr)]
print(f"\n  Window PR distribution:")
print(f"    Valid: {len(window_pr_valid)}")
print(f"    >= 0.97: {(window_pr_valid >= 0.97).sum()} ({100*(window_pr_valid >= 0.97).mean():.1f}%)")
print(f"    >= 0.99: {(window_pr_valid >= 0.99).sum()}")
print(f"    >= 1.00: {(window_pr_valid >= 1.00).sum()}")

# Sample some actual vs expected
print(f"\n  Sample of actual vs expected power in window:")
sample_idx = window_df.index[:10]
for i in sample_idx:
    ws = df.loc[i, 'wind_speed']
    ad = df.loc[i, 'air_density']
    actual = df.loc[i, 'power']
    expected = expected_power[i]
    pr = perf_ratio[i]
    cat = df.loc[i, 'category']
    print(f"    WS={ws:.1f} AD={ad:.3f} Actual={actual:.0f} Expected={expected:.0f} PR={pr:.3f} Cat={cat}")

# ============================================================================
# RUN FULL ANALYSIS
# ============================================================================

print(f"\n{'='*60}")
print("Running full analysis")
print(f"{'='*60}")

result = analysis.run_analysis(
    scada_data=scada_data,
    scada_columns=scada_columns,
    alarm_data=alarm_data,
    alarm_columns=alarm_columns,
    power_curve_text=power_curve_text,
    selected_wtgs=['A01'],
    rated_power_kw=params['rated_power_kw'],
    bin_minutes=params['bin_minutes'],
    test_hours=params['min_active_hours'],
    extension_hours=params['extension_hours'],
    min_availability_pct=params['min_availability_pct'],
    pr_threshold=params['pr_threshold'],
    require_nominal=params['require_nominal'],
    min_nominal_hours=params['min_nominal_hours']
)

print("\nResult:")
import json
print(json.dumps(result, indent=2, default=str))
