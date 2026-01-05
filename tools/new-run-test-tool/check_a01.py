import pandas as pd
import numpy as np
import analysis

# Load data
scada_df = pd.read_csv('Sheet2.csv', low_memory=False)
scada_data = {col: scada_df[col].tolist() for col in scada_df.columns}
scada_columns = list(scada_df.columns)

# Power curve from your image - Mode PO4-0S/PO4
power_curve_text = '''WS,1.225,0.950,0.975,1.000,1.025,1.050,1.075,1.100,1.125,1.150,1.175,1.200,1.250,1.275
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
12.5,4500,4497,4498,4498,4499,4499,4499,4500,4500,4500,4500,4500,4500,4500
13.0,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500,4500
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
22.5,2645,2645,2645,2645,2645,2645,2645,2645,2645,2645,2645,2645,2645,2645'''

power_curve_df = analysis.parse_power_curve_text(power_curve_text)
power_curve_arrays = analysis.prepare_power_curve(power_curve_df)

print('Power curve check at WS=8.0, AD=1.200:')
ws_arr, ad_arr, power_grid = power_curve_arrays
print(f'  WS array: {ws_arr[:5]}...{ws_arr[-3:]}')
print(f'  AD array: {ad_arr}')

# Find expected at WS=8.0, AD=1.200
ws_test = pd.Series([8.0])
ad_test = pd.Series([1.200])
expected = analysis.compute_expected_power(ws_test, ad_test, power_curve_arrays)
print(f'  Expected at WS=8.0, AD=1.200: {expected[0]:.0f} kW')

# A01 sample data
wtg_cols = analysis.extract_wtg_columns(scada_data, scada_columns, 'A01', 'PCTimeStamp')
df = pd.DataFrame({
    'timestamp': pd.to_datetime(wtg_cols['timestamp'], errors='coerce'),
    'category': [str(c).strip().lower() if pd.notna(c) else '' for c in wtg_cols['category']],
    'power': pd.to_numeric(wtg_cols['power'], errors='coerce'),
    'wind_speed': pd.to_numeric(wtg_cols['wind_speed'], errors='coerce'),
    'air_density': pd.to_numeric(wtg_cols['air_density'], errors='coerce')
}).dropna(subset=['timestamp'])

# Show first 20 normal operation bins
active = df['category'] == 'normal operation'
active_df = df[active].head(20)

print('\nFirst 20 normal operation bins for A01:')
print('WS       AD       Actual   Expected  PR')
print('-'*50)
for _, row in active_df.iterrows():
    exp = analysis.compute_expected_power(
        pd.Series([row['wind_speed']]), 
        pd.Series([row['air_density']]), 
        power_curve_arrays
    )[0]
    pr = row['power'] / exp if exp > 0 else 0
    print(f"{row['wind_speed']:5.1f}    {row['air_density']:.3f}    {row['power']:6.0f}   {exp:6.0f}    {pr:.1%}")

# Count bins where PR >= 90%
expected_all = analysis.compute_expected_power(df['wind_speed'], df['air_density'], power_curve_arrays)
pr_all = np.divide(df['power'].values, expected_all, out=np.full(len(df), np.nan), where=np.isfinite(expected_all) & (expected_all > 0))

active_mask = active.values
nominal_90 = active_mask & (pr_all >= 0.90)
nominal_95 = active_mask & (pr_all >= 0.95)
nominal_100 = active_mask & (pr_all >= 1.00)

print(f'\n\nNominal hours summary (during normal operation only):')
print(f'  Total active: {active_mask.sum()} bins = {active_mask.sum()*10/60:.1f}h')
print(f'  PR >= 90%: {nominal_90.sum()} bins = {nominal_90.sum()*10/60:.1f}h')
print(f'  PR >= 95%: {nominal_95.sum()} bins = {nominal_95.sum()*10/60:.1f}h')
print(f'  PR >= 100%: {nominal_100.sum()} bins = {nominal_100.sum()*10/60:.1f}h')

# Find best window - availability only matters for 72h active, not extension
print('\n\nSearching for viable windows...')
print('  Rule: Need 72h active with 96% availability, then can extend to get 24h nominal')
prefix_active = np.zeros(len(df)+1)
prefix_nominal = np.zeros(len(df)+1)
prefix_active[1:] = np.cumsum(active_mask)
prefix_nominal[1:] = np.cumsum(nominal_90)

min_active_h = 72
max_extension_h = 24
min_nominal_h = 24

viable_windows = []

for start in range(len(df)):
    # First find where we hit 72h active
    active_at_start = prefix_active[start]
    
    # Find the bin where we reach 72h active
    target_active = active_at_start + (min_active_h * 6)  # 72h = 432 bins
    
    # Binary search or linear scan for 72h active point
    end_72h = None
    for end in range(start + 1, len(df) + 1):
        active_h = (prefix_active[end] - prefix_active[start]) * 10 / 60
        if active_h >= min_active_h:
            end_72h = end
            break
    
    if end_72h is None:
        continue
    
    # Check availability for the 72h active portion
    wall_h_72 = (end_72h - start) * 10 / 60
    active_h_72 = (prefix_active[end_72h] - prefix_active[start]) * 10 / 60
    avail_pct = 100 * active_h_72 / wall_h_72
    
    if avail_pct < 96:
        continue  # Availability requirement not met for 72h portion
    
    # Now check nominal at 72h point
    nominal_at_72h = (prefix_nominal[end_72h] - prefix_nominal[start]) * 10 / 60
    
    if nominal_at_72h >= min_nominal_h:
        # Already have enough nominal at 72h
        viable_windows.append({
            'start': start,
            'end': end_72h,
            'wall_h': wall_h_72,
            'active_h': active_h_72,
            'nominal_h': nominal_at_72h,
            'avail_pct': avail_pct,
            'extended': False
        })
    else:
        # Need to extend - availability doesn't matter for extension
        max_end = min(len(df), start + int((72 + max_extension_h) * 6))
        
        for end_ext in range(end_72h + 1, max_end + 1):
            nominal_h = (prefix_nominal[end_ext] - prefix_nominal[start]) * 10 / 60
            if nominal_h >= min_nominal_h:
                wall_h = (end_ext - start) * 10 / 60
                active_h = (prefix_active[end_ext] - prefix_active[start]) * 10 / 60
                viable_windows.append({
                    'start': start,
                    'end': end_ext,
                    'wall_h': wall_h,
                    'active_h': active_h,
                    'nominal_h': nominal_h,
                    'avail_pct': avail_pct,  # Only for 72h portion
                    'extended': True
                })
                break

print(f'Found {len(viable_windows)} viable windows')

if viable_windows:
    # Show best (shortest wall time)
    viable_windows.sort(key=lambda w: w['wall_h'])
    best = viable_windows[0]
    print(f'\nBest viable window:')
    print(f'  Start: {df.iloc[best["start"]]["timestamp"]}')
    print(f'  End: {df.iloc[best["end"]-1]["timestamp"]}')
    print(f'  Wall hours: {best["wall_h"]:.1f}h')
    print(f'  Active hours: {best["active_h"]:.1f}h')
    print(f'  Nominal hours (PR>=90%): {best["nominal_h"]:.1f}h')
    print(f'  72h Availability: {best["avail_pct"]:.1f}%')
    print(f'  Extended: {best["extended"]}')
else:
    # Show best attempt - find window with best availability at 72h active
    print('\nNo viable window found. Checking why...')
    
    best_avail = 0
    best_nominal_in_96 = 0
    for start in range(len(df)):
        for end in range(start + 1, len(df) + 1):
            active_h = (prefix_active[end] - prefix_active[start]) * 10 / 60
            if active_h >= 72:
                wall_h = (end - start) * 10 / 60
                avail = 100 * active_h / wall_h
                nominal_h = (prefix_nominal[end] - prefix_nominal[start]) * 10 / 60
                if avail > best_avail:
                    best_avail = avail
                if avail >= 96 and nominal_h > best_nominal_in_96:
                    best_nominal_in_96 = nominal_h
                break
    
    print(f'  Best availability at 72h active: {best_avail:.1f}%')
    print(f'  Best nominal in windows with 96%+ availability: {best_nominal_in_96:.1f}h')
