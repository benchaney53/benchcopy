# Enhanced Run Test Analyzer

Find valid 72-hour reliability test windows from wind turbine SCADA data based on M01 contractual requirements.

## Features

- **Memory-efficient processing**: Analyzes one WTG at a time to minimize RAM usage
- **3-Strike fault detection**: Fails windows where any alarm code appears 3+ times
- **Availability tracking**: Validates ≥96% production-based availability
- **Nominal power validation**: Requires 24h cumulative hours at nominal power (PR-based or 90% rated fallback)
- **Cooling verification**: Checks power > 50% rated to confirm cooling systems functional
- **Browser-based**: Runs entirely in your browser using Pyodide (Python via WebAssembly)

## Input Data Format

### SCADA Data (Required)
Excel or CSV with 10-minute interval data. Expected columns per WTG:

| Column Pattern | Example | Purpose |
|----------------|---------|---------|
| `PCTimeStamp` | `2025-11-22 00:00:00` | Timestamp column |
| `{WTG}_1_Report Category` | `A01_1_Report Category (1)` | Availability category |
| `{WTG}_Total Active power` | `A01_Total Active power (79)` | Power output (kW) |
| `{WTG}_Ambient WindSpeed Avg.` | `A01_Ambient WindSpeed Avg. (183)` | Wind speed (m/s) |
| `{WTG}_Ambient Airdensity AirDensityAvg Avg` | `A01_Ambient Airdensity... (287)` | Air density (kg/m³) |

**Valid Report Categories:**
- `Normal operation` - Active, counts toward availability
- `Manufacturer` - Blocked, counts against availability
- `Unscheduled maintenance` - Blocked, counts against availability
- `Owner`, `Environmental`, `Utility`, `Scheduled maintenance` - Allowed in window

### Alarm Log (Optional)
Excel or CSV with alarm/fault events. Expected columns:

| Column | Purpose |
|--------|---------|
| `Unit` | WTG identifier (e.g., `A01`) |
| `Code` | Fault/alarm code (e.g., `10160`) |
| `Description` | Alarm description |
| `Detected` | Alarm timestamp |
| `Event type` | `Alarm log (A)` or `Warning log (W)` |

### Power Curve (Optional)
Paste a table with wind speed rows and air density columns:

```
WS    1.10    1.15    1.20    1.25
3     0       0       0       0
4     100     105     110     115
5     300     315     330     345
6     600     630     660     690
...
```

## M01 Contractual Requirements Validated

### Hard Rules (Must Pass)

| Requirement | Implementation |
|-------------|----------------|
| 72-hour operating period | All Test Hours must be present with no disqualifying categories |
| Minimum energy production | Total energy ≥ (Rated Power × Base Window Hours) ÷ 2 |
| ≥96% availability | Calculated from Report Category (Normal operation vs Blocked) |
| 3-strike rule | Windows fail if any fault code appears 3+ times |

### Soft Rules (Preferences)

| Requirement | Implementation |
|-------------|----------------|
| Nominal power hours | Window must accumulate user-specified nominal runtime hours |
| Extension for low nominal | If base window has no disqualifying categories but insufficient nominal hours, extend by minimum time needed |
| Prefer fewer alarms | Select window with fewest alarms, then fewest warnings |
| Cooling systems functional | Power > 50% rated power |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Rated Power (kW) | 4200 | Turbine rated capacity |
| Bin Size (min) | 10 | SCADA data interval |
| Test Duration (h) | 72 | Required test window length |
| Extension Hours | 24 | Additional hours if conditions not met |
| Min Availability (%) | 96 | Minimum production-based availability |
| PR Threshold | 0.99 | Performance ratio threshold for nominal power |
| Min Nominal Hours | 24 | Cumulative hours required at nominal power |

## Usage

1. **Upload SCADA data** - Excel file with Sheet2 format or CSV
2. **Upload Alarm log** (optional) - Excel file with Sheet1 format or CSV
3. **Select WTGs** - All detected WTGs are selected by default
4. **Adjust parameters** - Modify thresholds as needed for your contract
5. **Paste power curve** (optional) - For PR-based nominal power calculation
6. **Run Analysis** - Click the button and wait for results
7. **Export** - Download Excel summary of all results

## Output

For each WTG, the tool identifies candidate windows and reports:

- **Start/End Time** - Window boundaries
- **Availability %** - Production-based availability
- **Nominal Hours** - Cumulative hours at nominal power
- **Energy (MWh)** - Total energy produced in window
- **Total Alarms** - Count of alarm events in window
- **3-Strike Status** - Pass/fail with violating codes if any
- **Issues** - List of requirements not met

## Technology

- **Pyodide v0.25.1** - Python in WebAssembly
- **Pandas + NumPy** - Data analysis
- **SheetJS** - Excel file parsing in browser
- **Chart.js** - Results visualization
- **openpyxl** - Excel export generation
