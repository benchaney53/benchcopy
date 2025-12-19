# Run Test Tool

Browser-based SCADA analysis tool for finding optimal reliability and HOTS test windows.

## Features

- **Runs entirely in browser** - No server required, uses Pyodide (Python compiled to WebAssembly)
- **Supports multiple file formats** - xlsx, xls, csv, tsv
- **Auto-detection** - Automatically detects WTGs, timestamp columns, and bin sizes
- **Configurable parameters** - Test hours, availability thresholds, category filters
- **Alarm log filtering** - Optional alarm file to filter events for each test window
- **Fast 24h subwindow checks** - Optimized algorithm using prefix sums
- **Excel output** - Downloads results as formatted Excel workbook with per-WTG data and alarm sheets

## How It Works

1. Upload your SCADA data file (required)
2. Optionally upload an Alarm Log file
3. Configure test parameters (or use defaults)
4. Click "Run Analysis"
5. Download the Excel report

## Required SCADA Columns

The tool expects columns following this naming pattern:
- `{WTG}_Power, Average` or similar power column
- `{WTG}_System States TurbineState` 
- `{WTG}_1_Report Category`
- A timestamp column (PCTimeStamp, Timestamp, etc.)

## Optional Alarm Log Columns

The tool auto-detects these columns in alarm files:
- Timestamp: Detected, Event time, Occurred, Alarm time, etc.
- Unit: Unit, WTG, Turbine, Turbine ID, etc.
- Event Type: Event type, Type, EventType
- Severity: Severity, Level

## Modes

- **M01 Mode**: Finds 72h active windows meeting strict availability (96%+) and energy requirements
- **Reliability Mode**: Informational analysis without strict thresholds

## Parameters

- **Rated Power (kW)**: Default 4500
- **Test Hours**: Default 72
- **Extension Hours**: Default 24
- **Min Availability %**: Default 96
- **Nominal Threshold %**: Default 99
- **Energy Source**: Power Average or Total Active Power
- **Allowed/Disallowed Categories**: Configure which categories count as active or blocked
- **Allowed/Disqualifying Window Categories**: Additional window validation

## Technical Notes

- Uses Pyodide v0.25.1 with pandas, numpy, and openpyxl
- Initial load may take 10-15 seconds to download Python runtime
- Large files (>50MB) may be slow to process in browser
- Based on Reliability_Test_p1_hotfix.py algorithm
