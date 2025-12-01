# bench
A website repo that stores all the in-browser tools I've developed so far

## Overview

Bench is a dynamic landing page that showcases a collection of browser-based tools. All tools run entirely client-side, requiring no server-side processing.

## Features

- **Dynamic Tool Discovery**: Automatically loads and displays tools from the `/tools` directory
- **Responsive Design**: Modern, card-based layout that adapts to different screen sizes
- **Easy to Extend**: Simple JSON-based configuration for adding new tools

## Structure

```
bench/
├── index.html                      # Main landing page
├── tools/
│   ├── tools-manifest.json        # Central manifest listing all tools
│   └── [tool-name]/
│       ├── index.html             # Tool's main page
│       ├── tool.json              # Tool metadata
│       └── [other files]          # Tool-specific files
```

## Available Tools

### Python File Processor

Upload and analyze text files using Python running directly in your browser via PyScript/Pyodide.

**Features:**
- File Upload: Drag & drop or click to upload text files
- Python Processing: Files are processed using Python via [PyScript](https://pyscript.net/)
- File Analysis: Provides statistics like line count, word count, character count
- Format-specific Analysis: Special handling for JSON, CSV, and Python files
- Word Frequency: Shows the top 10 most common words
- Download Results: Export the processed output as a text file

## How It Works

### Automatic Tool Discovery

The landing page automatically displays all tools based on the `tools/tools-manifest.json` file. When you add a new tool:

1. **Create your tool folder** with `tool.json` and `index.html`
2. **Run the generator** (manually or via GitHub Actions)
3. **The manifest updates** automatically
4. **Your tool appears** on the landing page

### Manifest Generation

**GitHub Actions (Recommended):**
- Automatically runs when you push changes that include a `tool.json` file
- No manual steps required
- Updates and commits the manifest automatically

**Local Development:**
- Run `python generate-manifest.py` (or `generate-manifest.bat` on Windows)
- Manually commit the updated `tools/tools-manifest.json`

## Adding New Tools

To add a new tool to the landing page:

1. **Create a new folder** in `/tools` with your tool's name (use kebab-case)
   ```
   tools/your-tool-name/
   ```

2. **Add your tool files:**
   - `index.html` - Your tool's main page (required)
   - `tool.json` - Tool metadata (required)
   - Any other files your tool needs

3. **Create tool.json** with this structure:
   ```json
   {
     "id": "your-tool-id",
     "name": "Your Tool Name",
     "description": "Brief description of what your tool does",
     "icon": "🔧",
     "category": "Category Name",
     "technologies": ["Tech1", "Tech2"]
   }
   ```

4. **Update the manifest:**
   
   **Option A - Automatic (GitHub Actions):**
   - Just push your changes to GitHub
   - The workflow will automatically detect your new `tool.json` and update the manifest
   
   **Option B - Manual (Local):**
   - Run `python generate-manifest.py` from the root directory
   - This scans all tool directories and regenerates `tools/tools-manifest.json`
   - Commit the updated manifest file

That's it! Your tool will automatically appear on the landing page.

## Deployment

This site is designed for deployment on GitHub Pages:

1. Enable GitHub Pages in your repository settings
2. Set the source to the main branch
3. Your site will be available at `https://<username>.github.io/bench/`

## Local Development

To run locally:

```bash
python3 -m http.server 8080
```

Or with PowerShell:
```powershell
python -m http.server 8080
```

Then open `http://localhost:8080` in your browser.

## Technology

- **HTML/CSS/JavaScript**: Frontend interface and dynamic tool loading
- **JSON**: Tool configuration and manifest
- **PyScript/Pyodide**: Python compiled to WebAssembly (used in File Processor tool)
- **GitHub Pages**: Static site hosting
