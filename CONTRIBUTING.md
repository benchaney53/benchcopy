# Adding New Tools to Bench

## Quick Start

1. **Create Tool Directory**
   ```
   tools/
   └── your-tool-name/
       ├── index.html      # Required
       ├── tool.json       # Required
       └── [other files]   # Optional
   ```

2. **Create tool.json**
   ```json
   {
     "id": "your-tool-name",
     "name": "Display Name",
     "description": "What your tool does",
     "icon": "🔧",
     "category": "Category Name",
     "technologies": ["HTML", "JavaScript"]
   }
   ```

3. **Auto-generate the manifest**
   
   The manifest is automatically generated in two ways:
   
   **Automatic (GitHub):**
   - Push your changes to GitHub
   - A GitHub Action automatically runs and updates `tools/tools-manifest.json`
   - No manual steps needed!
   
   **Manual (Local Development):**
   ```bash
   python generate-manifest.py
   ```
   This scans all tool directories and regenerates the manifest.

4. **Create index.html**
   Include a back link to the main page:
   ```html
   <a href="../../index.html">← Back to Tools</a>
   ```

## Tool Metadata Fields

- **id** (required): Unique identifier, should match folder name
- **name** (required): Display name shown on the card
- **description** (required): Brief description of the tool
- **icon** (optional): Emoji or unicode character, defaults to 🔧
- **category** (optional): Category for grouping, defaults to "Uncategorized"
- **technologies** (optional): Array of technology tags

## Example Tool Structure

See the `file-processor` tool in `tools/file-processor/` for a complete example.

## Tips

- Use kebab-case for tool IDs and folder names
- Keep descriptions under 100 characters for best display
- Use emojis for icons to keep the design clean and simple
- Test locally before pushing to ensure paths are correct
- All tool paths should be relative to the tool's directory
