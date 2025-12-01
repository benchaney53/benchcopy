#!/usr/bin/env python3
"""
Auto-generate tools-manifest.json by scanning the tools directory
Run this script whenever you add a new tool to automatically update the manifest
"""

import json
import os
from pathlib import Path


def generate_manifest():
    """Scan tools directory and generate manifest from tool.json files"""
    tools_dir = Path(__file__).parent / 'tools'
    manifest = []
    
    if not tools_dir.exists():
        print("Error: tools directory not found")
        return
    
    # Scan each subdirectory in tools/
    for item in sorted(tools_dir.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            tool_json = item / 'tool.json'
            
            if tool_json.exists():
                try:
                    with open(tool_json, 'r', encoding='utf-8') as f:
                        tool_data = json.load(f)
                        manifest.append(tool_data)
                        print(f"✓ Added: {tool_data.get('name', item.name)}")
                except json.JSONDecodeError as e:
                    print(f"✗ Error reading {tool_json}: {e}")
                except Exception as e:
                    print(f"✗ Error processing {tool_json}: {e}")
            else:
                print(f"⚠ Skipped: {item.name} (no tool.json found)")
    
    # Write manifest
    manifest_path = tools_dir / 'tools-manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Generated manifest with {len(manifest)} tool(s)")
    print(f"📄 Saved to: {manifest_path}")


if __name__ == '__main__':
    generate_manifest()
