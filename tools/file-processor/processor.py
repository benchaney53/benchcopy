"""
Python File Processor
This module runs in the browser using PyScript/Pyodide.
It processes uploaded text files and returns the result.
"""

from pyscript import window
import json


def process_file(content: str, filename: str) -> str:
    """
    Process the uploaded file content.
    
    Args:
        content: The text content of the uploaded file
        filename: The name of the uploaded file
        
    Returns:
        Processed content as a string
    """
    # Get file extension
    ext = filename.lower().split('.')[-1] if '.' in filename else ''
    
    # Statistics
    lines = content.split('\n')
    line_count = len(lines)
    word_count = len(content.split())
    char_count = len(content)
    
    result = []
    result.append("=" * 50)
    result.append("📊 FILE ANALYSIS REPORT")
    result.append("=" * 50)
    result.append(f"\n📁 Filename: {filename}")
    result.append(f"📏 Lines: {line_count}")
    result.append(f"📝 Words: {word_count}")
    result.append(f"🔤 Characters: {char_count}")
    
    # File-type specific processing
    if ext == 'json':
        result.append("\n" + "-" * 50)
        result.append("🔍 JSON ANALYSIS")
        result.append("-" * 50)
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                result.append(f"Type: Object with {len(data)} keys")
                result.append(f"Keys: {', '.join(list(data.keys())[:10])}")
                if len(data.keys()) > 10:
                    result.append(f"... and {len(data.keys()) - 10} more keys")
            elif isinstance(data, list):
                result.append(f"Type: Array with {len(data)} items")
            result.append("✅ Valid JSON structure")
        except json.JSONDecodeError as e:
            result.append(f"❌ Invalid JSON: {str(e)}")
    
    elif ext == 'csv':
        result.append("\n" + "-" * 50)
        result.append("📊 CSV ANALYSIS")
        result.append("-" * 50)
        if lines:
            # Try to detect delimiter
            first_line = lines[0]
            if ',' in first_line:
                delimiter = ','
            elif '\t' in first_line:
                delimiter = '\t'
            elif ';' in first_line:
                delimiter = ';'
            else:
                delimiter = ','
            
            columns = first_line.split(delimiter)
            result.append(f"Columns: {len(columns)}")
            result.append(f"Total rows: {line_count}")
            result.append(f"First row values: {', '.join(columns[:5])}")
            if len(columns) > 5:
                result.append(f"... and {len(columns) - 5} more columns")
    
    elif ext == 'py':
        result.append("\n" + "-" * 50)
        result.append("🐍 PYTHON CODE ANALYSIS")
        result.append("-" * 50)
        
        # Count imports (excluding commented lines)
        import_count = sum(1 for line in lines 
                          if not line.strip().startswith('#') and 
                          (line.strip().startswith('import ') or line.strip().startswith('from ')))
        def_count = sum(1 for line in lines if line.strip().startswith('def '))
        class_count = sum(1 for line in lines if line.strip().startswith('class '))
        comment_count = sum(1 for line in lines if line.strip().startswith('#'))
        
        result.append(f"Imports: {import_count}")
        result.append(f"Functions: {def_count}")
        result.append(f"Classes: {class_count}")
        result.append(f"Comments: {comment_count}")
    
    # Word frequency analysis
    result.append("\n" + "-" * 50)
    result.append("📈 WORD FREQUENCY (Top 10)")
    result.append("-" * 50)
    
    # Simple word frequency counter
    words = content.lower().split()
    word_freq = {}
    for word in words:
        # Remove common punctuation
        word = word.strip('.,;:!?"\'()[]{}')
        if word and len(word) > 2:  # Only words longer than 2 chars
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (word, count) in enumerate(sorted_words, 1):
        result.append(f"{i}. '{word}': {count} times")
    
    # Empty line analysis
    empty_lines = sum(1 for line in lines if not line.strip())
    if empty_lines > 0:
        result.append(f"\n📄 Empty lines: {empty_lines}")
    
    result.append("\n" + "=" * 50)
    result.append("✅ Analysis Complete!")
    result.append("=" * 50)
    
    return '\n'.join(result)


# Expose function to JavaScript
window.processFile = process_file
