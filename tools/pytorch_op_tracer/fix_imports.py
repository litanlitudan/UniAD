#!/usr/bin/env python
"""Fix import issues in visualizer modules"""

import os
import re

def fix_imports_in_file(filepath):
    """Fix relative imports in a file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Pattern to find relative imports from ..core or ..analyzers etc
    pattern = r'^from \.\.([a-z_]+)\.([a-z_]+) import (.+)$'
    
    lines = content.split('\n')
    new_lines = []
    in_try_block = False
    already_has_try = False
    
    # Check if file already has try/except for imports
    for line in lines:
        if line.strip().startswith('try:') and 'import' in ''.join(lines[lines.index(line):lines.index(line)+5]):
            already_has_try = True
            break
    
    if already_has_try:
        print(f"Skipping {filepath} - already has try/except imports")
        return
    
    # Collect all relative imports
    relative_imports = []
    absolute_imports = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        match = re.match(pattern, line, re.MULTILINE)
        if match:
            module = match.group(1)
            submodule = match.group(2)
            imports = match.group(3)
            relative_imports.append(line)
            absolute_imports.append(f"    from {module}.{submodule} import {imports}")
            i += 1
        else:
            i += 1
    
    if not relative_imports:
        print(f"No relative imports in {filepath}")
        return
    
    # Build new content
    new_content = []
    import_section_done = False
    
    for line in lines:
        if not import_section_done and line in relative_imports:
            if not in_try_block:
                # Start try block
                new_content.append("try:")
                for rel_import in relative_imports:
                    new_content.append(f"    {rel_import.strip()}")
                new_content.append("except (ImportError, ValueError):")
                new_content.append("    import sys")
                new_content.append("    import os")
                new_content.append("    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))")
                for abs_import in absolute_imports:
                    new_content.append(abs_import)
                import_section_done = True
                in_try_block = True
        elif line not in relative_imports:
            new_content.append(line)
    
    # Write back
    with open(filepath, 'w') as f:
        f.write('\n'.join(new_content))
    
    print(f"Fixed imports in {filepath}")

# Fix all visualizer files
visualizers_dir = '/home/tanl/UniAD/tools/pytorch_op_tracer/visualizers'
files_to_fix = [
    'interactive_visualizer.py',
    'filter_engine.py', 
    'memory_timeline.py',
    'task_head_comparator.py',
    'queue_visualizer.py',
    'progressive_renderer.py'
]

for filename in files_to_fix:
    filepath = os.path.join(visualizers_dir, filename)
    if os.path.exists(filepath):
        fix_imports_in_file(filepath)

print("Done fixing imports")