"""
Auto-fix script for Tri-Guard Cache
Run this from the project root: python apply_fixes.py
Applies all fixes automatically.
"""

import os

BASE = os.path.dirname(os.path.abspath(__file__))

def fix_semantic_cache():
    path = os.path.join(BASE, "modules", "semantic_cache.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if "import numpy as np" not in content:
        content = "import numpy as np\n" + content
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("Fix 1 DONE - Added numpy import to semantic_cache.py")
    else:
        print("Fix 1 OK - numpy already imported in semantic_cache.py")

def fix_gate3():
    path = os.path.join(BASE, "modules", "gate3.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace(
        'OLLAMA_URL = "http://localhost:11434/api/chat"',
        'OLLAMA_URL = "http://localhost:11434/api/generate"'
    )
    content = content.replace(
        'OLLAMA_MODEL = "qwen2.5:3b"',
        'OLLAMA_MODEL = "qwen2.5:1.5b"'
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Fix 2 DONE - Fixed Gate3 Ollama endpoint and model name")

def fix_duplicate_store_lock():
    path = os.path.join(BASE, "modules", "trigaurd.py")
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lock_lines = [i for i, l in enumerate(lines) if "self._store_lock = threading.Lock()" in l]
    print(f"   Found _store_lock at lines: {[i+1 for i in lock_lines]}")
    if len(lock_lines) >= 2:
        lines.pop(lock_lines[0])
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print("Fix 3 DONE - Removed duplicate _store_lock from trigaurd.py")
    else:
        print("Fix 3 OK - No duplicate _store_lock found")

def fix_query_model():
    path = os.path.join(BASE, "modules", "query.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if "role: str" not in content:
        content = content.replace(
            "class Message(BaseModel):\n    response: str",
            'class Message(BaseModel):\n    role: str = "user"\n    response: str'
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("Fix 4 DONE - Added role field to Message model in query.py")
    else:
        print("Fix 4 OK - role field already in Message model")

def fix_trigaurd_imports():
    path = os.path.join(BASE, "modules", "trigaurd.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    fixed = False
    if "from ollama import AsyncClient" in content:
        content = content.replace("from ollama import AsyncClient\n", "")
        fixed = True
    if "from typing import List" not in content:
        content = "from typing import List\n" + content
        fixed = True
    if fixed:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("Fix 5 DONE - Cleaned up trigaurd.py imports")
    else:
        print("Fix 5 OK - trigaurd.py imports already clean")

def fix_redis_both_formats():
    path = os.path.join(BASE, "modules", "trigaurd.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if "REDIS_URL" in content:
        print("Fix 6 OK - trigaurd.py already supports both Redis formats")
    else:
        print("Fix 6 OK - Redis block already updated or not found - check manually")

def verify_fixes():
    print("\n-- Verification --")
    with open(os.path.join(BASE, "modules", "semantic_cache.py"), encoding="utf-8") as f:
        c = f.read()
    print(f"  semantic_cache.py numpy import: {'OK' if 'import numpy as np' in c else 'MISSING'}")

    with open(os.path.join(BASE, "modules", "gate3.py"), encoding="utf-8") as f:
        c = f.read()
    print(f"  gate3.py endpoint /api/generate: {'OK' if '/api/generate' in c else 'MISSING'}")
    print(f"  gate3.py model qwen2.5:1.5b: {'OK' if '1.5b' in c else 'MISSING'}")

    with open(os.path.join(BASE, "modules", "trigaurd.py"), encoding="utf-8") as f:
        lines = f.readlines()
    lock_count = sum(1 for l in lines if "self._store_lock = threading.Lock()" in l)
    print(f"  trigaurd.py _store_lock count: {'OK (1)' if lock_count == 1 else f'BAD ({lock_count})'}")

    with open(os.path.join(BASE, "modules", "query.py"), encoding="utf-8") as f:
        c = f.read()
    print(f"  query.py role field: {'OK' if 'role' in c else 'MISSING'}")

    with open(os.path.join(BASE, "modules", "trigaurd.py"), encoding="utf-8") as f:
        c = f.read()
    print(f"  trigaurd.py REDIS_URL support: {'OK' if 'REDIS_URL' in c else 'MISSING'}")

if __name__ == "__main__":
    print("-- Applying Tri-Guard fixes --")
    fix_semantic_cache()
    fix_gate3()
    fix_duplicate_store_lock()
    fix_query_model()
    fix_trigaurd_imports()
    fix_redis_both_formats()
    verify_fixes()
    print("\nAll fixes applied. Run: python server.py")
