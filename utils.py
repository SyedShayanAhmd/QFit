import os
import re

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

_invalid = re.compile(r'[^A-Za-z0-9._-]')
def safe_filename(filename):
    """Convert string to safe filename"""
    import re
    return re.sub(r'[^\w\-_.]', '_', filename)