# -*- mode: python ; coding: utf-8 -*-

import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# 1. Increase limits
sys.setrecursionlimit(5000)

# 2. Manual Hidden Imports (The Fix)
# We list the specific backports used by your libraries
hidden_imports = [
    # --- CRITICAL BACKPORTS FIX ---
    'backports',
    'backports.functools_lru_cache',
    'backports.shutil_get_terminal_size', 
    'backports.tarfile',
    'backports.zoneinfo',
    'jaraco',
    'jaraco.classes',
    'jaraco.context',
    'jaraco.functools',
    'jaraco.text',
    'importlib_metadata',
    'packaging',
    'packaging.version',
    'packaging.specifiers',
    'packaging.requirements',
    # ------------------------------
    
    'sklearn.neighbors._typedefs',
    'sklearn.utils._cython_blas',
    'sklearn.neighbors._quad_tree',
    'sklearn.tree',
    'sklearn.tree._utils',
    'sklearn.ensemble',
    'sklearn.linear_model',
    'pycaret.classification',
    'pycaret.regression',
    'pycaret.clustering',
    'pycaret.anomaly',
    'pandas',
    'numpy',
    'matplotlib',
    'seaborn',
    'joblib',
    'scipy.special.cython_special',
    'xgboost',
    'lightgbm',
    'catboost',
    'gradio_client'
]

# 3. Add Auto-Collected Modules (Safety Net)
hidden_imports += collect_submodules('pycaret')
hidden_imports += collect_submodules('sklearn')
hidden_imports += collect_submodules('lightgbm')
hidden_imports += collect_submodules('xgboost')

# 4. Collect Data Files (Fixes VERSION.txt error)
datas = []
datas += collect_data_files('pycaret')
datas += collect_data_files('sklearn')
datas += collect_data_files('gradio_client')
datas += collect_data_files('imblearn') 
datas += collect_data_files('lightgbm')
datas += collect_data_files('xgboost')

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PySide6', 'PyQt6', 'PySide2'], 
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='QFit_by_Shayan',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False, 
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='my_icon.ico' 
)