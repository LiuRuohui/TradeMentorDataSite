# -*- mode: python ; coding: utf-8 -*-
import os
import sys
import akshare
import py_mini_racer

block_cipher = None

# 获取路径
akshare_path = os.path.dirname(akshare.__file__)
mini_racer_path = os.path.dirname(py_mini_racer.__file__)

a = Analysis(
    ['stock_analyze.py'],
    pathex=['D:\\ws\\akshare_stock_analysis'],
    binaries=[
        (os.path.join(mini_racer_path, 'mini_racer.dll'), '.'),
        (os.path.join(mini_racer_path, 'icudtl.dat'), '.'),
        (os.path.join(mini_racer_path, 'snapshot_blob.bin'), '.')  # Add snapshot_blob.bin file
    ],
    datas=[
        (akshare_path, 'akshare'),
        (os.path.join(akshare_path, 'stock_feature'), 'akshare/stock_feature'),
        (os.path.join(akshare_path, 'utils'), 'akshare/utils'),
    ],
    hiddenimports=[
        'concurrent.futures',
        'concurrent.futures.thread',
        'concurrent.futures.process',
        'concurrent.futures._base',
        'threading',
        '_thread',
        'queue',
        'multiprocessing',
        'multiprocessing.pool',
        'multiprocessing.context',
        'multiprocessing.popen_spawn_win32',
        'multiprocessing.synchronize',
        'multiprocessing.heap',
        'numpy',
        'pandas',
        'matplotlib',
        'mplfinance',
        'tqdm',
        'matplotlib.backends.backend_tkagg',
        'requests',
        'bs4',
        'lxml',
        'urllib3',
        'py_mini_racer',
        'execjs',
        'html5lib',
        'pyquery',
        'jsonpath',
        'tabulate',
        'httpx',
        'aiohttp',
        'akshare.stock_feature.stock_zh_a_spot_em',
        'akshare.stock_feature.stock_zh_a_hist',
        'akshare.utils.request',
        'akshare.utils',
        'akshare.stock_feature',
        'akshare.pro',
        'akshare.index',
        'akshare.futures',
        'akshare.option',
        'akshare.bond',
        'akshare.fx',
        'akshare.stock'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='stock_analyze',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)