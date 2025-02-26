# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# 公共的隐藏导入
hidden_imports = [
    'librosa',
    'librosa.util',
    'librosa.filters',
    'librosa.feature',
    'scipy.signal',
    'sklearn.ensemble._forest',
    'sklearn.tree._utils',
    'sklearn.utils._cython_blas',
    'sklearn.neighbors.typedefs',
    'sklearn.neighbors.quad_tree',
    'sklearn.tree',
    'sklearn.ensemble',
    'soundfile',
    'joblib',
    'numpy',
    'scipy',
    'numba',
]

# 特征提取工具
a1 = Analysis(
    ['feature_extract.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# 事件检测工具
a2 = Analysis(
    ['events_guess.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# 模型训练工具
a3 = Analysis(
    ['train.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz1 = PYZ(a1.pure, a1.zipped_data, cipher=block_cipher)
pyz2 = PYZ(a2.pure, a2.zipped_data, cipher=block_cipher)
pyz3 = PYZ(a3.pure, a3.zipped_data, cipher=block_cipher)

# 特征提取工具
exe1 = EXE(
    pyz1,
    a1.scripts,
    a1.binaries,
    a1.zipfiles,
    a1.datas,
    [],
    name='特征提取工具',
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

# 事件检测工具
exe2 = EXE(
    pyz2,
    a2.scripts,
    a2.binaries,
    a2.zipfiles,
    a2.datas,
    [],
    name='事件检测工具',
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

# 模型训练工具
exe3 = EXE(
    pyz3,
    a3.scripts,
    a3.binaries,
    a3.zipfiles,
    a3.datas,
    [],
    name='模型训练工具',
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