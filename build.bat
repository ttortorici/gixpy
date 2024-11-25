REM Activate the Conda environment for Python 3.9 and build
call conda activate py309
python -m build
call conda deactivate

REM Activate the Conda environment for Python 3.10 and build
call conda activate py310
python -m build
call conda deactivate

REM Activate the Conda environment for Python 3.11 and build
call conda activate py311
python -m build
call conda deactivate

REM Activate the Conda environment for Python 3.12 and build
call conda activate py312
python -m build
call conda deactivate

REM Activate the Conda environment for Python 3.13 and build
call conda activate py313
python -m build
call conda deactivate

REM Pause to keep the window open
pause