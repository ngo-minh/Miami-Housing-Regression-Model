@echo off
echo Setting up the environment for the GUI application...

:: Optionally, check for Python and install if not found (advanced setup, not included here)

:: Ensure pip, setuptools, and wheel are up to date
python -m pip install --upgrade pip setuptools wheel

:: Install specific packages and versions
python -m pip install numpy
python -m pip install Pillow
python -m pip install joblib
python -m pip install scikit-learn==1.4.2

echo Environment setup is complete. You can now run the GUI application.
pause
