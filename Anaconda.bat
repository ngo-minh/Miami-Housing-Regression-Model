@echo off
CALL %USERPROFILE%\Anaconda3\Scripts\activate.bat %USERPROFILE%\Anaconda3
conda config --add channels conda-forge
conda install scikit-learn=1.4.2 -y
@echo Installation complete. Press any key to exit.
pause >nul
