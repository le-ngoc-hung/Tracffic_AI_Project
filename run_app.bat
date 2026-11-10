@echo off
title He Thong Giam Sat Giao Thong Do Thi & Phat Nguoi ANPR (DACN2)
echo ======================================================================
echo  HE THONG GIAM SAT GIAO THONG DO THI THONG MINH & PHAT NGUOI ANPR
echo  Do An Chuyen Nganh 2 (DACN2)
echo ======================================================================
echo Dang khoi chay he thong, vui long doi giay lat...
echo.

if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe main.py
) else (
    python main.py
)

pause
