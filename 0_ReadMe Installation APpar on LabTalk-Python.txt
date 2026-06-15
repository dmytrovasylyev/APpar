APpar.ogs
APpar.py

ATTN: all input values of APs (time and voltage) must be in seconds and Volts

See APpar user manual (pdf file) for details

==================================================================================================This part needs to be done only once per PC life
// OriginPro2025 or Origin2022b or later:
// Open from Origin menu: Connectivity - Python Packages
// Install the following packages:
numpy
scipy
pandas

// Open from Origin menu: Connectivity - Python Console
import sys
//ENTER
print(sys.executable)
//ENTER
import subprocess
//ENTER
subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "scipy", "pandas"])
//ENTER

//re-start Origin
==================================================================================================This part needs to be done only once per PC life


==================================================================================================Edit the path and copy files
//APpar.ogs
// Wrapper to run APpar.py from Origin with the same settings dialog style.
// Attn: in the file APpar.ogs replace dv93 with your username to update Python script location on your PC:
string pyFile$ = "C:\\Users\\dv93\\Documents\\OriginLab\\User Files\\APpar.py";

//APpar.ogs and APpar.py
// Attn: Place APpar.ogs and APpar.py files into the following directory in your PC (replace dv93 with your username):
C:\Users\dv93\Documents\OriginLab\User Files

//re-start Origin
==================================================================================================Edit the path and copy files

==================================================================================================Run APpar program
// Make sure Book/spreadsheet with the AP traces to be calculated is active (click on it)
// Open LabTalk Command Window from Origin menu: Window - Command Window or (Alt+3)
//Type APpar in the LabTalk Command Window and hit ENTER
==================================================================================================Run APpar program
