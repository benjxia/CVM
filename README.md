
# Crappy Voice Modulator (CVM)
## MUS15 Project 1

Benjamin Xia (A16738116)

Tyler Lee (A16976522)

### Description
The Crappy Voice Modulator (CVM) Program allows you to input any .wav file and alter its speed, pitch, and bass. There is also the "Deep Fried Mic" option that makes the .wav file sound like it came from a microphone of terrible quality. The program is runnable in the terminal with commands.

### Directions
1. You must have Python installed in order to run the program.
2. Click on the green "Code" button in GitHub and click "Donwload ZIP." Extract the "CVM" folder somewhere to your desktop.
3. Open the "CVM" folder. Import any .wav file of your choice by dragging and dropping the file into the "CVM" folder. 
4. Open a terminal in the "CVM" folder (Windows: type "cmd" into the path bar)
    Windows: type "cmd" into the path bar
    Mac: control-click the path bar, hover over "Open a new window" and choose "Open in terminal" (untested as we didn't use a Mac; google how to open a terminal in a folder if this doesn't work)
5. Install the required libraries by typing in the following command.
    ```pip install -r requirements.txt```
6. Finally, to use the program, run the following command and fill in the brackets with your desired values:
    ```python cvm.py -f [string] -b [float] -s [float] -p [float] -d [int]```
   
    -f represents the filename; enter it as 'your_filename.wav'
   
    -b represents the bass boost; enter any integer/decimal value
   
    -s represents the speed; enter any integer/decimal value greater than 0
   
    -p represents the pitch; enter any integer/decimal value
   
    -d represents the deep fried mic option; enter any integer value greater than 0
   

You can use the command with all of the arguments or just a select few.

The only argument that is required is the -f (filename) argument.

For example, if you only want to speed up the track, you can use the following command:
    ```python cvm.py -f 'your_filename.wav' -s 2```

This command should play back the .wav file at 2x the speed.