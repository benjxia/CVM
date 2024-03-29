
# Crappy Voice Modulator and Autotuner (CVM)
## MUS15 Project 1/2

Benjamin Xia (A16738116)

Tyler Lee (A16976522)

### Description
The Crappy Voice Modulator (CVM) Program allows you to input any .wav file and alter its speed, pitch, and bass. There is also the "Deep Fried Mic" option that makes the .wav file sound like it came from a microphone of terrible quality. The program is runnable in the terminal with commands.

In project 2, we added autotune functionality to the program.

### Directions
1. You must have Python installed in order to run the program.
2. Click on the green "Code" button in GitHub and click "Download ZIP." Extract the "CVM" folder somewhere to your desktop.
3. Open the "CVM" folder. Import any .wav file of your choice by dragging and dropping the file into the "CVM" folder. 
4. Open a terminal in the "CVM" folder
    <br/>Windows: type "cmd" into the path bar
    <br/>Mac: control-click the path bar, hover over "Open a new window" and choose "Open in terminal" (untested as we didn't use a Mac; google how to open a terminal in a folder if this doesn't work)
6. Install the required libraries by typing in the following command. <br/>
    ```pip install -r requirements.txt```
7. Finally, to use the program, run the following command and fill in the brackets with your desired values: <br/>
    ```python cvm.py -f [string] -a [string] -b [float] -s [float] -p [float] -d [int] -o [string]```
    <br/>-f represents the filename; enter it as `your_filename.wav`
    <br/>-a represents the scale you want to autotune your voice against; enter it as `TONIC:key` (e.g. C:maj for C major, C:min for C minor)
    <br/>-b represents the bass boost; enter any integer/decimal value
    <br/>-s represents the speed; enter any integer/decimal value greater than 0
    <br/>-p represents the pitch; enter any integer/decimal value
    <br/>-d represents the deep fried mic option; enter any integer value greater than 0
    <br/>-o represents the file to output your edited audio to; enter it as `output_filename.wav`
   
You can use the command with all of the arguments or just a select few. <br/>
The only argument that is required is the -f (filename) argument. <br/>
For example, if you only want to speed up the track, you can use the following command: <br/>
    ```python cvm.py -f 'your_filename.wav' -s 2``` <br/>
This command should play back the .wav file at 2x the speed.
