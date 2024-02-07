
# Crappy Voice Modulator (CVM)
## MUS15 Project 1

Benjamin Xia (A16738116)

Tyler Lee (A16976522)

### Description
The Crappy Voice Modulator (CVM) Program allows you to input any .wav file and alter its speed, pitch, and bass. There is also the "Deep Fried Mic" option that makes the .wav file sound like it came from a microphone of terrible quality. The program is runnable in the terminal with commands.

### Directions
1. You to have Python installed in order to run the program.
2. Using any environment of your choice (VSCode, Eclipse, etc.), open a terminal and navigate to where the "cvm.py" file is located.
3. Import a .wav file of your choice (if you are using VSCode, simply drag and drop the file into the folder system).
4. Use the following command:
    ```python cvm.py -f [string] -b [float] -s [float] -p [float] -d [float]```
   
    -f represents the filename; enter it as 'your_filename.wav'
   
    -b represents the bass boost; enter any integer/decimal
   
    -s represents the speed; enter any integer/decimal greater than 0
   
    -p represents the pitch; enter any integer/decimal
   
    -d represents the deep fried mic option; enter any integer/decimal
   

You can use the command with all of the arguments or just a select few.

The only argument that is required is the -f (filename) argument.

For example, if you only want to speed up the track, you can use the following command:
    ```python cvm.py -f 'your_filename.wav' -s 2```

This command should play back the .wav file at 2x the speed.
