# README: Perceptual Evaluation of Singing Quality

Created by: Chitralekha Gupta
Affiliation: NUS, Singapore
Last edited on: 8th May 2018
Last edited by: Chitralekha Gupta

Please refer to the following paper for details:
Gupta, C., Li, H. and Wang, Y., 2017, December, Perceptual Evaluation of Singing Quality. In Proceedings of APSIPA Annual Summit and Conference (Vol. 2017, pp. 12-15).

## Contents
This consists of the following (all used for the paper above):
1. Audio Dataset (folder: WavfileDataset)
2. Subjective ground-truths by music-experts (folder: groundtruthfiles)
3. Python scripts that extract the features for singing quality evaluation (main_file.py)
4. Output arff files generated from the python scripts, that can be directly used in WEKA (folder: output_arffs)

## Dependencies

- This program is designed for monophonic (without background music) audio files recorded at 16000Hz sampling rate. 

- This requires a reference singing and a test singing file, both singing the same song. Typically short singing clips are expected (<30 seconds) for better performance.

- This program also requires Praat executable to be present in the current folder. If the path changes, please change the path 'Praat.app/Contents/MacOS/Praat' in the script get_mfcc_dtw.py'

## How to run?
The python scripts main_file.py and get_mfcc_dtw.py are created by me. 
To run them, start with main_file.py, and follow the instructions in the file header. The main_file.py can run in two modes: single or group (flag: single_or_group). Single mode for you to check the feature values for a single file with respect to a reference file. Group mode is when you want to extract the features for a group of files and dump into an arff file format. In this program, the reference singer for both the songs "Edelweiss" and "I have a dream" is the professional singer named "MCUR".

To use the single mode for your data, change the reference and test audio clips in the folder wavfile_samples.
To use the group mode for your data, change the audio files in the folder WavfileDataset, and indicate the reference singer name in 

## Contact
Chitralekha Gupta: chitralekha[at]u[dot]nus[dot]edu
Haizhou Li: 
Ye Wang: 
