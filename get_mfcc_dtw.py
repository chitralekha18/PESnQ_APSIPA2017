###############
## Created by: Chitralekha Gupta
## Affiliation: NUS, Singapore
## Last edited on: 8th May 2018
## Last edited by: Chitralekha Gupta

## Please refer to the following paper for details:
## Gupta, C., Li, H. and Wang, Y., 2017, December.
## Perceptual Evaluation of Singing Quality.
## In Proceedings of APSIPA Annual Summit and Conference (Vol. 2017, pp. 12-15).
################
## This script is called from the main_file.py
## This python script can also run stand-alone
## Change the two input files on lines 984, 985 - "original" and "test"
## This script computes dtw between mfcc vectors from original and test
## Also gives plots when plot flag in on
################

from mfcc_copy import mfcc
from mfcc_copy import delta
import scipy.io.wavfile as wav
import numpy as np
from fastdtw import fastdtw
from matplotlib import pylab as plt
from itertools import chain
import os
import math
import decimal
import csv
from scipy.spatial.distance import euclidean
from scipy.fftpack import fft

runtime_filedumps = 'runtime_folder'
if not os.path.exists(runtime_filedumps):
    os.mkdir(runtime_filedumps)

def InitialFinalSilenceRemoved(sig):
    energy_thresh = 0.01
    window = 512
    hop = window/2
    energy = []
    i = 0
    energy_index = []
    while i<(len(sig)-window):
        chunk = sig[i:i+window][np.newaxis]
        energy.append(chunk.dot(chunk.T)[0][0])
        energy_index.append(i)
        i = i+hop

    energy = np.array(energy)
    significant_indices = np.where(energy>energy_thresh)[0]
    if significant_indices[0] == 0:
        start_point_sample = 0
    else:
        start_point_sample = (significant_indices[0]-1)*hop
    if significant_indices[-1] == len(energy)-1:
        end_point_sample = len(energy)*hop
    else:
        end_point_sample = (significant_indices[-1]+1)*hop
    new_sig = sig[start_point_sample:end_point_sample+1]
    if plot:
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(range(len(sig)),sig)
        plt.ylabel('amplitude')
        plt.title('Remove initial and final silences')
        plt.subplot(3,1,2)
        plt.plot(energy_index,energy)
        plt.ylabel('energy')
        plt.stem([start_point_sample,end_point_sample],[5,5],'k')
        plt.subplot(3,1,3)
        plt.plot(new_sig)
        plt.ylabel('amplitude')
        plt.xlabel('sample number')
        plt.show()
    return new_sig

def plot_pitch_contour(original_time_pitch,test_time_pitch):
    time_original = original_time_pitch[:, 0]
    pitch_original = original_time_pitch[:, 1]
    time_test = test_time_pitch[:, 0]
    pitch_test = test_time_pitch[:, 1]

    if plot:
        plt.figure()
        plt.plot(time_original, pitch_original)
        plt.plot(time_test, pitch_test,'r.')
        plt.xlabel('time (seconds)')
        plt.ylabel('pitch x 100 (cents)')
        plt.legend(('original','test'))
        plt.title('Pitch Contours')
        plt.show()


def plotalignment(therapist_time_pitch,patient_time_pitch,path):
    offset = 100
    if plot:
        plt.plot(therapist_time_pitch[:, 0], therapist_time_pitch[:, 1] + offset, 'o',label='reference pitch')
        plt.plot(patient_time_pitch[:, 0], patient_time_pitch[:, 1], 'ro', label='test pitch')
        for i, j in path:
            plt.plot((therapist_time_pitch[i,0], patient_time_pitch[j,0]), (therapist_time_pitch[i,1]+offset, patient_time_pitch[j,1]), 'k')
        plt.tick_params(
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            left='off',
            right='off',
            labelbottom='off',
            labelleft = 'off')  # labels along the bottom edge are off
        plt.xlabel('time (sec)')
        plt.legend()
        plt.show()

def median_filtering(x,y,win):
    y_median = []
    x_median = []
    for i in range(len(y)):
        y_median.append(np.median(y[i:i + win]))
        x_median.append(np.median(x[i:i + win]))
    return x_median,y_median

def appendzerostopitch(therapist_time_pitch,file):
    step = therapist_time_pitch[1,0]-therapist_time_pitch[0,0]
    if therapist_time_pitch[0,0]<=step: return therapist_time_pitch
    times = np.arange(0,therapist_time_pitch[0,0],step)
    for i in times[::-1][1:]:
        therapist_time_pitch = np.insert(therapist_time_pitch,0,np.array([i, 0.0]),axis=0)
    fs,data = wav.read(file)
    total_time = 1.0*len(data)/fs
    if total_time - therapist_time_pitch[-1,0] > step:
        times = np.arange(therapist_time_pitch[-1,0],total_time,step)
        for i in times[1:]:
            therapist_time_pitch = np.insert(therapist_time_pitch, -1, np.array([i, 0.0]), axis=0)
    return therapist_time_pitch

def pitch_preprocess(time_pitch):
    time = time_pitch[:,0]
    pitch = time_pitch[:,1]
    if plot:
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(time,pitch)
        plt.title('before pre-processing')

    # thresholding: pitch never going above 400 Hz
    thresh = 500
    pitch = np.asarray(pitch)
    super_threshold_indices = pitch >= thresh
    pitch[super_threshold_indices] = 0
    if plot:
        plt.subplot(3, 1, 2)
        plt.plot(time, pitch)
        plt.title('after thresholding at 500 Hz')


    return np.vstack((time.T,pitch.T)).T

def extract_time_pitch_forVibrato(file):
    f = open(file, 'r')
    obj = csv.reader(f, delimiter=' ')
    cols = []
    for row in obj:
        if len(row)<2: break
        if np.size(cols) == 0:
            cols = [-100.0 if 'undefined' in elem else float(elem) for elem in row]
        else:
            cols = np.vstack((cols, [-100.0 if 'undefined' in elem else float(elem) for elem in row]))

    return cols

def extract_time_pitch(file):
    f = open(file, 'r')
    obj = csv.reader(f, delimiter=' ')
    cols = []
    for row in obj:
        if len(row)<2: break
        if np.size(cols) == 0:
            cols = [-100.0 if 'undefined' in elem else float(elem) for elem in row]
        else:
            cols = np.vstack((cols, [-100.0 if 'undefined' in elem else float(elem) for elem in row]))

    cols_modified = []
    for row in cols:
        if row[1] == -100.0: continue
        # print row
        # cols_modified.append(row)
        if np.size(cols_modified) == 0:
            cols_modified = row[:]
        else:
            cols_modified = np.vstack((cols_modified,row))

    # cols_modified = pitch_preprocess(cols)
    # print cols_modified
    return cols_modified

def extract_pitch(wavfile,pitchfile,hop):
    pitch_ceiling = 650.0
    # Extracting pitch
    outputfile = wavfile.replace('.wav','_pitch.wav')
    # pitch_extract_cmd = 'aubiopitch -i '+wavfile+' -p yin > '+pitchfile
    pitch_extract_cmd = 'Praat.app/Contents/MacOS/Praat --run ExtractPitch.praat '+wavfile+' '+pitchfile+' '+str(pitch_ceiling)+' '+str(hop)
    os.system(pitch_extract_cmd)
    # pdb.set_trace()

def mfcc_dist(a, b):
    dist = 0
    # for x, y in zip(a, b):
    #     dist = dist + (x - y) * (x - y)
    dist = sum((a-b)*(a-b))
    return np.sqrt(dist)

def plot_dtw_matrix(path,title):
    unzippath = zip(*path)
    if plot:
        plt.figure()
        plt.plot(unzippath[0],unzippath[1],linewidth=2)
        plt.plot([0,unzippath[0][-1]],[0,unzippath[1][-1]],'r-.',linewidth=2)
        plt.xlabel('Reference frames',fontsize=22)
        plt.ylabel('Test frames',fontsize=22)
        plt.xlim(0,unzippath[0][-1])
        plt.ylim(0, unzippath[1][-1])
        plt.tick_params(labelsize=18)
        # plt.title(title)
        plt.show()

def plot_dtw_matrix_long(path,cost):
    # print cost
    # plt.imshow(cost)
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    # plt.imshow(cost)
    plt.imshow(cost, cmap='Greys', vmin=np.min(cost), vmax=np.max(cost),
               extent=[0, np.shape(cost)[0], 0, np.shape(cost)[1]],
               interpolation='nearest', origin='lower')
    # ax.set_aspect('equal')
    if plot:
        plt.plot(path[0], path[1], 'r')
        plt.xlim(0, np.shape(cost)[0])
        plt.ylim(0, np.shape(cost)[1])
        plt.plot([0,np.shape(cost)[0]],[0, np.shape(cost)[1]],'g-.')
        plt.xlabel('original')
        plt.ylabel('test')
        plt.show()

def flatten(listOfLists):
    "Flatten one level of nesting"
    return chain.from_iterable(listOfLists)

def adjust_files(ori_sig,test_sig,path,hop,hop_test,fs):
    ori_sig_mod = []
    test_sig_mod = []
    num_samp_per_frame = int(np.floor(hop*fs))
    num_samp_per_frame_test = int(np.floor(hop_test * fs))

    for item in path:
        ori_sig_mod.append(ori_sig[item[0]*num_samp_per_frame:item[0]*num_samp_per_frame+num_samp_per_frame-1])
        test_sig_mod.append(test_sig[item[1] * num_samp_per_frame_test:item[1] * num_samp_per_frame_test + num_samp_per_frame_test - 1])
    ori_sig_mod = np.array(list(flatten(ori_sig_mod)))
    test_sig_mod = np.array(list(flatten(test_sig_mod)))

    return ori_sig_mod,test_sig_mod

def adjust_pitch_tracks(ori_time_pitch,test_time_pitch,path):
    ori_pitch_mod = []
    test_pitch_mod = []
    ori_pitch = ori_time_pitch[:,1]
    test_pitch = test_time_pitch[:,1]

    for item in path:
        ori_pitch_mod.append(ori_pitch[item[0]])
        test_pitch_mod.append(test_pitch[item[1]])
    ori_pitch_mod = np.array(ori_pitch_mod)[np.newaxis].T
    test_pitch_mod = np.array(test_pitch_mod)[np.newaxis].T
    time = np.arange(0.016,0.016+len(ori_pitch_mod)*0.016,0.016)[np.newaxis].T
    if np.shape(time)[0] > np.shape(ori_pitch_mod)[0]: time = time[0:np.shape(ori_pitch_mod)[0]]
    ori_time_pitch_mod = np.hstack((time, ori_pitch_mod))
    test_time_pitch_mod = np.hstack((time, test_pitch_mod))

    return ori_time_pitch_mod,test_time_pitch_mod

def adjust_pitch_tracks2(ori_time_pitch,test_time_pitch):
    ori_pitch_mod = []
    test_pitch_mod = []
    ori_time = ori_time_pitch[:, 0]
    test_time = test_time_pitch[:, 0]

    cnt = 0
    if len(test_time)<len(ori_time):
        for ind in range(len(test_time)):
            if ori_time[cnt]>test_time[ind]: continue #missed pitch point in original
            while test_time[ind] != ori_time[cnt]:
                cnt = cnt+1

            if test_time[ind] == ori_time[cnt]:
                ori_pitch_mod.append(ori_time_pitch[ind,1])
                test_pitch_mod.append(test_time_pitch[ind, 1])
                cnt = cnt+1

    else:
        for ind in range(len(ori_time)):
            if test_time[cnt] > ori_time[ind]: continue  # missed pitch point in original
            while  ori_time[ind] != test_time[cnt]:
                cnt = cnt+1
            if test_time[cnt] == ori_time[ind]:
                ori_pitch_mod.append(ori_time_pitch[ind, 1])
                test_pitch_mod.append(test_time_pitch[ind, 1])
                cnt = cnt+1

    ori_pitch_mod = np.array(ori_pitch_mod)[np.newaxis].T
    test_pitch_mod = np.array(test_pitch_mod)[np.newaxis].T
    time = np.arange(0.01, 0.01 + len(ori_pitch_mod) * 0.01, 0.01)[np.newaxis].T
    if np.shape(time)[0]>np.shape(ori_pitch_mod)[0]: time = time[0:np.shape(ori_pitch_mod)[0]]
    ori_time_pitch_mod = np.hstack((time, ori_pitch_mod))
    test_time_pitch_mod = np.hstack((time, test_pitch_mod))

    return ori_time_pitch_mod, test_time_pitch_mod

def FrameDisturbance(path):
    disturbance = []
    for item in path:
        disturbance.append(item[0]-item[1])
    return disturbance

def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

def PitchDistanceComputation(original_time_pitch, test_time_pitch,title):
    ### Compute DTW between original and test
    distance, path = fastdtw(original_time_pitch, test_time_pitch, dist=euclidean)
    # title = 'DTW between rhythm-adjusted pitch contours'
    plot_dtw_matrix(path, title)

    ### Plot alignment
    # plotalignment(original_time_pitch,test_time_pitch,path)

    ############ pitch frame disturbance array ##############
    pitch_frame_disturbance = FrameDisturbance(path)
    start_frame = 0
    stop_frame = len(pitch_frame_disturbance) - 1

    if print_flag:
        print '\n'+title
        print "Pitch Deviation (L2-norm of the deviations in rhythm-compensated pitch feature vector) = ", np.linalg.norm(
            pitch_frame_disturbance, ord=2)
        print "Pitch Difference (Distance between rhythm-compensated pitch feature vectors) = ", distance
        print "Pitch Deviation (L6-L2-norm of the deviations in rhythm-compensated pitch feature vector) = ", CalcPESQnorm(pitch_frame_disturbance)


    if plot:
        plt.plot(pitch_frame_disturbance)
        plt.xlabel('frames')
        plt.ylabel('pitch dtw frame disturbance')
        plt.title(title)
        plt.show()
    return np.linalg.norm(pitch_frame_disturbance, ord=2), distance, CalcPESQnorm(pitch_frame_disturbance)

def PitchDistanceComputation2(original_time_pitch, test_time_pitch,title):
    ### Compute DTW between original and test
    distance, path = fastdtw(original_time_pitch, test_time_pitch, radius = 1, dist=mfcc_dist)
    # title = 'DTW between rhythm-adjusted pitch contours'
    plot_dtw_matrix(path, title)

    ### Plot alignment
    # plotalignment(original_time_pitch,test_time_pitch,path)

    ############ pitch frame disturbance array ##############
    pitch_frame_disturbance = FrameDisturbance(path)
    start_frame = 0
    stop_frame = len(pitch_frame_disturbance) - 1
    if print_flag:
        print '\n'+title
        print "Pitch Deviation (L2-norm of the deviations in rhythm-compensated pitch feature vector) = ", np.linalg.norm(
            pitch_frame_disturbance, ord=2)
        print "Pitch Difference (Distance between rhythm-compensated pitch feature vectors) = ", distance
        print "Pitch Deviation (L6-L2-norm of the deviations in rhythm-compensated pitch feature vector) = ", CalcPESQnorm(
            pitch_frame_disturbance)

    if plot:
        plt.plot(pitch_frame_disturbance)
        plt.xlabel('frames')
        plt.ylabel('pitch dtw frame disturbance')
        plt.title(title)
        plt.show()
    return np.linalg.norm(pitch_frame_disturbance, ord=2),distance, CalcPESQnorm(pitch_frame_disturbance)

def PitchDerivative(time_pitch,shift):
    time = time_pitch[:, 0]
    pitch = time_pitch[:, 1]
    pitchderivative = []
    for i in range(1,shift+1):
        pitch_shifted = np.hstack([pitch[i:],np.zeros(i)])
        derivative = pitch_shifted-pitch
        if i == 1:
            pitchderivative = derivative[np.newaxis].T
        else:
            pitchderivative = np.hstack((pitchderivative,derivative[np.newaxis].T))
    return pitchderivative

def PitchMedianSubtraction(time_pitch):
    time = time_pitch[:, 0]
    pitch = time_pitch[:, 1]

    median = np.median(pitch)
    pitch_new = pitch-median
    time_pitch_mediansubtracted = np.hstack([time[np.newaxis].T, pitch_new[np.newaxis].T])
    return time_pitch_mediansubtracted

def VibratoBoundariesCalc(time,viblikelihood1,thresh):
    time_sub = time[np.array(viblikelihood1) > thresh]
    if plot:
        plt.plot(time_sub, thresh*np.ones(len(time_sub)),'r.')
        plt.show()

    time_deriv = np.abs(time_sub-np.append(time_sub[1:],0))
    if plot:
        plt.figure()
        plt.plot(time_sub,time_deriv,'r.')
        plt.show()

    timestamps_end = time_sub[time_deriv>0.032][np.newaxis].T
    end_indices = np.where(time_deriv>0.032)[0]
    timestamps_begin = np.append(time_sub[0],time_sub[end_indices[:-1]+1])[np.newaxis].T
    num_samp_per_interval = np.abs((timestamps_begin - timestamps_end)) / 0.016
    begin_t = timestamps_begin[num_samp_per_interval>10.0][np.newaxis].T
    end_t = timestamps_end[num_samp_per_interval>10.0][np.newaxis].T
    return np.hstack((begin_t,end_t))

def VibratoFeatureCalc(vibrato_time_stamps,time_pitch,test_flag=0):
    NFFT = 512
    pitch_hop = 0.016
    f = np.arange(0, 1 / (2 * pitch_hop), (1 / (pitch_hop * NFFT)))
    F_low = 5
    F_high = 8

    time = time_pitch[:,0]
    pitch = time_pitch[:,1]

    vib_features = np.array([])
    vibrato_time_stamps_final = np.array([])

    if np.shape(vibrato_time_stamps) == (2,): # to handle the case of single vibrato segment
        vibrato_time_stamps = vibrato_time_stamps[np.newaxis]
    for ind in range(np.shape(vibrato_time_stamps)[0]):
        features = []
        n1 = np.argmin(np.abs(np.subtract(time, vibrato_time_stamps[ind,0])))
        n2 = np.argmin(np.abs(np.subtract(time, vibrato_time_stamps[ind,1])))
        pitch_snippet = pitch[n1:n2+1]
        time_snippet = time[n1:n2+1]

        pitch_snippet = pitch_snippet - np.mean(pitch_snippet)

        ### Likeliness feature
        X = np.abs(fft(pitch_snippet * np.hamming(len(pitch_snippet)), NFFT))
        Xhalf_norm = X[0:NFFT / 2] / sum(X[0:NFFT / 2])

        vib_region = Xhalf_norm[(f >= F_low) & (f <= F_high)]
        vib_power = sum(vib_region)
        vib_sharp = sum(np.abs(vib_region - np.hstack([vib_region[1:], np.zeros(1)])))
        vib_likeliness_nakano = vib_power * vib_sharp
        features.append(vib_likeliness_nakano)

        ### Rate feature
        zero_crossings = np.where(np.diff(np.sign(pitch_snippet)))[0]
        time_zerocrossings = time_snippet[zero_crossings[0::2]]
        if len(time_zerocrossings) <=1:
            if test_flag == 0:
                if print_flag:
                    print "not enough cycles, thus ignore"
            else:
                if vib_features.size == 0:
                    vib_features = np.array([0.,0.,0.])
                else:
                    vib_features = np.vstack((vib_features, np.array([0.,0.,0.])))

                if vibrato_time_stamps_final.size == 0:
                    vibrato_time_stamps_final = vibrato_time_stamps[ind, :]
                else:
                    vibrato_time_stamps_final = np.vstack((vibrato_time_stamps_final, vibrato_time_stamps[ind, :]))
            continue
        # print np.abs(time_zerocrossings[0:-1]-time_zerocrossings[1:])
        # print time_zerocrossings

        rate = 1.0/((1.0/(len(time_zerocrossings)-1))*sum(np.abs(time_zerocrossings[0:-1]-time_zerocrossings[1:])))
        features.append(rate)

        ### Extent Feature
        pitch_max = []
        for index in range(len(zero_crossings)-1):
            pitch_max.append(max(np.abs(pitch_snippet[zero_crossings[index]:zero_crossings[index+1]])))
        pitch_max = np.array(pitch_max)
        extent =  (0.5/(len(pitch_max)-1))*sum(np.abs(pitch_max[0:-1]+pitch_max[1:]))
        features.append(extent)

        ### feature and time stamp prep
        if vib_features.size == 0:
            vib_features = np.array(features)
        else:
            vib_features = np.vstack((vib_features,np.array(features)))

        if vibrato_time_stamps_final.size == 0:
            vibrato_time_stamps_final = vibrato_time_stamps[ind,:]
        else:
            vibrato_time_stamps_final = np.vstack((vibrato_time_stamps_final,vibrato_time_stamps[ind,:]))
        # print "vib_likeliness = ", vib_likeliness_nakano
        # print "rate = ",rate
        # print "extent = ",extent

        if plot:
            plt.plot(time_snippet,pitch_snippet)
            plt.show()
    return vib_features, vibrato_time_stamps_final

def VibratoDetection(time_pitch):
    pitch_hop = 0.016
    time = time_pitch[:, 0]
    pitch = time_pitch[:, 1]
    win_samples = 32 # 0.016*32 = 512 ms window size
    NFFT = 512
    F_low = 5 #Hz
    F_high = 8 #Hz
    viblikelihood1 = []
    viblikelihood2 = []

    f = np.arange(0,1/(2*pitch_hop),(1/(pitch_hop*NFFT)))

    pitch_zeroappended = np.hstack([pitch,np.zeros(win_samples)])
    for i in range(len(pitch)):
        pitch_snippet = pitch_zeroappended[i:i+win_samples]
        pitch_snippet = pitch_snippet-np.mean(pitch_snippet)
        X = np.abs(fft(pitch_snippet*np.hamming(win_samples),NFFT))
        Xhalf_norm = X[0:NFFT/2]/sum(X[0:NFFT/2])

        vib_region = Xhalf_norm[(f>=5) & (f<=8)]

        ## Interspeech 2006: Nakano: An automatic singing skill evaluation method for unknown melodies using pitch interval and vibrato features
        vib_power = sum(vib_region)
        vib_sharp = sum(np.abs(vib_region-np.hstack([vib_region[1:],np.zeros(1)])))
        vib_likeliness_nakano = vib_power*vib_sharp
        viblikelihood1.append(vib_likeliness_nakano)
        ########

        ## My method
        tot_power_2HzToFsby2 = sum(Xhalf_norm)
        vib_power_ratio = vib_power/tot_power_2HzToFsby2
        viblikelihood2.append(vib_power_ratio)


    if plot:
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(time,pitch)
        plt.ylabel('pitch (cents)')
        plt.subplot(3,1,2)
        plt.plot(time,viblikelihood1)
        plt.ylabel('vibrato likeliness 1')
        plt.subplot(3,1,3)
        plt.plot(time,viblikelihood2)
        plt.ylabel('vibrato likeliness 2')
        plt.xlabel('time (sec)')
        plt.show()

    if plot:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(time, pitch)
        plt.xlim(2,23)
        plt.ylim(-8,0)
        plt.ylabel('Pitch x 100 (cents)')
        plt.subplot(2, 1, 2)
        plt.plot(time, viblikelihood2)
        plt.xlim(2, 23)
        plt.xlabel('Time (s)')
        plt.ylabel('Vibrato likeliness')
        # plt.show()

    vibrato_time_stamps = VibratoBoundariesCalc(time,viblikelihood2,thresh=0.4)
    vib_features, vibrato_time_stamps_final = VibratoFeatureCalc(vibrato_time_stamps,time_pitch)
    return vibrato_time_stamps_final, vib_features

def WriteWavValidPitchFrames(sig,rate,valid_pitch_frames_filename,time_pitch):
    time_ori = np.arange(0,len(sig)*1.0/rate,1.0/rate)
    time = time_pitch[:, 0]

    start = []
    stop = []
    old = 0
    for elem in time:

        diff = np.abs(old-elem)
        if diff>0.0161:
            if old!=0:
                stop.append(old)
            start.append(elem)
            old = elem
            continue
        else:
            old = elem

    stop.append(elem) # last element

    # Pick from original signal
    valid_pitch_sig = []
    for i in range(len(start)):
        start_ori = np.argmin(np.abs(np.subtract(time_ori, start[i])))
        stop_ori = np.argmin(np.abs(np.subtract(time_ori, stop[i])))

        valid_pitch_sig = valid_pitch_sig + (list(sig[start_ori:stop_ori+1]))

    valid_pitch_sig = np.array(valid_pitch_sig)
    wav.write(valid_pitch_frames_filename, rate, np.int16(valid_pitch_sig * 32767))

def CalcMFCC(wavfile):
    (rate, ori_sig) = wav.read(wavfile)
    ori_sig = ori_sig / 32768.0
    ori_sig = ori_sig - np.mean(ori_sig)  # remove DC offset
    window = NFFT / (rate * 1.0)
    hop = window / 2.0
    ori_sig = InitialFinalSilenceRemoved(ori_sig)
    num_ori_frames = int(np.floor(len(ori_sig) / (rate * hop)))  # np.floor

    mfcc_original = mfcc(ori_sig, rate * 1.0, winlen=window, winstep=hop, nfft=NFFT, numcep=13)
    if num_mfcc == 39:
        d_mfcc_feat1 = delta(mfcc_original, 2)
        d_mfcc_feat2 = delta(d_mfcc_feat1, 2)
        mfcc_original = np.hstack((mfcc_original, d_mfcc_feat1, d_mfcc_feat2))
    return mfcc_original

def readPraatShortTextFile(fileName, obj):
    file = open(fileName, "r")
    cnt = 0
    numDataPoints = 0
    offset = 0
    dataX = []
    dataY = []
    dataIdx = 0
    timeStep = 0
    timeOffset = 0

    arrFileTypes = [
        'Harmonicity 2', 'PitchTier', 'Intensity', 'SpectrumTier', \
            'Spectrum 2', 'Cepstrum 1'
    ]

    if not obj in arrFileTypes:
        raise Exception('readPraatShortTextFile - file type must be: '
            + ', '.join(arrFileTypes))
    metaData = []
    for line in file:
        line = line.strip()
        cnt += 1
        #print cnt, line # debug information
        if cnt > 6:
            if obj == 'Harmonicity 2' or obj == 'Intensity 2':
                if cnt > 13:
                    val = float(line)
                    # if val > -100:
                    #     dataY.append(val)
                    # else:
                    #     dataY.append(None)
                    dataY.append(val)
                    dataX.append(timeOffset + float(dataIdx) * timeStep)
                    dataIdx += 1
                else:
                    if cnt == 7:
                        timeStep = float(line)
                    if cnt == 8:
                        timeOffset = float(line)
            else:
            # read data here
                if cnt % 2 == 0:
                    dataY.append(float(line))
                    dataIdx += 1
                else:
                    dataX.append(float(line))
        else:
            if cnt > 3:
                metaData.append(line)
            # error checking and loop initialization
            if cnt == 1:
                if line != "File type = \"ooTextFile\"":
                    raise Exception ("file " + fileName \
                        + " is not a Praat pitch" + " tier file")
            if cnt == 2:
                err = False
                #print line
                if obj == 'Harmonicity':
                    if line != "Object class = \"Harmonicity\"" \
                            and line != "Object class = \"Harmonicity 2\"":
                        err = True
                elif obj == 'Intensity':
                    if line != "Object class = \"IntensityTier\"" \
                            and line != "Object class = \"Intensity 2\"":
                        err = True
                else:
                    if line != "Object class = \"" + obj + "\"":
                        err = True
                if err == True:
                    raise Exception ("file " + fileName + " is not a Praat "
                        + obj + " file")
            if cnt == 6:
                if line[0:15] == 'points: size = ':
                    numDataPoints = int(line.split('=')[1].strip())
                    raise Exception (\
                        "only the 'short text file' type is supported. " \
                        + " Save your Praat " + obj \
                        + " with 'Write to short text file.")
                else:
                    numDataPoints = int(line)
    return (np.array(dataX)[np.newaxis].T, np.array(dataY)[np.newaxis].T, metaData)

def DetectHighPeriodicity(time_pit,pitch,time_per,periodicity_dB,periodocity_thresh):

    frameno = 0
    row = []
    for t in time_pit:
        per_ind = np.argmin(np.abs(np.subtract(time_per, t)))
        if periodicity_dB[per_ind]<10*np.log10(periodocity_thresh*100):
            frameno = frameno+1
            continue #periodicity<0.6 (17.78dB) or 0.7 (18.45dB) => ignore that frame
        if row == []:
            row = np.hstack((time_pit[frameno],pitch[frameno]))
        else:
            row = np.vstack((row,np.hstack((time_pit[frameno],pitch[frameno]))))
        frameno = frameno+1
    if plot:
        # plt.figure()
        # plt.subplot(2,1,1)
        # plt.plot(time_pit,pitch)
        # plt.ylabel('pitch without periodicity detection')
        plt.subplot(3,1,3)
        plt.plot(row[:,0],row[:,1],'.',linewidth=2)
        plt.ylim(-30,-10)
        plt.ylabel('Pitch after periodicity\ndetection x 100 (cents)',fontsize=15)
        plt.xlabel('Time (s)',fontsize=15)
        plt.show()
    return row


def ExtractHighPeriodicityPitchFrames(wavfile,hop,highperiodicity_wavfile,periodocity_thresh):
    pitchfile = runtime_filedumps+os.sep+'pitchfile_beforeperioddet.pitch'

    ##Hack for making the weird requirement of the praat script to function in the current directory ## Can be improved!
    os.system('cp '+wavfile+' .')

    infile = wavfile.rstrip('.wav').split(os.sep)[1]
    harmonic_file = runtime_filedumps+os.sep+'harmonics'
    # Extracting periodicity of frames, output file name is "harmonics"
    harmonicity_extract_cmd = 'Praat.app/Contents/MacOS/Praat --run ExtractHarmonicity.praat ' + infile + ' ' + str(hop)
    os.system(harmonicity_extract_cmd)

    ##Hack for making the weird requirement of the praat script to function in the current directory ## Can be improved!
    os.system('mv harmonics '+ runtime_filedumps)
    os.system('rm -rf '+infile+'.wav')
    # Extracting a column vector of harmonic value in dB for every frame. There is an extra frame at the beginning compared to pitch
    time_per,periodicity_dB,Z = readPraatShortTextFile(harmonic_file,'Harmonicity 2')


    # Extract pitch in a column vector
    extract_pitch(wavfile, pitchfile, hop)
    time_pitch = extract_time_pitch(pitchfile)
    time_pit = time_pitch[:,0][np.newaxis].T
    pitch = time_pitch[:,1][np.newaxis].T
    # print np.shape(pitch)

    #Plot
    if plot:
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(time_pit,pitch,'.',linewidth=2)
        plt.ylim(-30,-10)
        plt.ylabel('Pitch x 100 (cents)',fontsize=15)
        plt.subplot(3,1,2)
        plt.plot(time_per,periodicity_dB)
        plt.ylabel('Periodicity (dB)',fontsize=15)
        # plt.title('(10*log10(x)) where x is between 0 and 100')
        # plt.show()

    # Detect pitch frames with high periodicity
    time_pitch = DetectHighPeriodicity(time_pit,pitch,time_per,periodicity_dB, periodocity_thresh)

    ### Write an audio file with only valid pitch frames
    (rate, sig) = wav.read(wavfile)
    sig = sig / 32768.0
    sig = sig-np.mean(sig)
    WriteWavValidPitchFrames(sig, rate, highperiodicity_wavfile, time_pitch)
    return time_pitch

def VibratoTest(test_time_pitch_oldest,vibrato_time_stamps, vibrato_features_original):
    time = test_time_pitch_oldest[:,0]
    pitch = test_time_pitch_oldest[:,1]
    vibrato_time_stamps_test = np.array([])

    #to tackle the case of single row
    if np.shape(vibrato_time_stamps) == (2,):
        elem1 = time[np.argmin(np.abs(time - vibrato_time_stamps[0]))]
        elem2 = time[np.argmin(np.abs(time - vibrato_time_stamps[1]))]
        if vibrato_time_stamps_test.size == 0:
            vibrato_time_stamps_test = np.array([elem1, elem2])
        else:
            vibrato_time_stamps_test = np.vstack((vibrato_time_stamps_test, np.array([elem1, elem2])))
    # for multiple rows
    else:
        for row in vibrato_time_stamps:
            elem1 = time[np.argmin(np.abs(time-row[0]))]
            elem2 = time[np.argmin(np.abs(time - row[1]))]
            if vibrato_time_stamps_test.size == 0:
                vibrato_time_stamps_test = np.array([elem1,elem2])
            else:
                vibrato_time_stamps_test = np.vstack((vibrato_time_stamps_test,np.array([elem1,elem2])))
    vib_features_test, vibrato_time_stamps_final_test = VibratoFeatureCalc(vibrato_time_stamps_test, test_time_pitch_oldest,test_flag=1)
    return vibrato_time_stamps_final_test, vib_features_test

def TestFileAdjust_forVibrato(original,test,NFFT):
    ### mfcc-based DTW with these new files
    ##Original
    (rate, ori_sig) = wav.read(original)
    ori_sig = ori_sig / 32768.0
    ori_sig = ori_sig - np.mean(ori_sig)  # remove DC offset
    window = NFFT / (rate * 1.0)
    hop = window / 2.0

    mfcc_original = mfcc(ori_sig, rate * 1.0, winlen=window, winstep=hop, nfft=NFFT, numcep=13)

    ## Test
    (rate, test_sig) = wav.read(test)
    test_sig = test_sig / 32768.0
    test_sig = test_sig - np.mean(test_sig)  # remove DC offset
    window = NFFT / (rate * 1.0)
    hop = window / 2.0

    mfcc_test = mfcc(test_sig, rate * 1.0, winlen=window, winstep=hop, nfft=NFFT, numcep=13)

    distance, path = fastdtw(mfcc_original, mfcc_test, radius=1, dist=mfcc_dist)

    ori_sig_mod, test_sig_mod = adjust_files(ori_sig, test_sig, path, hop, hop, rate)
    original_rhythm_compensated = runtime_filedumps+os.sep+'original_rhythm_compensated_forVib.wav'
    test_rhythm_compensated = runtime_filedumps+os.sep+'test_rhythm_compensated_forVib.wav'
    wav.write(original_rhythm_compensated, rate, np.int16(ori_sig_mod * 32767))
    wav.write(test_rhythm_compensated, rate, np.int16(test_sig_mod * 32767))

    original_pitch_file = runtime_filedumps+os.sep+'original_forVib.pitch'
    test_pitch_file = runtime_filedumps+os.sep+'test_forVib.pitch'

    extract_pitch(original_rhythm_compensated, original_pitch_file, hop)
    extract_pitch(test_rhythm_compensated, test_pitch_file, hop)

    original_time_pitch_oldest = extract_time_pitch_forVibrato(original_pitch_file)
    test_time_pitch_oldest = extract_time_pitch_forVibrato(test_pitch_file)

    # DTW on pitch as a final step
    distance, path = fastdtw(original_time_pitch_oldest, test_time_pitch_oldest, dist=euclidean)
    original_time_pitch1, test_time_pitch1 = adjust_pitch_tracks(original_time_pitch_oldest, test_time_pitch_oldest, path)

    vibrato_time_stamps, vibrato_features_original = VibratoDetection(original_time_pitch1)
    vibrato_time_stamps_test, vibrato_features_test = VibratoTest(test_time_pitch1,vibrato_time_stamps, vibrato_features_original)

    return vibrato_features_original, vibrato_features_test

def CalcPESQnorm(frame_disturbance):
    split_sec_interval_frames = 20 #20*0.016 = 320ms
    #L6-norm
    L6norm = []
    for ind in range(0,len(frame_disturbance),split_sec_interval_frames/2):
        frame = frame_disturbance[ind:ind+split_sec_interval_frames]
        L6 = np.power((1.0/len(frame))*np.sum(np.power(frame,6)),1.0/6)
        if math.isnan(L6): continue
        L6norm.append(L6)
    L6L2norm = np.power((1.0/len(L6norm))*np.sum(np.power(L6norm,2)),1.0/2)
    return L6L2norm

def ExtractVibratoFeatures(time_pitch):
    pitch_hop = 0.01
    time = time_pitch[:, 0]
    pitch = time_pitch[:, 1]
    win_samples = 32  # 0.016*32 = 320 ms window size
    NFFT = 512
    F_low = 5  # Hz
    F_high = 8  # Hz
    viblikelihood = []
    rate = []
    extent = []

    f = np.arange(0, 1 / (2 * pitch_hop), (1 / (pitch_hop * NFFT)))

    pitch_zeroappended = np.hstack([pitch, np.zeros(win_samples)])
    for i in range(len(pitch)):
        pitch_snippet = pitch_zeroappended[i:i + win_samples]
        time_snippet = time[i:i + win_samples]
        pitch_snippet = pitch_snippet - np.mean(pitch_snippet)

        ## Interspeech 2006: Nakano: An automatic singing skill evaluation method for unknown melodies using pitch interval and vibrato features
        X = np.abs(fft(pitch_snippet * np.hamming(win_samples), NFFT))
        Xhalf_norm = X[0:NFFT / 2] / sum(X[0:NFFT / 2])
        vib_region = Xhalf_norm[(f >= F_low) & (f <= F_high)]
        vib_power = sum(vib_region)
        vib_sharp = sum(np.abs(vib_region - np.hstack([vib_region[1:], np.zeros(1)])))
        vib_likeliness_nakano = vib_power * vib_sharp
        viblikelihood.append(vib_likeliness_nakano)

        ### Rate feature
        zero_crossings = np.where(np.diff(np.sign(pitch_snippet)))[0]
        time_zerocrossings = time_snippet[zero_crossings[0::2]]
        if len(time_zerocrossings) <= 1:
            rate.append(0)
            extent.append(0)
            continue
        else:
            rate.append(1.0 / ((1.0 / (len(time_zerocrossings) - 1)) * sum(np.abs(time_zerocrossings[0:-1] - time_zerocrossings[1:]))))

        ### Extent Feature
        pitch_max = []
        for index in range(len(zero_crossings) - 1):
            pitch_max.append(max(np.abs(pitch_snippet[zero_crossings[index]:zero_crossings[index + 1]])))
        pitch_max = np.array(pitch_max)
        extent.append((0.5 / (len(pitch_max) - 1)) * sum(np.abs(pitch_max[0:-1] + pitch_max[1:])))

    if plot:
        plt.figure()
        plt.subplot(4, 1, 1)
        plt.plot(time, pitch)
        plt.ylabel('pitch (cents)')
        plt.subplot(4, 1, 2)
        plt.plot(time, viblikelihood)
        plt.ylabel('vibrato likeliness')
        plt.subplot(4, 1, 3)
        plt.plot(time, rate)
        plt.ylabel('rate')
        plt.subplot(4, 1, 4)
        plt.plot(time, extent)
        plt.ylabel('extent')
        plt.show()

    return np.vstack((np.array(viblikelihood)[np.newaxis].T,np.array(rate)[np.newaxis].T,np.array(extent)[np.newaxis].T))

def ComputeLogEnergy(sig,win,hop):
    logenergy = []
    for ind in range(0,len(sig),hop):
        snippet =sig[ind:ind+win]
        logenergy.append(10*np.log10(sum(snippet*snippet)))
    return logenergy

def VolumeDistance(ori_sig,test_sig,rate):
    win = int(0.03*rate)
    hop = int(win/2)

    logenergy_ori = ComputeLogEnergy(ori_sig,win,hop)
    logenergy_test = ComputeLogEnergy(test_sig,win,hop)

    distance, path_vol = fastdtw(logenergy_ori, logenergy_test, radius=1, dist=euclidean)
    return distance

def PitchDynamicRangeCompute(original_time_pitch,test_time_pitch):
    ori_pitch = original_time_pitch[:,1]
    test_pitch = test_time_pitch[:,1]
    ori_range = (max(ori_pitch)-min(ori_pitch))
    test_range = (max(test_pitch)-min(test_pitch))
    return ori_range-test_range

def EmolinaDistance(path):
    ## E. Molina, I. Barbancho, E. Gomez, A. M. Barbancho, and
    # L. J. Tardon, "Fundamental frequency alignment vs. note-based
    # melodic similarity for singing voice assessment," IEEE ICASSP, pp.
    # 744-748, 2013.
    x = []
    y = []
    for item in path:
        x.append(item[0])
        y.append(item[1])

    A = np.vstack([np.array(x), np.ones(len(x))]).T
    sol,res, rank, singular = np.linalg.lstsq(A, np.array(y))
    plot_dtw_matrix(path,'Molina\'s rhythm distance')

    return res[0]

def EmolinaRhythm_mfcc(ori_sig,test_sig,rate, win):
    mfcc_original = mfcc(ori_sig, rate * 1.0, winlen=win, winstep=win/2, nfft=NFFT, numcep=13)
    mfcc_test = mfcc(test_sig, rate * 1.0, winlen=win, winstep=win / 2, nfft=NFFT, numcep=13)

    distance, path = fastdtw(mfcc_original, mfcc_test, radius=1, dist=mfcc_dist)
    emolina_rhythm_dist = EmolinaDistance(path)

    return emolina_rhythm_dist

def EmolinaRhythm_pitch(original_time_pitch, test_time_pitch, rate, window):
    distance, path = fastdtw(original_time_pitch, test_time_pitch, dist=euclidean)
    emolina_rhythm_dist = EmolinaDistance(path)
    return emolina_rhythm_dist

####################################################################################################################
# original = '../Data/wavfiles/good_clip.wav'
# test = '../Data/wavfiles/bad_clip.wav'
num_mfcc = 13
NFFT = 512
shift = 1
plot = 0
print_flag = 0

def get_features(original,test):
    raw_disturbance_features = []
    perceptual_disturbance_features = []
    distance_features = []

    ######### Original file reading and mfcc computation ##########
    (rate,ori_sig) = wav.read(original)
    ori_sig = ori_sig/32768.0
    ori_sig = ori_sig-np.mean(ori_sig) # remove DC offset
    window = NFFT/(rate*1.0)
    hop = window/2.0
    ori_sig = InitialFinalSilenceRemoved(ori_sig)
    num_ori_frames = int(np.floor(len(ori_sig)/(rate*hop)))

    mfcc_original = mfcc(ori_sig,rate*1.0,winlen=window,winstep=hop,nfft=NFFT, numcep=13)
    if num_mfcc == 39:
        d_mfcc_feat1 = delta(mfcc_original, 2)
        d_mfcc_feat2 = delta(d_mfcc_feat1, 2)
        mfcc_original = np.hstack((mfcc_original,d_mfcc_feat1,d_mfcc_feat2))

    ######### Test file reading and mfcc computation ##########
    (rate, test_sig) = wav.read(test)
    test_sig = test_sig/32768.0
    test_sig = test_sig-np.mean(test_sig) #remove DC offset
    test_sig = InitialFinalSilenceRemoved(test_sig)
    window = NFFT/(rate*1.0)
    Nh = np.ceil((len(test_sig)-NFFT)/(num_ori_frames-1)) # number of samples in a hop or frame shift # Make the two files of (approx) equal number of frames
    hop_test = Nh/rate # hop in time #window/2.0 #

    mfcc_test = mfcc(test_sig, rate * 1.0, winlen=window, winstep=hop_test, nfft=NFFT, numcep=13) #test sig truncated by hop size because overlap is now different from 50%
    if num_mfcc == 39:
        d_mfcc_feat1 = delta(mfcc_test, 2)
        d_mfcc_feat2 = delta(mfcc_test, 2)
        mfcc_test = np.hstack((mfcc_test,d_mfcc_feat1,d_mfcc_feat2))

    ########## Compute DTW on mfcc ################
    distance, path = fastdtw(mfcc_original, mfcc_test, radius = 1, dist=mfcc_dist)
    title = 'DTW between length-adjusted MFCCs'

    plot_dtw_matrix(path,title)

    ############ MFCC frame disturbance array ##############
    mfcc_frame_disturbance = FrameDisturbance(path)
    start_frame = 0
    stop_frame = len(mfcc_frame_disturbance)-1

    if print_flag:
        print "Rhythm Disturbance (L2-norm of the deviations in length-equalized mfcc feature vector) = ",np.linalg.norm(mfcc_frame_disturbance, ord=2)
        print "Timbral Difference (Distance between length-equalized mfcc feature vectors) = ",distance
        print "Rhythm Disturbance (L6+L2-norm of the deviations in length-equalized mfcc feature vector) = ",CalcPESQnorm(mfcc_frame_disturbance)
    raw_disturbance_features.append(np.linalg.norm(mfcc_frame_disturbance, ord=2))
    distance_features.append(distance)
    perceptual_disturbance_features.append(CalcPESQnorm(mfcc_frame_disturbance))

    if plot:
        plt.figure()
        plt.plot(mfcc_frame_disturbance,linewidth=2)
        plt.ylim(-90, 5)
        plt.xlabel('Frames',fontsize=22)
        plt.ylabel('MFCC dtw frame disturbance',fontsize=22)
        plt.tick_params(labelsize=18)
        plt.show()

    #############EMOLINA's Rhythm Calc#####################
    ## E. Molina, I. Barbancho, E. Gomez, A. M. Barbancho, and
    # L. J. Tardon, "Fundamental frequency alignment vs. note-based
    # melodic similarity for singing voice assessment," IEEE ICASSP, pp.
    # 744-748, 2013.
    emolina_rhythm_mfcc_distance = EmolinaRhythm_mfcc(ori_sig,test_sig,rate, window)

    #############VOLUME####################################
    volume_dist = VolumeDistance(ori_sig,test_sig,rate)

    ############# PITCH PROCESSING ########################
    # ### Compensate for rhythm/making the two files with equal number samples
    # extract pitch from actual tracks, create an audio with only
    # those frame that contain valid pitch, and then apply dtw using this modified audio track,
    # and then extract final pitch for further processing.
    ###########################################################################################
    ### Extract Pitch Using Praat tool (autocorrelation-based pitch extraction)
    original_aftsilremov = runtime_filedumps+os.sep+'ori_sig_aftsilrem.wav'
    test_aftsilremov = runtime_filedumps+os.sep+'test_sig_aftsilrem.wav'

    wav.write(original_aftsilremov,rate,np.int16(ori_sig*32767))
    wav.write(test_aftsilremov,rate,np.int16(test_sig*32767))

    original_pitch_file = runtime_filedumps+os.sep+'original.pitch'
    test_pitch_file = runtime_filedumps+os.sep+'test.pitch'

    extract_pitch(original_aftsilremov,original_pitch_file,hop)
    extract_pitch(test_aftsilremov,test_pitch_file,hop)

    original_time_pitch_oldest = extract_time_pitch(original_pitch_file)
    test_time_pitch_oldest = extract_time_pitch(test_pitch_file)

    ### Write an audio file with only valid pitch frames
    original_valid_pitch_frames = runtime_filedumps+os.sep+'original_valid_pitch.wav'
    test_valid_pitch_frames = runtime_filedumps+os.sep+'test_valid_pitch.wav'
    WriteWavValidPitchFrames(ori_sig,rate,original_valid_pitch_frames,original_time_pitch_oldest)
    WriteWavValidPitchFrames(test_sig,rate,test_valid_pitch_frames,test_time_pitch_oldest)

    ### Remove low periodicity pitch frames
    original_highperiodicity = original_valid_pitch_frames.rstrip('.wav')+'_highperiodicity.wav'
    test_highperiodicity = test_valid_pitch_frames.rstrip('.wav')+'_highperiodicity.wav'
    original_time_pitch_validHighPer = ExtractHighPeriodicityPitchFrames(original_valid_pitch_frames, hop,original_highperiodicity,0.2)
    test_time_pitch_validHighPer = ExtractHighPeriodicityPitchFrames(test_valid_pitch_frames, hop, test_highperiodicity,0.5)

    ### E. Molina's rhythm distance based on pitch
    ## E. Molina, I. Barbancho, E. Gomez, A. M. Barbancho, and
    # L. J. Tardon, "Fundamental frequency alignment vs. note-based
    # melodic similarity for singing voice assessment," IEEE ICASSP, pp.
    # 744-748, 2013.
    emolina_rhythm_pitch_distance = EmolinaRhythm_pitch(original_time_pitch_validHighPer, test_time_pitch_validHighPer, rate, window)

    ### mfcc-based DTW with these new files
    ##Original
    (rate, ori_sig) = wav.read(original_highperiodicity)
    ori_sig = ori_sig / 32768.0
    ori_sig = ori_sig - np.mean(ori_sig)  # remove DC offset
    window = NFFT / (rate * 1.0)
    hop = window / 2.0

    mfcc_original = mfcc(ori_sig, rate * 1.0, winlen=window, winstep=hop, nfft=NFFT, numcep=13)

    ## Test
    (rate, test_sig) = wav.read(test_highperiodicity)
    test_sig = test_sig / 32768.0
    test_sig = test_sig - np.mean(test_sig)  # remove DC offset
    window = NFFT / (rate * 1.0)
    hop = window / 2.0

    mfcc_test = mfcc(test_sig, rate * 1.0, winlen=window, winstep=hop, nfft=NFFT, numcep=13)

    distance, path = fastdtw(mfcc_original, mfcc_test, radius = 1, dist=mfcc_dist)
    title = 'DTW between raw MFCCs'

    ori_sig_mod, test_sig_mod = adjust_files(ori_sig,test_sig,path,hop,hop,rate)
    original_rhythm_compensated = runtime_filedumps+os.sep+'original_rhythm_compensated.wav'
    test_rhythm_compensated = runtime_filedumps+os.sep+'test_rhythm_compensated.wav'
    wav.write(original_rhythm_compensated,rate,np.int16(ori_sig_mod*32767))
    wav.write(test_rhythm_compensated,rate,np.int16(test_sig_mod*32767))

    ### Extract Pitch Using Praat tool (autocorrelation-based pitch extraction)
    original_pitch_file = runtime_filedumps+os.sep+'original_comp.pitch'
    test_pitch_file = runtime_filedumps+os.sep+'test_comp.pitch'
    extract_pitch(original_rhythm_compensated,original_pitch_file,0.01)
    extract_pitch(test_rhythm_compensated,test_pitch_file,0.01)

    original_time_pitch_old = extract_time_pitch(original_pitch_file)
    test_time_pitch_old = extract_time_pitch(test_pitch_file)

    original_time_pitch, test_time_pitch = adjust_pitch_tracks2(original_time_pitch_old,test_time_pitch_old)

    ### Pitch Plots
    plot_pitch_contour(original_time_pitch,test_time_pitch)

    ### Pitch distance computation
    disturbance,dist,pdisturbance = PitchDistanceComputation(original_time_pitch, test_time_pitch,'Raw Pitch Distance')
    raw_disturbance_features.append(disturbance)
    distance_features.append(dist)
    perceptual_disturbance_features.append(pdisturbance)
    ############ key compensation for pitch - method 1: Derivative ##############

    original_pitchderivative = PitchDerivative(original_time_pitch,shift)
    test_pitchderivative = PitchDerivative(test_time_pitch,shift)
    if plot:
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(original_time_pitch[:,0],original_time_pitch[:,1])
        plt.ylabel('Reference pitch contour\nx 100 (cents)')
        plt.subplot(2,1,2)
        plt.plot(original_time_pitch[:,0],original_pitchderivative)
        plt.ylabel('One frame pitch derivative\nx 100 (cents)')
        plt.xlabel('Time (s)')
        plt.show()
    disturbance, dist, pdisturbance = PitchDistanceComputation2(original_pitchderivative, test_pitchderivative, 'Pitch Derivative distance')
    raw_disturbance_features.append(disturbance)
    distance_features.append(dist)
    perceptual_disturbance_features.append(pdisturbance)

    ############ key compensation for pitch - method 2: median Subtraction ##############
    original_time_pitchmediansubtracted = PitchMedianSubtraction(original_time_pitch)
    test_time_pitchmediansubtracted = PitchMedianSubtraction(test_time_pitch)

    plot_pitch_contour(original_time_pitchmediansubtracted,test_time_pitchmediansubtracted)
    if plot:
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(original_time_pitch[:,0],original_time_pitch[:,1])
        plt.plot(original_time_pitch[:,0],np.zeros(len(original_time_pitch[:,0])),'--r')
        plt.ylabel('Reference pitch contour\nx 100 (cents)')
        plt.subplot(2,1,2)
        plt.plot(original_time_pitchmediansubtracted[:,0],original_time_pitchmediansubtracted[:,1])
        plt.plot(original_time_pitchmediansubtracted[:, 0], np.zeros(len(original_time_pitchmediansubtracted[:, 0])), '--r')
        plt.ylabel('Median-subtracted pitch\ncontour x 100 (cents)')
        plt.xlabel('Time (s)')
        plt.show()

    disturbance, dist, pdisturbance = PitchDistanceComputation(original_time_pitchmediansubtracted, test_time_pitchmediansubtracted, 'Pitch Median subtracted distance')
    raw_disturbance_features.append(disturbance)
    distance_features.append(dist)
    perceptual_disturbance_features.append(pdisturbance)

    ############# Pitch Dynamic Range ###############
    pitch_dynamic_range_dist = PitchDynamicRangeCompute(original_time_pitch,test_time_pitch)

    ########### Vibrato detection and evaluation #################
    vibrato_features_original, vibrato_features_test = TestFileAdjust_forVibrato(original,test,NFFT)
    if np.shape(vibrato_features_original) == (3,):
        vibrato_features_original = vibrato_features_original[np.newaxis]
        vibrato_features_test = vibrato_features_test[np.newaxis]
    distance, path = fastdtw(vibrato_features_original, vibrato_features_test, radius = 1, dist=mfcc_dist)
    plot_dtw_matrix(path,'Vibrato DTW')
    vibrato_section_disturbance = FrameDisturbance(path)
    start_frame = 0
    stop_frame = len(vibrato_section_disturbance)-1
    if print_flag:
        print "Vibrato Segment Disturbance (L2-norm of the deviations in length-equalized vibrato feature vector) = ",np.linalg.norm(vibrato_section_disturbance, ord=2)
        print "Vibrato Segment Difference (Distance between length-equalized vibrato feature vectors) = ",distance

    distance_features.append(distance)

    if plot:
        plt.figure()
        plt.plot(mfcc_frame_disturbance)
        plt.xlabel('frames')
        plt.ylabel('vibrato feature dtw frame disturbance')
        plt.show()

    ###### Vibrato evaluation - frame-based ####### NOT USED IN THE PAPER: CONCEPTUALLY THIS IS A NOISY AND UNRELIABLE FEATURE
    vibrato_features_original = ExtractVibratoFeatures(original_time_pitch)
    vibrato_features_test = ExtractVibratoFeatures(test_time_pitch)
    distance, path = fastdtw(vibrato_features_original, vibrato_features_test, radius = 1, dist=mfcc_dist)
    plot_dtw_matrix(path,'Vibrato DTW')
    vibrato_section_disturbance = FrameDisturbance(path)
    start_frame = 0
    stop_frame = len(vibrato_section_disturbance)-1
    if print_flag:
        print "Vibrato Disturbance (frame-based) (L2-norm of the deviations in length-equalized vibrato feature vector) = ",np.linalg.norm(vibrato_section_disturbance, ord=2)
        print "Vibrato Difference (frame-based) (Distance between length-equalized vibrato feature vectors) = ",distance
        print "Vibrato Disturbance (frame-based) (L2+L6-norm of the deviations in length-equalized vibrato feature vector) = ", CalcPESQnorm(vibrato_section_disturbance)

    if plot:
        plt.figure()
        plt.plot(vibrato_section_disturbance)
        plt.xlabel('frames')
        plt.ylabel('vibrato feature dtw frame disturbance')
        plt.show()
    raw_disturbance_features.append(np.linalg.norm(vibrato_section_disturbance, ord=2))
    perceptual_disturbance_features.append(CalcPESQnorm(vibrato_section_disturbance))
    distance_features.append(distance)

    ### Append Volume Distance Feature
    distance_features.append(volume_dist)

    ### Pitch Dynamic Range
    distance_features.append(pitch_dynamic_range_dist)

    ### Emolina's rhythm distance
    distance_features.append(emolina_rhythm_mfcc_distance)
    distance_features.append(emolina_rhythm_pitch_distance)

    ## Final Features contain:
    ## raw_disturbance_features
    # 1. Rhythm Disturbance (L2-norm)
    # 2. Raw Pitch Disturbance (L2-norm)
    # 3. Pitch-derivative Disturbance (L2-norm)
    # 4. Pitch Median subtracted Disturbance (L2-norm)
    # 5. Vibrato-frame-based Disturbance (L2-norm)

    ## perceptual_disturbance_features
    # 1. Rhythm Disturbance (L6+L2-norm)
    # 2. Raw Pitch Disturbance (L6+L2-norm)
    # 3. Pitch-derivative Disturbance (L6+L2-norm)
    # 4. Pitch Median subtracted Disturbance (L6+L2-norm)
    # 5. Vibrato-frame-based Disturbance (L6+L2-norm)

    ## distance_features
    # 1. Timbral Difference (Distance)
    # 2. Raw Pitch Difference (Distance)
    # 3. Pitch-derivative Difference (Distance)
    # 4. Pitch-median subtracted Difference (Distance)
    # 5. Vibrato-segment difference (Distance)
    # 6. Vibrato: whole pitch contour frame-based evaluation (Distance)
    # 7. Volume Distance
    # 8. Pitch Dynamic Range
    # 9. E.Molina's Rhythm Distance based on mfcc
    # 10. E.Molina's Rhythm Distance based on pitch

    return np.array(raw_disturbance_features)[np.newaxis],np.array(perceptual_disturbance_features)[np.newaxis],np.array(distance_features)[np.newaxis]


