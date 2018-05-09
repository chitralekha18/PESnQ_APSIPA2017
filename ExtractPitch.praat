form Variables
    sentence filename
    sentence outputfilename
	real ceiling
	real hop
endform
deleteFile: outputfilename$
Read from file... 'filename$'

tmin = Get start time
tmax = Get end time
To Pitch (ac): 'hop', 75, 15, "on", 0.05, 0.5, 0.01, 0.35, 0.15, 'ceiling'
# hop, floor (Hz), num candidates, very accurate, silence threshold, voicing threshold, octave cost, octave-jump cost, voiced/unvoiced cost, ceiling
frames = Get number of frames

output$ = ""

for f from 1 to frames
    t = Get time from frame number... 'f'
    t$ = fixed$(t, 3)
    pitch = Get value at time: 't', "semitones re 440 Hz", "Linear"
    pitch$ = fixed$(pitch, 2)
	output$ = output$+t$+" "+pitch$+newline$
	
endfor

appendFileLine: outputfilename$, output$