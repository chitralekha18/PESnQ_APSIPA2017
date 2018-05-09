form Variables
    sentence filename
    real hop
endform
wavFile$ = filename$ + ".wav"
Read from file... 'wavFile$'

select Sound 'filename$'
To Harmonicity (cc)... hop 75 0.1 4.5
Write to short text file... harmonics


