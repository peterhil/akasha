#! /usr/bin/python


# DTMF keypad frequencies (with sound clips)
#   Hz  1209  1336  1477  1633 Hz
#  697     1     2     3     A
#  770     4     5     6     B
#  852     7     8     9     C
#  941     *     0     #     D

class DTMF(object):
    """Dual-tone Multifrequency Tones

    DTMF keypad frequencies:
    
    Hz   1209  1336  1477  1633
    697     1     2     3     A
    770     4     5     6     B
    852     7     8     9     C
    941     *     0     #     D
    
    Special tones:
    
    Event        Low frequency  High frequency
    Busy signal         480 Hz  620 Hz
    Ringback tone (US)  440 Hz  480 Hz
    Dial tone           350 Hz  440 Hz
    """
    sp = [ 350,  440,  480,  620]
    lo = [ 697,  770,  852,  941]
    hi = [1209, 1336, 1477, 1633]    
    
    nkeys = '123A' + \
            '456B' + \
            '789C' + \
            '*0#D'
    
    table = dict()

    for i in xrange(len(nkeys)):
        l, h = divmod(i, 4)
        table[nkeys[i]] = (lo[l], hi[h])

    def __init__(self, string, pulselength=0.07, pause=0.05):
        self.number = string 
        self.pulselength = pulselength
        self.pause = pause
        return self

    def sample(self, iter):
        """Make DTML dialing tone."""
        #snd = Sound()
        #snd.add()
        pass
        
    def __len__(self):
        int(round(len(self.number) * (self.pulselength + self.pause) * Sampler.rate - self.pause))
