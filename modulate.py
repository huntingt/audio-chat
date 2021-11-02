import numpy as np

PORCH = 15
PREFIX = np.array([1.]*PORCH + [0., 1., 1., 0.], dtype=complex)
SUFFIX = np.array([1.]*PORCH, dtype=complex)
END_DETECT = PORCH // 2
SYMBOLS = 10
BETA = 0.3
F_CRCT = 1

A_MAX = 0.5 #2 ** (-0.5)
MK_MAP = [
    A_MAX/3.,
    A_MAX,
    -A_MAX/3.,
    -A_MAX
]

@np.vectorize
def qam16_encode(bits):
    m = bits & 0b11
    k = (bits >> 2) & 0b11
    return complex(MK_MAP[k], MK_MAP[m])

def qam16_classify(num):
    if num >= 0:
        if num < A_MAX*2./3:
            return 0b00
        else:
            return 0b01
    else:
        if num > -A_MAX*2./3:
            return 0b10
        else:
            return 0b11

@np.vectorize
def qam16_decode(num):
    return (qam16_classify(num.real) << 2) | qam16_classify(num.imag)

@np.vectorize
def root_raised_cos_impulse(t, beta):
    if t == 0:
        return 1 + beta*(4/np.pi - 1)
    elif abs(4 * beta * t) == 1:
        return beta/np.sqrt(2) * (
            (1 + 2/np.pi)*np.sin(np.pi/(4*beta)) +
            (1 - 2/np.pi)*np.cos(np.pi/(4*beta))
        )
    else:
        return (
            np.sin(np.pi*t*(1 - beta)) +
            4*beta*t*np.cos(np.pi*t*(1 + beta))
        ) / (np.pi*t * (1 - (4*beta*t)**2))

def root_raised_cos_filter(symbols, oversample, beta):
    N = symbols * oversample
    t = np.linspace(-symbols/2., symbols/2., N)
    return root_raised_cos_impulse(t, beta) / oversample

THRESHOLD = 0.1
class Calibrator:
    def __init__(self, oversample):
        self.mode = "standby"
        self.oversample = oversample

    def step(self, x):
        a = abs(x)
        if self.mode == "standby":
            if a > THRESHOLD:
                self.mode = "scale"
                self.buffer = [x]
                self.max = a
        elif self.mode == "scale":
            if a > 0.5 * self.max:
                self.max = max(self.max, a)
                self.buffer.append(x)
            else:
                d = len(self.buffer) // 4
                buf = np.array(self.buffer[d:len(self.buffer)-d], dtype=complex)
                if len(buf) == 0 or len(self.buffer) < 0.5 * PORCH * self.oversample:
                    self.mode = "standby"
                else:
                    self.scale = np.average(buf)
                    self.buffer = [a]
                    self.mode = "time"
        elif self.mode == "time":
            if a < 0.5 * self.max:
                self.buffer.append(a)
            else:
                p = np.polyfit(np.arange(len(self.buffer)), self.buffer, 2)
                center = len(self.buffer)/2. if p[0] == 0 else -p[1]/(2*p[0])
                self.time = int(center) - len(self.buffer) + 4 *\
                    self.oversample - 1
                self.mode = "delay"
        elif self.mode == "delay":
            if self.time > 0:
                self.time -= 1
            else:
                self.mode = "standby"
                return 1/self.scale
        return None

def modulate(data, fc, baud, sample_rate):
    oversample = sample_rate // baud

    iq = np.concatenate((PREFIX, data, SUFFIX))

    sampled_iq = np.convolve(
        np.repeat(iq, oversample),
        root_raised_cos_filter(SYMBOLS, oversample/F_CRCT, BETA)
    )

    samples = sampled_iq * np.exp(-1j *
        2*np.pi*fc/sample_rate * np.arange(len(sampled_iq)))

    return samples

class Demodulate:
    def __init__(self, fc, baud, sample_rate, callback=None):
        self.callback = callback
        self.oversample = sample_rate // baud
        self.filter = root_raised_cos_filter(SYMBOLS, self.oversample, BETA)
        self.buffer = np.zeros(len(self.filter), dtype=complex)
        
        self.mode = "standby"
        self.t = 0
        self.w = 2*np.pi*fc/sample_rate
        self.cal = Calibrator(self.oversample)
        self.samples = []

    def consume(self, samples):
        demod = samples * np.exp(1j * self.w * np.arange(self.t, self.t+len(samples)))
        self.t += len(samples)
        self.buffer = np.concatenate((self.buffer, demod))

        # convolve as much data as possible, then pass on to later stages
        if len(self.buffer) < len(self.filter):
            return
        convolve = np.convolve(self.buffer, self.filter, 'valid')
        self.buffer = self.buffer[len(convolve):]

        for x in convolve:
            if self.mode == "standby":
                self.scale = self.cal.step(x)
                if self.scale:
                    self.time = 0
                    self.mode = "record"
                    self.stop = 0
            elif self.mode == "recover":
                if self.time == 0:
                    self.mode = "standby"
                else:
                    self.time -= 1
            if self.mode == "record":
                y = x * self.scale
                end = abs(1. - y) < 0.3
                if self.time > 0:
                    self.time -= 1
                elif end and self.stop >= END_DETECT:
                    self.mode = "recover"
                    self.time = self.oversample * len(SUFFIX) - 1
                    if self.callback:
                        self.callback(self.samples[:-self.stop])
                        self.samples = []
                else:
                    if end:
                        self.stop += 1
                    else:
                        self.stop = 0
                    self.time = self.oversample - 1
                    self.samples.append(y)

def ber(tx, rx, width=4):
    total = 0
    for t, r in zip(tx, rx):
        total += bin(t^r).count("1")
    total += width*(max(len(tx), len(rx)) - min(len(tx), len(rx)))
    return total / (width*len(tx))

if __name__ == "__main__":
    SAMPLE_RATE = 48000
    fc = 3000
    baud = 300
    bits = np.random.randint(0, 16, 1000)
    
    samples = modulate(qam16_encode(bits), fc, baud, SAMPLE_RATE)
    samples = samples + np.random.normal(0,1,len(samples))
    
    def compare(samples_):
        print(ber(bits, qam16_decode(samples_)))

    demod = Demodulate(fc, baud, SAMPLE_RATE, callback=compare)
