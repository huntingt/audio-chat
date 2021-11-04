import numpy as np

def bg(bits):
    num = 0
    for b in reversed(bits):
        num <<= 1
        num |= b
    return num

def un(num, nbits):
    return [(num >> i) & 1 for i in range(nbits)]

def from_grey(num):
    num ^= num >> 16
    num ^= num >> 8
    num ^= num >> 4
    num ^= num >> 2
    num ^= num >> 1
    return num

def to_grey(num):
    return num ^ (num >> 1)

def square_qam(bits, l):
    np.concatenate((bits, [0]*(2*l-1)))
    samples = []
    for j in range(0, len(bits), 2*l):
        i = 1 - to_grey(bg(bits[j:j+l]))*2./(2**l-1)
        q = 1 - to_grey(bg(bits[j+l:j+2*l]))*2./(2**l-1)
        samples.append(i + 1.j*q)
    return np.array(samples) / np.sqrt(2)

def partition_square(a, s):
    for i in range(1, s):
        if a * np.sqrt(2) > 1 - 2*i/s:
            return i - 1
    return s - 1

def from_square_qam(points, l):
    bits = []
    for point in points:
        bits += un(from_grey(partition_square(point.real, 2**l)), l)
        bits += un(from_grey(partition_square(point.imag, 2**l)), l)
    return np.array(bits)

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

def root_raised_cos_filter(symbols, U, beta):
    N = symbols * U + 1
    t = np.linspace(-symbols/2., symbols/2., N)
    filter = root_raised_cos_impulse(t, beta)
    return filter / np.linalg.norm(filter)

class PreambleDetector:
    def __init__(self, n_peaks, U):
        self.peaks = []
        self.n_peaks = n_peaks
        self.max = None
        self.t = 0
        self.U = U 

    def peak_avg(self):
        if len(self.peaks) == 0:
            return 0
        return sum(a for t, a in self.peaks)/len(self.peaks)

    def corrections(self):
        times = np.array([t for t, x in self.peaks[:-1]])
        amp = np.array([x for t, x in self.peaks[:-1]])

        times = times - 2 * self.U * np.arange(len(times))
        time_of_first_peak = np.mean(times)
        time_of_first_data = time_of_first_peak + (2*len(times)+1)*self.U
        dt = time_of_first_data - self.t
        sigma_dt = np.std(times)

        self.peaks = []
        self.t = 0

        return {
            "dt": dt,
            "sigma_dt": sigma_dt,
            "sigma_|a|/|a|": np.std(abs(amp))/np.mean(abs(amp)),
            "1/a": 1/np.mean(amp)
        }

    def step(self, x):
        # detect scanning mode for new peaks
        if self.max is None:
            if abs(x) > 0.6 * abs(self.peak_avg()):
                self.max = (self.t, x)
        # detect new max for tracking peak
        elif abs(x) > abs(self.max[1]):
            self.max = (self.t, x)
        # detect end of tracking peak
        elif abs(x) < 0.4 * abs(self.max[1]):
            self.peaks.append(self.max)
            if len(self.peaks) > self.n_peaks:
                self.peaks = self.peaks[1:]
                # found a reversed peak, signals end of preamble
                if self.max[1] * np.conj(self.peak_avg()) < 0:
                    return self.corrections()
            self.max = None
        self.t += 1
        return None

class QAMModem:
    def __init__(self, fs, fc, baud, beta, max=None):
        self.fs = fs
        
        self.U = fs // baud
        baud = fs // self.U

        self.fc = (fc // baud) * baud

        self.beta = beta
        self.filter = root_raised_cos_filter(10, self.U, beta)

        self.n_preamble = 15
        self.preamble = PreambleDetector(10, self.U)

        self.buffer = np.zeros(len(self.filter), dtype=complex)
        self.mode = "standby"
        self.samples = []
        self.suffix = 3
        self.t = 0

        self.max = max

    def modulate(self, pairs):
        pairs = np.concatenate(([0,1]*self.n_preamble+[-1, 1], pairs, [0]*5))

        # upsample to the sampling frequency
        iq = np.zeros(len(pairs) * self.U, dtype=complex)
        iq[::self.U] = pairs

        # pulse shaping filter
        iq = np.convolve(iq, self.filter)

        # modulate
        mod = iq * np.exp(-2j*np.pi*self.fc/self.fs * np.arange(len(iq)))
        return mod.real

    def bandwidth(self):
        return self.fs * (1+self.beta) / (2*self.U)

    def demodulate2(self, samples):
        demod = samples * np.exp(2j*np.pi*self.fc/self.fs *
                                 np.arange(len(samples)))
        return demod
        # matched filter
        iq = np.convolve(demod, self.filter)
        return iq

    def step_state(self, x):
        if self.mode == "standby":
            self.corrections = self.preamble.step(x)
            if self.corrections and\
               self.corrections['sigma_|a|/|a|'] < 0.1 and\
               abs(self.corrections['1/a']) < 100:
                print(self.corrections)
                self.timer = int(self.corrections['dt'])
                self.stop = 0
                self.mode = "record"
        elif self.mode == "record":
            y = x * self.corrections['1/a']
            # end = abs(1 - y) < 0.2 TODO
            end = abs(y) < 0.2
            if self.timer > 0:
                self.timer -= 1
            elif end and self.stop >= self.suffix:
                samples = self.samples
                self.samples = []
                self.mode = "standby"
                return samples[:-self.stop]
            else:
                if end:
                    self.stop += 1
                else:
                    self.stop = 0
                self.timer = self.U - 1
                self.samples.append(y)

                if self.max is not None and len(self.samples) - self.stop >\
                   self.max:
                    self.samples = []
                    self.mode = "standby"
        return None

    def demodulate(self, samples):
        demod = samples * np.exp(2j*np.pi*self.fc/self.fs *
                                 np.arange(self.t, self.t+len(samples)))
        self.t += len(samples)
        self.buffer = np.concatenate((self.buffer, demod))

        # convolve as much data as possible, then pass on to later stages
        if len(self.buffer) < len(self.filter):
            return []
        iq = np.convolve(self.buffer, self.filter, 'valid')
        self.buffer = self.buffer[len(iq):]

        results = []
        for point in iq:
            samples = self.step_state(point)
            if samples is not None:
                results.append(np.array(samples))
        return results
        
def MER(ideal, actual):
    num = sum(ideal*np.conj(ideal))
    diff = actual - ideal
    return 10*np.log10(num/sum(diff*np.conj(diff))).real

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    modem = QAMModem(50000, 3000, 500, 0.25)
    bits = np.random.randint(0,2,1000)
    ideal = square_qam(bits, 2)
    tx = modem.modulate(ideal)

    rx = tx + 0.05 * np.random.normal(size=len(tx))

    freq = np.fft.fftfreq(rx.size, d=1./modem.fs)[:len(rx)//2]
    fig, ax = plt.subplots(figsize = (9, 6))
    ax.plot(freq, abs(np.fft.fft(rx))[:len(rx)//2])
    ax.set_yscale("log")
    plt.show()

    decoded = modem.demodulate(rx)
    if len(decoded) == 1:
        actual = decoded[0]
        print(f"MER = {MER(ideal, actual)} db")
        plt.scatter(actual.real, actual.imag, marker="+")
        plt.show()
    else:
        print(f"failed with {len(decoded)} results")
