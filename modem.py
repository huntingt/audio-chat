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
        if a * np.sqrt(2) > 1 - (2*i-1)/(s-1):
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
    def __init__(self, peaks, U, **kwargs):
        self.kwargs = {
            "peak_end_factor": 0.6,
            "peak_start_factor": 0.8,
            "min_|a|": 0.001,
            "max_sigma_|a|/|a|": 0.1,
            **kwargs
        }

        self.peaks = []
        self.ideal_peaks = peaks
        self.max = None
        self.max_t = 3 * U * len(self.ideal_peaks)
        self.t = 0
        self.U = U

    def preamble(self):
        result = [0]
        for peak in self.ideal_peaks:
            result += [peak, 0]
        return result

    def peak_max(self):
        if len(self.peaks) == 0:
            return 0
        return max(abs(a) for t, a in self.peaks)
        # return sum(abs(a) for t, a in self.peaks)/len(self.peaks)

    def check_preamble(self):
        # trim the peaks if necessary
        while len(self.peaks) > len(self.ideal_peaks):
            self.peaks = self.peaks[1:]

        # discard peaks that are too old, important to make sure that spurrious
        # signals don't prevent further peak detection
        while len(self.peaks) > 0 and self.t - self.peaks[0][0] > self.max_t:
            self.peaks = self.peaks[1:]

        # reset the time if there are no peaks left
        if len(self.peaks) == 0:
            self.t = 0

        if len(self.peaks) != len(self.ideal_peaks):
            return None

        # multiply by the complex conjugate of the expected peaks to get the
        # per peak amplitude and phase offset
        amp = np.array([x for (t, x) in self.peaks])\
            * np.conj(self.ideal_peaks)

        mean_amp = np.mean(amp)
        dev_amp = np.std(abs(amp)) / np.mean(abs(amp))

        times = np.array([t for t, x in self.peaks])

        times = times - 2 * self.U * np.arange(len(times))
        time_of_first_peak = np.mean(times)
        time_of_first_data = time_of_first_peak + (2*len(times)+0)*self.U

        dt = time_of_first_data - self.t
        sigma_dt = np.std(times)

        if abs(mean_amp) >= self.kwargs["min_|a|"] and\
           dev_amp < self.kwargs["max_sigma_|a|/|a|"]:

            self.peaks = []
            self.t = 0

            return {
                "dt": dt,
                "sigma_dt": sigma_dt,
                "sigma_|a|/|a|": dev_amp,
                "|a|": abs(mean_amp),
                "1/a": 1/mean_amp
            }
        else:
            return None

    def step(self, x):
        # detect scanning mode for new peaks
        if self.max is None:
            if abs(x) > self.kwargs["peak_start_factor"] * self.peak_max():
                self.max = (self.t, x)
        # detect new max for tracking peak
        elif abs(x) > abs(self.max[1]):
            self.max = (self.t, x)
        # detect end of tracking peak
        elif abs(x) < self.kwargs["peak_end_factor"] * abs(self.max[1]):
            self.peaks.append(self.max)
            self.max = None
            return self.check_preamble()
        self.t += 1
        return None

class QAMModem:
    def __init__(self, fs, fc, baud, **kwargs):
        self.kwargs = {
            "preamble": [1, 1, 1, 1, 1, 1j, -1j, -1, -1, -1j, 1j, 1],
            "dt_bias": -7.,
            "conclusion_length": 5,
            "conclusion_length_threshold": 3,
            "conclusion_point": 0.,
            "conclusion_radius": 0.2,
            "srrc_beta": 0.25,
            "srrc_symbols": 10,
            "max_length": None,
            **kwargs
        }

        self.fs = fs
        
        self.U = fs // baud
        baud = fs // self.U

        self.fc = (fc // baud) * baud

        self.filter = root_raised_cos_filter(
            self.kwargs["srrc_symbols"],
            self.U,
            self.kwargs["srrc_beta"]
        )

        self.preamble_detector = PreambleDetector(self.kwargs["preamble"],
                                                  self.U, **kwargs)
        self.preamble = self.preamble_detector.preamble()

        self.conclusion = [self.kwargs["conclusion_point"]]\
                * self.kwargs["conclusion_length"]

        self.buffer = np.zeros(len(self.filter), dtype=complex)
        self.mode = "standby"
        self.samples = []
        self.t = 0

    def modulate(self, pairs):
        pairs = np.concatenate((self.preamble, pairs, self.conclusion))

        # upsample to the sampling frequency
        iq = np.zeros(len(pairs) * self.U, dtype=complex)
        iq[::self.U] = pairs

        # pulse shaping filter
        iq = np.convolve(iq, self.filter)

        # modulate
        mod = iq * np.exp(-2j*np.pi*self.fc/self.fs * np.arange(len(iq)))
        return mod.real

    def bandwidth(self):
        return self.fs * (1+self.kwargs["srrc_beta"]) / (2*self.U)

    def step_state(self, x):
        if self.mode == "standby":
            self.corrections = self.preamble_detector.step(x)
            if self.corrections is not None:
                print(self.corrections)
                self.timer = int(self.corrections['dt'] + self.kwargs["dt_bias"])
                self.stop = 0
                self.mode = "record"
        elif self.mode == "record":
            y = x * self.corrections['1/a']
            end = abs(self.kwargs["conclusion_point"] - y) <\
                self.kwargs["conclusion_radius"]
            if self.timer > 0:
                self.timer -= 1
            elif end and self.stop >= self.kwargs["conclusion_length_threshold"]:
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

                if self.kwargs["max_length"] is not None and len(self.samples) - self.stop >\
                   self.kwargs["max_length"]:
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

    def square_qam_constellation(pairs, l):
        fig, ax = plt.subplots(figsize = (6, 6))
        ax.scatter(pairs.real, pairs.imag, c="black", linewidths=1., s=15.,
                   marker="+")
        ax.tick_params(left=True, right=True, bottom=True, top=True)
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        size = np.array((-1,1))*(1+1/(2**l-1))/np.sqrt(2)
        ax.set_xlim(size)
        ax.set_ylim(size)
        if l is not None:
            ax.set_title(f"{2**(l+l)}QAM Constellation")
            for i in range(1, 2**l):
                line = (1 - (2*i-1)/(2**l-1))/np.sqrt(2)
                ax.axvline(x=line, ls="--", c="black", lw=.5)
                ax.axhline(y=line, ls="--", c="black", lw=.5)
        else:
            ax.set_title(f"QAM Constellation")
        plt.tight_layout()
        plt.show()

    l = 2
    modem = QAMModem(50000, 3000, 500)
    bits = np.random.randint(0,2,1000)
    ideal = square_qam(bits, l)
    tx = modem.modulate(ideal)

    rx = tx + 0.05 * np.random.normal(size=len(tx))

    freq = np.fft.fftfreq(rx.size, d=1./modem.fs)[:len(rx)//2]
    fft = abs(np.fft.fft(rx))[:len(rx)//2]

    fig, ax = plt.subplots(figsize = (9, 6))
    ax.plot(freq, fft)
    ax.set_yscale("log")
    plt.show()

    s_start = np.searchsorted(freq, modem.fc-modem.bandwidth()/3.)
    s_end   = np.searchsorted(freq, modem.fc+modem.bandwidth()/3.)
    n_start = np.searchsorted(freq, modem.fc+modem.bandwidth()*1.5)
    n_end   = np.searchsorted(freq, modem.fc+modem.bandwidth()*2.0)
    snr = 20 * np.log10(
        np.mean(fft[s_start:s_end])/np.mean(fft[n_start:n_end]))
    print(f"SNR = {snr} dB")

    decoded = modem.demodulate(rx)
    if len(decoded) == 1:
        actual = decoded[0]
        print(f"MER = {MER(ideal, actual)} dB")
        print(f"BER = {sum(bits != from_square_qam(actual, l))/len(bits)}")
        square_qam_constellation(actual, l)
    else:
        print(f"failed with {len(decoded)} results")
