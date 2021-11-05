import numpy as np

def window(lst, interval, pad=None):
    if pad:
        lst += [pad] * (interval - 1)
    for i in range(0, len(lst), interval):
        yield lst[i:i+interval]

@np.vectorize
def mul_m2(a, b):
    a &= b
    a ^= a >> 1
    a ^= a >> 2
    a ^= a >> 4
    a ^= a >> 8
    a ^= a >> 16
    return a & 1

class ConvolutionCode:
    def __init__(self, generators, ideals):
        self.g = generators
        
        self.r = len(self.g)
        self.K = int(max(np.log2(self.g[self.g!=0])) + 1)
        if self.K > 31:
            raise RuntimeError(f"K={self.K} exceds mul_m2 capability")

        self.num_states = 2**(self.K - 1)
        self.mask = self.num_states - 1

        self.collapse = 2**np.arange(self.r)
        self.ideals = ideals
        if len(ideals) != 2**self.r:
            raise ValueError(f"the number of parity bits doesn't correspond"
                             + " with the constellation size")

    def encode(self, bits):
        state = 0
        for bit in bits:
            x = (state << 1) | bit
            state = x & self.mask

            parity = mul_m2(self.g, x)
            yield self.ideals[np.dot(parity, self.collapse)]

    def decode(self, samples):
        pm = np.array([np.Inf] * self.num_states)
        pm[0] = 0.
        
        paths = []

        for sample in samples:
            next_pm = np.array([np.Inf] * self.num_states)
            path = [None] * self.num_states

            delta = sample - self.ideals
            delta_pm = delta.real**2 + delta.imag**2
            
            for state, metric in enumerate(pm):
                for state_0 in range(2):
                    x = (state << 1) | state_0
                    next_state = x & self.mask

                    parity = mul_m2(self.g, x)
                    candidate = metric + delta_pm[np.dot(parity, self.collapse)]

                    if candidate < next_pm[next_state]:
                        next_pm[next_state] = candidate
                        path[next_state] = state

            pm = next_pm
            paths.append(path)

        best_pm = min(pm)
        state = list(pm).index(best_pm)

        bits = np.zeros(len(paths), dtype=int)
        for i, row in enumerate(reversed(paths)):
            bits[-i-1] = state & 1
            state = row[state]

        return (best_pm, bits)

if __name__ == "__main__":
    import modem as m
    
    l = 1
    parity = np.array([0b111, 0b011])
    constellation = np.array(
        [m.square_qam(m.un(num, 2*l), l)[0] for num in range(4)]
    )
    code = ConvolutionCode(parity, constellation)

    modem = m.QAMModem(5000, 3000, 500, 0.25)
    bits = np.random.randint(0, 2, 1000)
    ideal = np.array(list(code.encode(bits)))
    tx = modem.modulate(ideal)

    rx = tx + 0.1 * np.random.normal(size=len(tx))

    demodulated = modem.demodulate(rx)
    if len(demodulated) == 1:
        actual = demodulated[0]
        dbits = code.decode(actual)[1]
        print(f"MER = {m.MER(ideal, actual)} dB")
        print(f"BER = {sum(dbits != bits)/len(bits)}")
    else:
        print(f"failed with {len(demodulated)} results")
