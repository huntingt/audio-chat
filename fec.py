import numpy as np

def window(lst, interval, pad=None):
    if pad:
        lst += [pad] * (interval - 1)
    for i in range(0, len(lst), interval):
        yield lst[i:i+interval]

class ConvolutionalCode:
    def __init__(self, generators, ideals):
        self.g = generators
        
        self.r = len(self.g)
        self.K = int(max(np.log2(self.g)) + 1)

        self.num_states = 2**(self.K - 1)
        self.mask = self.num_states - 1

        collapse = 2**np.arange(self.r)
        self.ideals = ideals
        if len(ideals) != 2**self.r:
            raise ValueError(f"the number of parity bits doesn't correspond"
                             + " with the constellation size")

        @np.vectorize
        def mul_m2(a, b):
            return bin(a&b).count("1") & 1

        @np.vectorize
        def calculate_parity(x):
            bits = mul_m2(self.g, x)
            return np.dot(bits, collapse)

        self.parity = calculate_parity(np.arange(2*self.num_states))

        #
        # 'x' is choosen to be a permutation of the internal states such that
        # the resulting rows of next(x) are identity permutations. We can then
        # calculate the previous state that correspond to each position to
        # correctly swizzle the path metrics.
        #
        # x              previous      next
        # | 0 1 2 3 |    | 0 0 2 2 |   | 1 2 3 4 |
        # | 4 5 6 7 | -> | 1 1 3 3 | , | 1 2 3 4 |
        #             |
        #             |> parity matrix of x
        #
        x = np.reshape(np.arange(2*self.num_states), (2,self.num_states))

        self.swiz_state = x >> 1
        self.swiz_parity = self.parity[x]

    def encode(self, bits):
        state = 0
        samples = []
        for bit in bits:
            state = (state << 1) | bit
            samples.append(self.ideals[self.parity[state]])
            state &= self.mask
        return samples

    def decode(self, samples):
        pm = np.array([np.Inf] * self.num_states)
        pm[0] = 0.
        
        paths = []

        for sample in samples:
            delta = sample - self.ideals
            delta_pm = delta.real**2 + delta.imag**2
            candidate = pm[self.swiz_state] + delta_pm[self.swiz_parity]
            
            pm = candidate[0]
            path = self.swiz_state[0]
            for i in range(1, 2):
                ls = candidate[i] < pm
                pm = np.where(ls, candidate[i], pm)
                path = np.where(ls, self.swiz_state[i], path)
            
            paths.append(path)

        best_pm = min(pm)
        state = list(pm).index(best_pm)

        bits = np.zeros(len(paths), dtype=int)
        for i, row in enumerate(reversed(paths)):
            bits[-i-1] = state & 1
            state = row[state]

        return (best_pm, bits)

class HalvedQAMConvolutionalCode:
    def __init__(self, generators, l):
        ideals = np.array([
            m.square_qam(m.un(num, l)*2, l)[0].real
             for num in range(2**l)
        ])
        self.code = ConvolutionalCode(generators, ideals)

    def encode(self, bits):
        return np.array([
            complex(i,q) for i,q in window(self.code.encode(bits), 2, pad=0.)
        ])

    def decode(self, samples):
        halved = np.zeros(len(samples)*2)
        halved[::2] = samples.real
        halved[1::2] = samples.imag
        return self.code.decode(halved)

if __name__ == "__main__":
    import modem as m
    
    l = 2
    parity = np.array([0b111, 0b011])
    constellation = np.array(
        [m.square_qam(m.un(num, 2*l), l)[0] for num in range(2**(l+l))]
    )

    code = HalvedQAMConvolutionalCode(parity, l)

    modem = m.QAMModem(5000, 3000, 500)
    bits = np.random.randint(0, 2, 1000)
    ideal = np.array(code.encode(bits))
    tx = modem.modulate(ideal)

    rx = tx + 0.01 * np.random.normal(size=len(tx))

    demodulated = modem.demodulate(rx)
    if len(demodulated) == 1:
        actual = demodulated[0]

        m.square_qam_constellation(actual, l)

        dbits = code.decode(actual)[1]
        print(f"MER = {m.MER(ideal, actual)} dB")
        print(f"BER = {sum(dbits != bits)/len(bits)}")

        num = 100
        import time
        start = time.time()
        for i in range(num):
            code.decode(actual)
        end = time.time()
        print(f"{num} loops in {(end-start)/num} sec/loop")
    else:
        print(f"failed with {len(demodulated)} results")
