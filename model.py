import numpy as np
import matplotlib.pyplot as plt
import modulate as m
import pyaudio
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump

if __name__ == "__main__" and True:
    results = np.load("data.npy")

    def evaluate(func):
        bers = [m.ber(sym.real.astype(int), func(sam)) for sym, sam in results]
        return np.average(bers)

    # 3.9% ber with factor = 1.165
    @np.vectorize
    def scale(factor):
        def decode(samples):
            return m.qam16_decode(samples * factor)
        return evaluate(decode)

    def pairs():
        X = []
        Y = []
        for y, x in results:
            old = np.concatenate(([1.+0.j], x))[:-1]
            assert(len(old) == len(x))
            Y += list(y.real.astype(int))
            X += list(zip(old.real, old.imag, x.real, x.imag))
        return (X, Y)

    X, y = pairs()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

    model = KNeighborsClassifier(n_neighbors = 16)
    model.fit(X_train, y_train)

    print(f"Accuracy on training set: {model.score(X_train, y_train)}")
    print(f"Accuracy on test set: {model.score(X_test, y_test)}")

    dump(model, "model.joblib")

if __name__ == "__main__" and False:
    CHUNK = 1024
    SAMPLE_RATE = 48000
    FC = 3000
    BAUD = 300

    results = []
    for i in range(100):
        s = None
        def display(samples):
            global s
            s = np.array(samples)

        demod = m.Demodulate(FC, BAUD, SAMPLE_RATE, display)
        
        def callback(in_data, frame_count, time_info, status):
            demod.consume(np.frombuffer(in_data, dtype=np.float32))
            data = np.zeros(frame_count, dtype=np.float32).tostring()
            return (data, pyaudio.paContinue)

        p = pyaudio.PyAudio()
        istream = p.open(format=pyaudio.paFloat32,
                         channels=1,
                         rate=SAMPLE_RATE,
                         input=True,
                         frames_per_buffer=CHUNK,
                         stream_callback=callback)
        istream.start_stream()

        time.sleep(1)

        ostream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=SAMPLE_RATE,
                        output=True,
                        output_device_index=1)

        bits = np.random.randint(0, 16, 1000)
        samples = m.modulate(m.qam16_encode(bits), FC, BAUD, SAMPLE_RATE).real
        ostream.write(samples.astype(np.float32).tostring())

        time.sleep(1)

        istream.stop_stream()
        istream.close()
        ostream.close()
        p.terminate()

        if s is None:
            print("no read")
        elif len(s) != len(bits):
            print(f"mismatch {len(s)} != {len(bits)}")
        else:
            results.append((bits, s))
    np.save('data.npy', results)
