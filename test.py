import pyaudio
import numpy as np
import time
import modem as m

def play_and_record(osamples, sample_rate):
    isamples = []

    cursor = 0        
    def callback(in_data, frame_count, time_info, status):
        isamples.append(np.frombuffer(in_data, dtype=np.float32))
        data = np.zeros(frame_count, dtype=np.float32).tostring()
        return (data, pyaudio.paContinue)

    p = pyaudio.PyAudio()
    istream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=1024,
                    stream_callback=callback)
    ostream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    output=True)

    istream.start_stream()

    time.sleep(0.5)
    ostream.write(osamples.astype(np.float32).tostring())
    time.sleep(0.5)

    istream.stop_stream()

    ostream.close()
    istream.close()
    p.terminate()

    return np.concatenate(isamples)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fs = 48000
    modem = m.QAMModem(fs, 500, 50, 0.25)
    bits = np.random.randint(0, 2, 1000)
    ideal = m.square_qam(bits, 1)

    tx = modem.modulate(ideal)
    rx = play_and_record(tx, fs)

    #plt.plot(rx)
    #plt.show()

    decoded = modem.demodulate(rx)
    if len(decoded) == 1:
        actual = decoded[0]
        print(f"MER = {m.MER(ideal, actual)}")
        plt.scatter(actual.real, actual.imag, marker="+")
        plt.show()
    else:
        print(f"Failed with {len(decoded)} results")
        print(len(modem.samples))
        print(modem.mode)
