import pyaudio
import modulate as m
import time
import numpy as np
import tkinter as tk
import hamming
from joblib import load

QAM_MODEL = load("model.joblib")
def qam16_decode(samples):
    old = np.concatenate(([1.], samples))[:-1]
    return QAM_MODEL.predict(list(zip(
        old.real, old.imag,
        samples.real, samples.imag
    )))

class Model:
    def __init__(self, display):
        self.display = display

        self.demodulator = None

        self.p = pyaudio.PyAudio()
        self.istream = None
        self.ostream = None

    def convert_samples(self, samples):
        nibs = qam16_decode(np.array(samples))
        nibs = hamming.decode(nibs)
        
        if len(nibs) % 2 == 1:
            self.display(f"[error] length is odd len={len(samples)}")

        raw = []
        for ms, ls in zip(nibs[1::2], nibs[::2]):
            raw.append(ms << 4 | ls)

        try:
            output = bytearray(raw).decode()
        except:
            output = f"[error] invalid UTF-8 {repr(bytearray(raw))}"
        self.display(output)

    def start(self,
              rx_sample_rate, rx_fc, rx_baud,
              tx_sample_rate, tx_fc, tx_baud):
        if self.demodulator is None:
            self.demodulator = m.Demodulate(rx_fc, rx_baud, rx_sample_rate,
                                            self.convert_samples)
            
            def callback(in_data, frame_count, time_info, status):
                self.demodulator.consume(np.frombuffer(in_data, dtype=np.float32))
                data = np.zeros(frame_count, dtype=np.float32).tostring()
                return (data, pyaudio.paContinue)

            self.istream = self.p.open(format=pyaudio.paFloat32,
                            channels=1,
                            rate=rx_sample_rate,
                            input=True,
                            frames_per_buffer=CHUNK,
                            stream_callback=callback)
            self.istream.start_stream()

            self.ostream = self.p.open(format=pyaudio.paFloat32,
                            channels=1,
                            rate=tx_sample_rate,
                            output=True)
            
            self.tx_sample_rate = tx_sample_rate
            self.tx_fc = tx_fc
            self.tx_baud = tx_baud

            time.sleep(0.1)

    def stop(self):
        if self.demodulator:
            self.demodulator = None

            self.istream.stop_stream()
            self.istream.close()
            self.ostream.close()

            self.istream = None
            self.ostream = None

    def send(self, string):
        nibs = []
        for byte in string.encode():
            nibs += [byte & 0xF, (byte >> 4) & 0xF]

        nibs = hamming.encode(nibs)
        samples = m.modulate(m.qam16_encode(nibs),
                             self.tx_fc,
                             self.tx_baud,
                             self.tx_sample_rate).real
        self.ostream.write(samples.astype(np.float32).tostring())

def isPositiveInteger(inp):
    try:
        i = int(inp)
        return i >= 0
    except:
        return inp == ""

class Window:
    def __init__(self, master):
        self.master = master
        self._create_settings()
        self._create_chat()
        self.model = Model(self.receive)

    def _create_settings(self):
        self.settings = tk.Frame(self.master, padx=5, pady=5)
        self.settings.place(anchor=tk.CENTER, relx=0.5, rely=0.5)

        isPInt = self.master.register(isPositiveInteger)
        def field(row, label, default=None):
            l = tk.Label(self.settings, text=label)
            l.grid(row=row, column=0)
            x = tk.Entry(self.settings,
                         validate="key",
                         validatecommand=(isPInt, "%P"))
            if default:
                x.insert(tk.END, default)
            x.grid(row=row, column=1)
            return x
        row = 0
        def next():
            nonlocal row
            row += 1
            return row - 1

        tk.Label(self.settings, text="Rx").grid(row=next(), column=0, columnspan=2)
        self.rx_sample_rate = field(next(), "Sample Rate (Hz)", "48000")
        self.rx_fc = field(next(), "Carrier (Hz)", "1500")
        self.rx_baud = field(next(), "Baud Rate (Hz)", "300")

        tk.Label(self.settings, text="Tx").grid(row=next(), column=0, columnspan=2)
        self.tx_sample_rate = field(next(), "Sample Rate (Hz)", "48000")
        self.tx_fc = field(next(), "Carrier (Hz)", "1500")
        self.tx_baud = field(next(), "Baud Rate (Hz)", "300")

        self.btn_start = tk.Button(self.settings,
                                   text="Start",
                                   padx=10, pady=3,
                                   command = self.start)
        self.btn_start.grid(row=next(), columnspan=2)

    def _create_chat(self):
        self.chat = tk.Frame(self.master, padx=5, pady=5)
        
        controls = tk.Frame(self.chat)
        controls.pack(fill=tk.X)

        self.btn_stop = tk.Button(controls,
                                  text="Stop",
                                  padx=10, pady=3,
                                  command = self.stop)
        self.btn_stop.pack(side=tk.LEFT)

        self.input = tk.Entry(controls)
        self.input.pack(fill=tk.X)
        self.input.bind('<Return>', self.send)

        self.log = tk.Text(self.chat, state=tk.DISABLED)
        self.log.pack(fill=tk.BOTH)

    def start(self):
        self.settings.place_forget()
        self.chat.pack(fill=tk.BOTH)

        rx_sample_rate = int(self.rx_sample_rate.get())
        rx_fc = int(self.rx_fc.get())
        rx_baud = int(self.rx_baud.get())
        
        tx_sample_rate = int(self.tx_sample_rate.get())
        tx_fc = int(self.tx_fc.get())
        tx_baud = int(self.tx_baud.get())

        self.model.start(
            rx_sample_rate, rx_fc, rx_baud,
            tx_sample_rate, tx_fc, tx_baud
        )

    def stop(self):
        self.chat.pack_forget()
        self.settings.place(anchor=tk.CENTER, relx=0.5, rely=0.5)
        self.model.stop()

    def receive(self, data):
        def helper():
            self.printToLog(f"< {data}\n")
        self.log.after(0, helper)

    def send(self, event):
        data = self.input.get()
        self.input.delete(0, tk.END)
        self.printToLog(f"> {data}\n");
        self.model.send(data)

    def printToLog(self, string):
        atBottom = self.log.yview()[1] == 1.0

        self.log.config(state=tk.NORMAL)
        self.log.insert(tk.END, string)
        height = float(self.log.index(tk.END))
        self.log.config(height=height, state=tk.DISABLED)

        if atBottom:
            self.log.see(tk.END)

CHUNK = 1024

if __name__ == "__main__":
    if True:
        root = tk.Tk()
        root.geometry("600x400")
        window = Window(root)
        def on_close():
            window.model.stop()
            root.destroy()
        root.protocol("WM_DELETE_WINDOW", on_close)
        root.mainloop()

    if False:
        SAMPLE_RATE = 48000
        FC = 3000
        BAUD = 300

        def display(samples):
            print("\033[200D< ", m.qam16_decode(samples))
            print("> ", end="", flush=True)

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

        time.sleep(0.1)

        ostream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=SAMPLE_RATE,
                        output=True,
                        output_device_index=1)

        while True:
            line = input("> ")
            if line == ":end":
                break

            bits = []
            for byte in line.encode():
                bits += [byte & 0xF, (byte >> 2) & 0xF]

            samples = m.modulate(m.qam16_encode(bits), FC, BAUD, SAMPLE_RATE).real
            ostream.write(samples.astype(np.float32).tostring())

        istream.stop_stream()
        istream.close()
        ostream.close()
        p.terminate()

    if False:
        SAMPLE_RATE = 48000
        FC = 3000
        BAUD = 300


        s = None
        def display(samples):
            global s
            s = np.array(samples)
            print(m.qam16_decode(samples))

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

        import matplotlib.pyplot as plt
        s *= 1.15
        plt.scatter(s.real, s.imag)
        for x in [-2/3*m.A_MAX, 0, 2/3*m.A_MAX]:
            plt.axvline(x=x)
            plt.axhline(y=x)
        plt.show()
