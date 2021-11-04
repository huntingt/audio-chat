import pyaudio
import modem as m
import time
import numpy as np
import tkinter as tk

class Model:
    def __init__(self, display):
        self.display = display
        self.modem = None
        self.p = pyaudio.PyAudio()

    def convert_samples(self, samples):
        bits = m.from_square_qam(samples, 1)
        byte = [m.bg(bits[i:i+8]) for i in range(0,len(bits),8)]

        try:
            output = bytearray(byte).decode()
        except:
            output = f"[error] invalid UTF-8 {repr(bytearray(byte))}"
        self.display(output)

    def start(self, fs, fc, baud):
        if self.modem is None:
            self.modem = m.QAMModem(fs, fc, baud, 0.25, max=1024)
            
            def callback(in_data, frame_count, time_info, status):
                msgs = self.modem.demodulate(np.frombuffer(in_data, dtype=np.float32))
                for msg in msgs:
                     self.convert_samples(msg)

                data = np.zeros(frame_count, dtype=np.float32).tostring()
                return (data, pyaudio.paContinue)

            self.istream = self.p.open(format=pyaudio.paFloat32,
                            channels=1,
                            rate=fs,
                            input=True,
                            frames_per_buffer=CHUNK,
                            stream_callback=callback)
            self.istream.start_stream()

            self.ostream = self.p.open(format=pyaudio.paFloat32,
                            channels=1,
                            rate=fs,
                            output=True)

            time.sleep(0.1)

    def stop(self):
        if self.modem:
            self.modem = None

            self.istream.stop_stream()
            self.istream.close()
            self.ostream.close()

    def send(self, string):
        byte = string.encode()
        def get_bit(i):
            base = i // 8
            offset = i % 8
            return (byte[base] >> offset) & 1
        bits = np.array([get_bit(i) for i in range(len(byte)*8)])
        samples = self.modem.modulate(m.square_qam(bits, 1))
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

        #tk.Label(self.settings, text="Rx").grid(row=next(), column=0, columnspan=2)
        self.sample_rate = field(next(), "Sample Rate (Hz)", "48000")
        self.fc = field(next(), "Carrier (Hz)", "3000")
        self.baud = field(next(), "Baud Rate (Hz)", "500")

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

        fs = int(self.sample_rate.get())
        fc = int(self.fc.get())
        baud = int(self.baud.get())

        self.model.start(fs, fc, baud)

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
