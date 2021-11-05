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
        bits = m.from_square_qam(samples, self.l)
        byte = [m.bg(bits[i:i+8]) for i in range(0,len(bits),8)]

        try:
            output = bytearray(byte).decode()
        except:
            output = f"[error] invalid UTF-8 {repr(bytearray(byte))}"
        self.display(output)

    def start(self, fs, fc, baud, l):
        if self.modem is None:
            self.l = l

            self.modem = m.QAMModem(fs, fc, baud, max_length=1024)
            
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
        samples = self.modem.modulate(m.square_qam(bits, self.l))
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
        tk.Label(self.settings, text="Modulation").grid(row=row, column=0)
        
        self.modulation = tk.StringVar(self.settings, value="QAM4")
        tk.OptionMenu(self.settings, self.modulation, "QAM4", "QAM16",)\
                .grid(row=next(), column=1)


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
        l = ["QAM4", "QAM16"].index(self.modulation.get()) + 1

        self.model.start(fs, fc, baud, l)

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
