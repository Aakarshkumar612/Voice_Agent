import audioop
import queue
import threading
import pyaudio
from config import CHANNELS, CHUNK_SIZE, INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE, SILENCE_THRESHOLD

class AudioHandler:
    def __init__(self):
        self._pa = pyaudio.PyAudio()
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self._running = False
        self._speaking = False
        self._capture_thread = None
        self._playback_thread = None
        self._input_stream = None
        self._output_stream = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._input_stream = self._pa.open(format=pyaudio.paInt16, channels=CHANNELS, rate=INPUT_SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SIZE)
        self._output_stream = self._pa.open(format=pyaudio.paInt16, channels=CHANNELS, rate=OUTPUT_SAMPLE_RATE, output=True, frames_per_buffer=CHUNK_SIZE)
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._capture_thread.start()
        self._playback_thread.start()

    def stop(self):
        self._running = False
        self.output_queue.put(None)
        if self._input_stream:
            self._input_stream.stop_stream()
            self._input_stream.close()
        if self._output_stream:
            self._output_stream.stop_stream()
            self._output_stream.close()
        self._pa.terminate()

    def interrupt(self):
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break
        self._speaking = False

    @property
    def is_speaking(self):
        return self._speaking

    def _capture_loop(self):
        while self._running:
            try:
                chunk = self._input_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                if self._speaking:
                    rms = audioop.rms(chunk, 2)
                    if rms > SILENCE_THRESHOLD:
                        self.interrupt()
                self.input_queue.put(chunk)
            except Exception:
                break

    def _playback_loop(self):
        while self._running:
            chunk = self.output_queue.get()
            if chunk is None:
                continue
            self._speaking = True
            try:
                self._output_stream.write(chunk)
            except Exception:
                pass
            if self.output_queue.empty():
                self._speaking = False
