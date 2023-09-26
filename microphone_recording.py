import pyaudio
import wave


def mic_audio():
    #========================================SETS CHUNK TO 1024,CHANNEL TO 1ST AND RATE TO 16KHZ
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()

    stream = p.open(
        format = FORMAT,
        channels = CHANNELS,
        rate = RATE,
        input = True,
        frames_per_buffer = CHUNK)

    
#allow user to choose how long they want to record for
    def choose_length():
        seconds = int(input("How many seconds would you like to record for?:"))
        print("Starting recording...")
        return seconds
    frames = []
    seconds = choose_length()
    for i in range(0,int(RATE/CHUNK * seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("recording stopped")

    stream.stop_stream()
    stream.close()
    p.terminate

    #===========CREATE WAV FILE=

    wf = wave.open("output.wav","wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close


#===============CREATE WAV FILE====
