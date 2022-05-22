from pydub import AudioSegment

SECOND = 1000

all = AudioSegment.from_mp3("all.mp3")

first = all[1738*SECOND: 1768*SECOND]

first.export('10.mp3')
