from pydub import AudioSegment

# files                                                                       
src = "10.mp3"
dst = "10.wav"

# convert wav to mp3                                                            
audSeg = AudioSegment.from_mp3("10.mp3")
audSeg.export(dst, format="wav")
