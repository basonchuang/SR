import sys
from pytube import YouTube

def progress(chunk,file_handle,bytes_remaining):
	contentSize=video.filesize
	size=contentSize-bytes_remaining
	print('\r' + '[Download progress]:%.2f%%;\n' % (' ' * int(size*20/contentSize), ' '*(20-int(size*20/contentSize)), float(size/contentSize*100)), end='')

def main(url):
    yt=YouTube(url,on_progress_callback=progress)
    video=yt.streams.first()
    video.download()

if __name__== "__main__":
    main(sys.argv[1])
