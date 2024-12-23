import cv2
import numpy as np
import matplotlib.pyplot as plt 
# %matplotlib inline

# if 'google.colab' in str(get_ipython()):
#     print("Downloading Code to Colab Environment")
#     !wget https://www.dropbox.com/sh/t7x50ww3ultvwn3/AACnmQGBD7rIbznXzi1sRWJqa?dl=1 -O module-code.zip -q --show-progress
#     !unzip -qq module-code.zip
# else:
#     pass


from IPython.display import Video, display

input_video = 'race_car.mp4'

# To display a video from a local file
display(Video(input_video, width=640))



source = input_video  # source = 0 for webcam

video_cap = cv2.VideoCapture(source)
if (video_cap.isOpened()== False): 
  print("Error opening video stream or file")
  
  
  
ret, frame = video_cap.read()
plt.figure(figsize = (10, 8))
plt.imshow(frame[...,::-1])
plt.show()

# Retrieve video frame properties.
frame_w   = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h   = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_fps = int(video_cap.get(cv2.CAP_PROP_FPS))

# Specify the value for fourcc
# fourcc_avi = cv2.VideoWriter_fourcc('M','J','P','G')
fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')

# Specify the video output filenames.
# file_out_avi = 'video_out.avi'
file_out_mp4 = 'finaL.mp4'

# Slow down the speed of the video.
frame_fps = int(frame_fps/3)

# Create the video writer objects.
# out_avi = cv2.VideoWriter(file_out_avi, fourcc_avi, frame_fps, (frame_w,frame_h))
out_mp4 = cv2.VideoWriter(file_out_mp4, fourcc_mp4, frame_fps, (frame_w,frame_h))



def drawBannerText(frame, text, banner_height_percent = 0.05, text_color = (0,255,0)):
    # Draw a black filled banner across the top of the image frame.
    # percent: set the banner height as a percentage of the frame height.
    banner_height = int(banner_height_percent * frame.shape[0])
    cv2.rectangle(frame, (0,0), (frame.shape[1],banner_height), (0,0,0), thickness=-1)
    
    # Draw text on banner.
    left_offset = 20
    location = (left_offset, int( 5 + (banner_height_percent * frame.shape[0])/2 ))
    fontScale = 1.5
    fontThickness = 2
    cv2.putText(frame, text, location, cv2.FONT_HERSHEY_PLAIN, fontScale, text_color, fontThickness, cv2.LINE_AA)
    
    
# Read all the frames in the video.
frame_count = 0
while True:
  
    # Read one frame at a time.
    ok, frame = video_cap.read()
    if not ok:
        break

    # Increment the frame count for the annotation.
    frame_count += 1
  
    # Annotate each video frame.
    drawBannerText(frame, 'Frame: ' + str(int(frame_count)) + ' FPS: ' + str(int(frame_fps)))
    
    # Write the frame to the output files.
    # out_avi.write(frame)
    out_mp4.write(frame)
    
# Release the VideoCapture and VideoWriter objects
video_cap.release()
# out_avi.release()
out_mp4.release()



"""
    For Ubuntu/Linux
Update the Package List:

sudo apt update
Install FFmpeg:

sudo apt install ffmpeg
Verify the Installation:

ffmpeg -version
For macOS
Install Homebrew (if not already installed):

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
Install FFmpeg via Homebrew:

brew install ffmpeg
Verify the Installation:

ffmpeg -version
For Windows
Download FFmpeg:

Visit the FFmpeg download page.
Select a Windows build (e.g., from Gyan.dev).
Extract and Configure FFmpeg:

Extract the downloaded ZIP file to a folder (e.g., C:\ffmpeg).
Add FFmpeg to PATH:

Open System Properties → Advanced → Environment Variables.
Find the Path variable under System Variables → Click Edit.
Add the path to the bin folder inside the extracted FFmpeg folder (e.g., C:\ffmpeg\bin).
Verify the Installation:

Open Command Prompt and run:
ffmpeg -version
"""


import subprocess

# Define the FFMPEG command
command = [
    "ffmpeg", 
    "-i", "video_out.mp4", 
    "-vcodec", "libx264", 
    "-acodec", "aac", 
    "finaL1.mp4"
]

# Run the command
subprocess.run(command, check=True)


# from IPython.display import Video, display

output_video = 'finaL1.mp4'

# To display a video from a local file
display(Video(output_video, width=640))