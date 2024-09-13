from moviepy.editor import VideoFileClip

# Load the video file
clip = VideoFileClip("Video/Out/twin.mp4")
# Extract the segment from 3 to 9 seconds
clip = clip.subclip(29, 35)
# Convert to gif
clip.write_gif("Video/Out/twin2.gif")
