from moviepy.editor import VideoFileClip
import moviepy.video.fx.all as vfx

# Define input and output file locations
in_loc = './result.mp4'  # Path to the input video file
out_loc = './result0.mp4'  # Path to save the output video
multiplier = 2  # Speed up the video by this factor 

# Load the video clip
clip = VideoFileClip(in_loc)

# Print original FPS
print("Original FPS: {}".format(clip.fps))

# Increase the FPS to 30 (or any other desired value)
clip = clip.set_fps(30)

# Speed up the video by a factor of 2 (you can change this factor)
final = clip.fx(vfx.speedx, multiplier)

# Save the modified video with audio
final.write_videofile(out_loc, audio=True)

# Print the final FPS
print("Final FPS: {}".format(final.fps))
