

from moviepy.editor import *
from path import Path
from PIL import Image, ImageDraw
import numpy as np 

import gradio 
import os
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

os.makedirs('video', exist_ok=True)

def procss_video(video_str):
    source_frames = []

    clip = VideoFileClip(video_str)
    for item in clip.iter_frames():
        source_frames.append(item)

    audioclip = clip.audio


    app = FaceAnalysis(providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    im2 = Image.open('mask_output.png')
    dealed_frames = []

    for item in source_frames:
        pil_image = Image.fromarray(item)
        faces = app.get(item)
        # rimg = app.draw_on(img, faces)
        for face in faces:
            # print(face)
            face.bbox = face.bbox.astype(np.int)
            top ,  right,bottom,  left = face.bbox
            #find top right bottom left from face.bbox
            left, bottom, right, top = face.bbox

            im = im2.resize((int(abs(top-bottom)*0.8), int(abs(left-right)*0.8) ))
            pil_image.paste(im, (left, int((top+bottom)/2)),  im)

        dealed_frames.append(np.array(pil_image))

    output_clip = ImageSequenceClip(dealed_frames, fps = clip.fps)
    new_audioclip = CompositeAudioClip([audioclip])
    output_clip.audio = new_audioclip
    output_clip.write_videofile(os.path.join('video', 'processed_'+Path(video_str).name),codec="libx264", audio_codec="aac")

    return os.path.join('video', 'processed_'+Path(video_str).name)
# os.mkdir('video')


def video_identity(video):
    # print(video)
    return procss_video(video)


demo = gradio.Interface(video_identity, 
                    gradio.Video(), 
                    "playable_video", 
                    examples=[
                        os.path.join(os.path.dirname(__file__), 
                                     "video/ZW9WPPMgpWP4bjjs.mp4")], 
                    cache_examples=False)

if __name__ == "__main__":
    demo.launch(share=True)
