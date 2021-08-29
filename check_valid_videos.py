import os
from glob import glob
from slowfast.datasets import decoder as decoder
from slowfast.datasets import video_container as container
from tqdm import tqdm


VIDEO_DIR = "/u01/server_100_backup/ComputerVision/ActionRecognition/datasets/kinetics/400/extracted/val"
video_paths = glob(os.path.join(VIDEO_DIR, "*.mp4"))
valid_videos = []
load_error_videos = []
load_error_meta_videos = []
decode_error_videos = []

for vd_path in tqdm(video_paths):
    video_container = None
    try:
        video_container = container.get_video_container( vd_path, True, "pyav")
    except Exception as e:
        load_error_videos.append(vd_path)
        continue

    if video_container is None:
        load_error_meta_videos.append(vd_path)
        continue

    frames = decoder.decode(
                video_container,
                2,
                32,
                0,
                3,
                video_meta={},
                target_fps=30,
                backend="pyav",
                max_spatial_scale=256,
                use_offset=False,
            )

    if frames is None:
        decode_error_videos.append(vd_path)
        continue
    valid_videos.append(vd_path)

load_error_per = len(load_error_videos)/len(video_paths)*100
load_meta_error_per = len(load_error_meta_videos)/len(video_paths)*100
decode_error_per = len(decode_error_videos)/len(video_paths)*100
valid_per = len(valid_videos)/len(video_paths)*100
print("\nNum of video: {} \nLoad error percent: {} \nLoad meta error percent: {}  \nDecode error percent: {}  \nvalid video: {}".format(
    len(video_paths),
    load_error_per,
    load_meta_error_per,
    decode_error_per,
    valid_per
))

log = "Load error videos: \n"
for v in load_error_videos:
    log += v + "\n"
log += "\n\Load meta error videos: \n"
for v in load_error_meta_videos:
    log += v + "\n"
log += "\nDecode error videos: \n"
for v in decode_error_videos:
    log += v + "\n"

with open("output/valid_videos.txt", 'w') as f:
    f.write(log)







