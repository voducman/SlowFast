import os
import multiprocessing as mp
import time
from glob import glob
from slowfast.datasets import decoder as decoder
from slowfast.datasets import video_container as container
from tqdm import tqdm


VIDEO_DIR = "/u01/server_100_backup/ComputerVision/ActionRecognition/datasets/kinetics/400/extracted/val"
NUM_PROCESS = 10
video_paths = glob(os.path.join(VIDEO_DIR, "*.mp4"))
video_per_proc = int(len(video_paths)/NUM_PROCESS)
error_queue = mp.Queue()
status_queue = mp.Queue()


def check_videos(num_proc, vpaths, error_q, status_q):
    def collect_error_video(v_path):
        error_q.put(v_path)
        status_q.put({'num_process': num_proc, 'status': False})

    for path in vpaths:
        video_container = None
        try:
            video_container = container.get_video_container(path, True, "pyav")
        except Exception as e:
            collect_error_video(path)
            continue

        if video_container is None:
            collect_error_video(path)
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
            collect_error_video(path)
            continue

        status_q.put({'num_process': num_proc, 'status': True})


if __name__ == "__main__":
    processes = []
    for i in range(NUM_PROCESS):
        if i < NUM_PROCESS - 1:
            sub_paths = video_paths[i*video_per_proc:(i+1)*video_per_proc]
        else:
            sub_paths = video_paths[i*video_per_proc:]
        task = mp.Process(target=check_videos, args=(i, sub_paths, error_queue))
        task.run()
        processes.append(task)

    for t in processes:
        t.join()


    error_percent = round(error_queue.qsize()/len(video_paths)*100, 2)
    valid_percent = 100 - error_percent

    print("\nNum of video: {} \nError: {}  \nvalid video: {}".format(
        len(video_paths),
        error_percent,
        valid_percent
    ))

    log = "Error videos: \n"
    for i in range(error_queue.qsize()):
        vpath = error_queue.get()
        log += vpath + "\n"

    with open("output/invalid_videos.txt", 'w') as f:
        f.write(log)







