import numpy as np
import multiprocessing as mp
from subprocess import Popen, PIPE
import time


def frame_generator(frames_buffer, infos_buffer, fps=30):
    while True:
        yield frames_buffer.get(), infos_buffer.get()


def stream_frames(
    frames_buffer,
    infos_buffer,
    fps=30,
    url="rstp://localhost:8554/stream",
    img_size=(1280, 720),
):
    p = Popen(
        [
            "ffmpeg",
            "-f",
            "rawvideo",
            "-framerate", f"{fps}/1",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            "x".join(map(str, img_size)),
            "-i",
            "pipe:",
            "-an",
            "-c:v",
            "libx264",
            "-bf",
            "0",
            "-g",
            "30",
            "-r", str(fps),
            "-pix_fmt",
            "yuv420p",
            "-f",
            "fifo",
            "-fifo_format",
            "flv",
            "-map",
            "0:v",
            "-flags",
            "+global_header",
            "-drop_pkts_on_overflow",
            "1",
            "-attempt_recovery",
            "1",
            "-recovery_wait_time",
            "1",
            url,
        ],
        stdin=PIPE,
    )
    for frame, infos in frame_generator(frames_buffer, infos_buffer, fps):
        p.stdin.write(frame.tobytes())
        p.stdin.flush()
        time.sleep(1 / fps)


class LivestreamingAgentMixin:
    def __init__(
        self,
        *args,
        stream_url="rtmp://localhost:1935/stream",
        throttle_queue=500,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.url = stream_url
        self.capturing = True
        self.frames_buffer = mp.Queue()
        self.infos_buffer = mp.Queue()
        self.throttle_queue = throttle_queue
        self.stream_process = mp.Process(
            target=stream_frames,
            args=(self.frames_buffer, self.infos_buffer),
            kwargs={
                "fps": self.env.metadata.get("render_fps", 30),
                "url": self.url,
                "img_size": self.env.render().shape[:2][::-1],
            },
        )
        self.stream_process.start()

    def step(self, action, env=None):
        res = super().step(action, env)
        if self.capturing:
            self.frames_buffer.put(self.env.render())
            self.infos_buffer.put(
                {
                    "action": action,
                    "reward": res[2],
                    "done": res[3],
                    "training_steps": self.training_steps,
                    "testing": self.test,
                }
            )
        return res

    def run_episode(self, test, *args, **kwargs):
        self.capturing = self.frames_buffer.qsize() < self.throttle_queue or test
        return super().run_episode(test, *args, **kwargs)
