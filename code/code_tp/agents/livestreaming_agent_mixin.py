import numpy as np
import multiprocessing as mp
from subprocess import Popen, PIPE
import time
import pathlib
import socket
import os
import traceback
import signal


def PRESET_ENCODED(img_size, fps, output):
    return [
        "ffmpeg",
        "-f",
        "rawvideo",
        "-framerate", f"{fps}/1",
        "-readrate", "1",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        "x".join(map(str, img_size)),
        "-i",
        "pipe:",
        "-vf",
        "scale=-2:480",
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
        output,
    ]

def PRESET_RAW(img_size, fps, output):
    return [
        "ffmpeg",
        "-f",
        "rawvideo",
        "-framerate", f"{fps}/1",
        "-readrate", "1",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        "x".join(map(str, img_size)),
        "-i",
        "pipe:",
        "-c", "copy",
        "-f",
        "fifo",
        "-fifo_format",
        "rtp",
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
        output,
    ]


def frame_generator(frames_buffer, infos_buffer, fps=30):
    while True:
        yield frames_buffer.get(), infos_buffer.get()


def stream_frames(
    frames_buffer,
    infos_buffer,
    stop_event,
    fps=30,
    url="rstp://localhost:8554/stream",
    img_size=(1280, 720),
):
    p = Popen(
        PRESET_RAW(img_size, fps, url),
        #PRESET_ENCODED(img_size, fps, url),
        stdin=PIPE,
    )
    for frame, infos in frame_generator(frames_buffer, infos_buffer, fps):
        p.stdin.write(frame.tobytes())
        p.stdin.flush()
        time.sleep(1 / fps)
        if stop_event.is_set():
            break


def chunk_frame(frame):
    packet_size = 1400  # Safe UDP packet size
    frame_bytes = frame.tobytes()
    total_packets = (len(frame_bytes) + packet_size - 1) // packet_size  # Number of packets

    for i in range(total_packets):
        chunk = frame_bytes[i * packet_size : (i + 1) * packet_size]
        yield chunk

    


def stream_to_pipes(
    frames_buffer,
    infos_buffer,
    stop_event,
    fps=30,
    url="/tmp/mario_stream",
    img_size=(1280, 720),
):
    pipe_path = pathlib.Path(url+".pipe")
    fps_path = pathlib.Path(url+".fps")
    format_path = pathlib.Path(url+".format")
    format_str = "x".join(map(str, img_size))
    with open(fps_path, "w") as f:
        f.write(str(fps))
    with open(format_path, "w") as f:
        f.write(str(format_str))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = ("127.0.0.1", 45454)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    for frame_id, (frame, infos) in enumerate(frame_generator(frames_buffer, infos_buffer, fps)):
        try:
            for chunk_id, chunk in enumerate(chunk_frame(frame)):
                # Add a simple RTP-style header: [frame_id (4 bytes), packet_id (2 bytes)]
                sock.sendto(chunk, addr)  # Send chunk with header
            time.sleep(1 / fps)
        except Exception as e:
            print(traceback.format_exc())
            time.sleep(1)
        if stop_event.is_set():
            print("Stop event received")
            break
    print(frame_id)
    frame = np.ones_like(frame, dtype=np.uint8)
    for i in range(1,100):
        for chunk_id, chunk in enumerate(chunk_frame(frame)):
            # Add a simple RTP-style header: [frame_id (4 bytes), packet_id (2 bytes)]
            sock.sendto(chunk, addr)  # Send chunk with header
        time.sleep(1/fps)
    sock.close()
    print("Socket closed")
    return True



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
        self.stop_event = mp.Event()
        self.throttle_queue = throttle_queue
        self.stream_process = mp.Process(
            target=stream_to_pipes,
            #target=stream_frames,
            args=(self.frames_buffer, self.infos_buffer, self.stop_event),
            kwargs={
                "fps": self.env.metadata.get("render_fps", 30),
                "url": self.url,
                "img_size": self.env.render().shape[:2][::-1],
            },
        )

    def interrupt(self, *args):
        self.should_stop = True
        self.stop_event.set()
        self.stream_process.join()
        self.original_sigint_handler(*args)

    def train(self, *args, **kwargs):
        self.stream_process.start()
        self.original_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.interrupt)
        try:
            super().train(*args, **kwargs)
        except KeyboardInterrupt:
            print("Keyboard interrupted")
        self.stop_event.set()
        self.stream_process.join()

    def step(self, action, env=None):
        res = super().step(action, env)
        if self.capturing:
            self.frames_buffer.put(self.env.render())
            self.infos_buffer.put(
                {
                    "action": action,
                    "reward": res[1],
                    "done": res[2] or res[3],
                    "training_steps": self.training_steps,
                    "testing": self.test,
                    "infos": res[-1],
                }
            )
        self._accumulated_reward+= res[1]
        if (res[2] or res[3]) and not self.test:
            self.tensorboard.add_scalars("score_per_level", {str(res[-1]["stage"]): self._accumulated_reward}, self.training_steps)
        return res

    def run_episode(self, test, *args, **kwargs):
        self.capturing = self.frames_buffer.qsize() < self.throttle_queue or test
        self._accumulated_reward = 0
        return super().run_episode(test, *args, **kwargs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog="Test streaming")
    parser.add_argument("-u", "--url", default="rtmp://localhost:1935/test")
    parser.add_argument("-s", "--socket", action="store_true")
    parser.add_argument("-r", "--framerate", default=30, type=int)
    parser.add_argument("-q", "--queue-size", default=500, type=int)
    parser.add_argument("-W", "--width", default=256, type=int)
    parser.add_argument("-H", "--height", default=256, type=int)
    args = parser.parse_args()
    frames_buffer = mp.Queue()
    infos_buffer = mp.Queue()
    stop_event = mp.Event()
    throttle_queue = args.queue_size
    stream_process = mp.Process(
        target=stream_to_pipes if args.socket else stream_frames,
        #target=stream_frames,
        args=(frames_buffer, infos_buffer, stop_event),
        kwargs={
            "fps": args.framerate,
            "url": args.url,
            "img_size": (args.width, args.height),
        },
        daemon=True,
    )
    stream_process.start()
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    def stop_process(*args):
        stop_event.set()
        stream_process.join()
        original_sigint_handler(*args)
    signal.signal(signal.SIGINT, stop_process)
    try:
        while not stop_event.is_set():
            frames_buffer.put(np.random.uniform(0, 255, (args.height, args.width, 3)).astype(np.uint8))
            infos_buffer.put({})
            if frames_buffer.qsize() > args.queue_size:
                time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
        stream_process.join()
    print("Ended main loop")

