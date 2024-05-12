from moviepy.editor import AudioFileClip
import json
import math
import subprocess as sp
import sys
import tempfile
from pathlib import Path

import cairo
import numpy as np
import tqdm

def convert_mp3_to_wav(mp3_path, wav_path):
    audio_clip = AudioFileClip(mp3_path)
    audio_clip.write_audiofile(wav_path, codec='pcm_s16le')  # pcm_s16le is the codec for .wav format
    audio_clip.close()
    print(f"Converted {mp3_path} to {wav_path}")


def read_info(media):
    """
    Return some info on the media file.
    """
    proc = sp.run([
        'ffprobe', "-loglevel", "panic",
        str(media), '-print_format', 'json', '-show_format', '-show_streams'
    ],
                  capture_output=True)
    if proc.returncode:
        raise IOError(f"{media} does not exist or is of a wrong type.")
    return json.loads(proc.stdout.decode('utf-8'))


def read_audio(audio, seek=None, duration=None):
    """
    Read the `audio` file, starting at `seek` (or 0) seconds for `duration` (or all)  seconds.
    Returns `float[channels, samples]`.
    """

    info = read_info(audio)
    channels = None
    stream = info['streams'][0]
    if stream["codec_type"] != "audio":
        raise ValueError(f"{audio} should contain only audio.")
    channels = stream['channels']
    samplerate = float(stream['sample_rate'])

    # Good old ffmpeg
    command = ['ffmpeg', '-y']
    command += ['-loglevel', 'panic']
    if seek is not None:
        command += ['-ss', str(seek)]
    command += ['-i', audio]
    if duration is not None:
        command += ['-t', str(duration)]
    command += ['-f', 'f32le']
    command += ['-']

    proc = sp.run(command, check=True, capture_output=True)
    wav = np.frombuffer(proc.stdout, dtype=np.float32)
    return wav.reshape(-1, channels).T, samplerate


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def envelope(wav, window, stride):
    """
    Extract the envelope of the waveform `wav` (float[samples]), using average pooling
    with `window` samples and the given `stride`.
    """
    # pos = np.pad(np.maximum(wav, 0), window // 2)
    wav = np.pad(wav, window // 2)
    out = []
    for off in range(0, len(wav) - window, stride):
        frame = wav[off:off + window]
        out.append(np.maximum(frame, 0).mean())
    out = np.array(out)
    # Some form of audio compressor based on the sigmoid.
    out = 1.9 * (sigmoid(2.5 * out) - 0.5)
    return out


def draw_env(envs, out, fg_color, bg_color, size):
    """
    draw a single frame using cairo and save it to the `out` file as png. envs is a list of envelopes over channels, each env
    is a float[bars] representing the height of the envelope to draw. Each entry will be represented by a bar.
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, *size)
    ctx = cairo.Context(surface)
    ctx.scale(*size)
    bg_color = (1, 1, 1)
    ctx.set_source_rgb(*bg_color)
    ctx.rectangle(0, 0, 1, 1)
    ctx.fill()

    T = len(envs[0]) # Numbert of time steps
    pad_ratio = 0.1 # spacing ratio between 2 bars
    width = 1. / (T * (1 + 2 * pad_ratio))
    pad = pad_ratio * width
    delta = 2 * pad + width

    ctx.set_line_width(width)
    for step in range(T):
        if step % 7 == 1:
            fg_color = (1.0, 0.0, 0.0)
        elif step % 7 == 2:
            fg_color = (1.0, 0.65, 0.0)
        elif step % 7 == 3:
            fg_color = (1.0, 1.0, 0.0)
        elif step % 7 == 4:
            fg_color = (0.0, 1.0, 0.0)
        elif step % 7 == 5:
            fg_color = (0.0, 0.0, 1.0)
        elif step % 7 == 6:
            fg_color = (0.29, 0.0, 0.51)
        elif step % 7 == 0:
            fg_color = (0.93, 0.51, 0.93)
        half = 0.5 * envs[0][step] # (semi-)height of the bar
        midrule = 1/2 # midrule of i-th wave
        ctx.set_source_rgb(*fg_color)
        ctx.move_to(pad + step * delta, midrule - half)
        ctx.line_to(pad + step * delta, midrule)
        ctx.stroke()
        ctx.set_source_rgba(*fg_color, 0.8)
        ctx.move_to(pad + step * delta, midrule)
        ctx.line_to(pad + step * delta, midrule + 0.9 * half)
        ctx.stroke()

    surface.write_to_png(out)

def interpole(x1, y1, x2, y2, x):
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

def visualize(audio,
              tmp,
              out = Path("generated_background.mp4"),
              seek=None,
              duration=None,
              rate=60,
              bars=50,
              speed=4,
              time=0.4,
              oversample=3,
              fg_color=(1.0, 0.0, 0.0),
              bg_color=(1, 1, 1),
              size=(1710, 1080),
              stereo=False,
              ):
    """
    Generate the visualisation for the `audio` file, using a `tmp` folder and saving the final
    video in `out`.
    `seek` and `durations` gives the extract location if any.
    `rate` is the framerate of the output video.

    `bars` is the number of bars in the animation.
    `speed` is the base speed of transition. Depending on volume, actual speed will vary
        between 0.5 and 2 times it.
    `time` amount of audio shown at once on a frame.
    `oversample` higher values will lead to more frequent changes.
    `fg_color` is the rgb color to use for the foreground.
    `bg_color` is the rgb color to use for the background.
    `size` is the `(width, height)` in pixels to generate.
    `stereo` is whether to create 2 waves.
    """
    try:
        wav, sr = read_audio(audio, seek=seek, duration=duration)
    except (IOError, ValueError) as err:
        print(err, file=sys.stderr)
        raise
    # wavs is a list of wav over channels
    wav = wav.mean(0)
    wav/wav.std()

    window = int(sr * time / bars)
    stride = int(window / oversample)
    # envs is a list of env over channels
    env = envelope(wav, window, stride)
    env = np.pad(env, (bars // 2, 2 * bars))

    duration = len(wav) / sr
    frames = int(rate * duration)
    smooth = np.hanning(bars)

    print("Generating the frames...")
    for idx in tqdm.tqdm(range(frames), unit=" frames", ncols=80):
        pos = (((idx / rate)) * sr) / stride / bars
        off = int(pos)
        loc = pos - off
        denvs = []
        env1 = env[off * bars:(off + 1) * bars]
        env2 = env[(off + 1) * bars:(off + 2) * bars]

        # we want loud parts to be updated faster
        maxvol = math.log10(1e-4 + env2.max()) * 10
        speedup = np.clip(interpole(-6, 0.5, 0, 2, maxvol), 0.5, 2)
        w = sigmoid(speed * speedup * (loc - 0.5))
        denv = (1 - w) * env1 + w * env2
        denv *= smooth
        denvs.append(denv)
        draw_env(denvs, tmp / f"{idx:06d}.png", fg_color, bg_color, size)

    audio_cmd = []
    if seek is not None:
        audio_cmd += ["-ss", str(seek)]
    audio_cmd += ["-i", audio.resolve()]
    if duration is not None:
        audio_cmd += ["-t", str(duration)]
    print("Encoding the animation video... ")
    sp.run([
        "ffmpeg", "-y", "-loglevel", "panic", "-r",
        str(rate), "-f", "image2", "-s", f"{size[0]}x{size[1]}", "-i", "%06d.png"
    ] + audio_cmd + [
        "-c:a", "aac", "-vcodec", "libx264", "-crf", "10", "-pix_fmt", "yuv420p",
        out.resolve()
    ],
           check=True,
           cwd=tmp)

def main():
    mp3_path = 'audio.mp3'
    wav_path = "output_audio.wav"
    convert_mp3_to_wav(mp3_path, wav_path)
    with tempfile.TemporaryDirectory() as tmp:
        visualize(Path("output_audio.wav"),
                  Path(tmp),)


if __name__ == "__main__":
    _is_main = True
    main()