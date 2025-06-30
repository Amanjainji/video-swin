import argparse
import os
import os.path as osp
import cv2
import numpy as np
import torch
import webcolors
from mmengine.config import Config
from mmengine.config import DictAction

from mmaction.apis import inference_recognizer, init_recognizer
import random

# For reproducibility
os.environ['PYTHONHASHSEED'] = '0'
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file/url')
    parser.add_argument('video', help='video file/url or rawframes directory')
    parser.add_argument('label', help='label file')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--fps',
        default=30,
        type=int,
        help='fps value for output video')
    parser.add_argument(
        '--font-scale',
        default=0.5,
        type=float,
        help='font scale for label text')
    parser.add_argument(
        '--font-color',
        default='white',
        help='font color for label text')
    parser.add_argument(
        '--target-resolution',
        nargs=2,
        default=None,
        type=int,
        help='Target resolution (w h). Use -1 to keep aspect ratio.')
    parser.add_argument(
        '--resize-algorithm',
        default='bicubic',
        help='resize algorithm for output')
    parser.add_argument('--out-filename', default=None, help='output filename')
    return parser.parse_args()


def get_output(video_path,
               out_filename,
               label,
               fps=30,
               font_scale=0.5,
               font_color='white',
               target_resolution=None,
               resize_algorithm='bicubic',
               use_frames=False):
    """Generate output video with overlayed label text."""

    if video_path.startswith(('http://', 'https://')):
        raise NotImplementedError('URL input not supported.')

    try:
        from moviepy.editor import ImageSequenceClip
    except ImportError:
        raise ImportError('Please install moviepy to enable output generation.')

    # Read frames
    if use_frames:
        frame_list = sorted([osp.join(video_path, x) for x in os.listdir(video_path)])
        frames = [cv2.imread(x) for x in frame_list]
    else:
        frames = []
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from {video_path}")

    # Resize
    if target_resolution:
        w, h = target_resolution
        frame_h, frame_w, _ = frames[0].shape
        if w == -1:
            w = int(h / frame_h * frame_w)
        if h == -1:
            h = int(w / frame_w * frame_h)
        frames = [cv2.resize(f, (w, h)) for f in frames]

    # Add label text
    textsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)[0]
    textheight = textsize[1]
    padding = 10
    location = (padding, padding + textheight)

    if isinstance(font_color, str):
        font_color = webcolors.name_to_rgb(font_color)[::-1]

    for frame in frames:
        cv2.putText(frame, label, location, cv2.FONT_HERSHEY_DUPLEX,
                    font_scale, font_color, 1)

    # Convert to RGB for MoviePy
    frames = [f[..., ::-1] for f in frames]
    video_clip = ImageSequenceClip(frames, fps=fps)

    # Write out
    ext = osp.splitext(out_filename)[1][1:]
    if ext == 'gif':
        video_clip.write_gif(out_filename)
    else:
        video_clip.write_videofile(out_filename, remove_temp=True)


def main():
    args = parse_args()
    device = torch.device(args.device)

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)

    model = init_recognizer(
        cfg, args.checkpoint, device=device)

    output_layer_names = None

    if output_layer_names:
        results, _ = inference_recognizer(
            model,
            args.video,
            args.label,
            outputs=output_layer_names
        )
    else:
        results = inference_recognizer(
            model,
            args.video,
            args.label,
        )

    print('Top-5 labels with scores:')
    for result in results:
        print(f'{result[0]}: {result[1]}')

    if args.out_filename is not None:
        if args.target_resolution is not None:
            w, h = args.target_resolution
            if w == -1:
                assert h > 0
            if h == -1:
                assert w > 0
            args.target_resolution = (w, h)

        get_output(
            args.video,
            args.out_filename,
            results[0][0],
            fps=args.fps,
            font_scale=args.font_scale,
            font_color=args.font_color,
            target_resolution=args.target_resolution,
            resize_algorithm=args.resize_algorithm,
        )


if __name__ == '__main__':
    main()
