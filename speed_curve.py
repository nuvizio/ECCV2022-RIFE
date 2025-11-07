#!/usr/bin/env python3
"""
Apply speed curve to interpolated video frames.

Usage:
    python speed_curve.py --input video.mp4 --normal-speed 1.0 --peak-speed 4.0 --output curved.mp4

This script assumes you've already generated a high frame count video using RIFE.
It will apply a smooth speed curve that ramps from normal speed to peak speed and back.
"""

import argparse
import cv2
import numpy as np
from tqdm import tqdm
import os

def ease_in_out_cubic(t):
    """Smooth ease-in-out curve (cubic)"""
    if t < 0.5:
        return 4 * t * t * t
    else:
        return 1 - pow(-2 * t + 2, 3) / 2

def generate_speed_curve(total_frames, normal_speed=1.0, peak_speed=4.0, curve_type='cubic'):
    """
    Generate a speed curve that ramps up to peak and back down.

    Args:
        total_frames: Number of input frames
        normal_speed: Normal playback speed (1.0 = realtime)
        peak_speed: Peak playback speed (higher = faster)
        curve_type: Type of easing curve

    Returns:
        Array of speed values for each frame
    """
    speeds = np.zeros(total_frames)

    for i in range(total_frames):
        # Normalize position (0 to 1)
        t = i / (total_frames - 1) if total_frames > 1 else 0

        # Create symmetric curve: ramp up, then ramp down
        # Peak is at the middle (t=0.5)
        if t <= 0.5:
            # First half: ease from normal to peak
            progress = ease_in_out_cubic(t * 2)  # 0 to 1
        else:
            # Second half: ease from peak back to normal
            progress = ease_in_out_cubic((1 - t) * 2)  # 1 to 0

        # Interpolate between normal and peak speed
        speeds[i] = normal_speed + (peak_speed - normal_speed) * progress

    return speeds

def apply_speed_curve(input_video, output_video, normal_speed=1.0, peak_speed=4.0,
                     target_fps=None, audio_file=None):
    """
    Apply speed curve to video by selecting frames according to the curve.

    Args:
        input_video: Path to input video (should be highly interpolated)
        output_video: Path to output video
        normal_speed: Normal playback speed
        peak_speed: Peak playback speed
        target_fps: Target output FPS (default: same as input)
        audio_file: Optional audio file to merge
    """
    # Open input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_video}")

    # Get video properties
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    total_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if target_fps is None:
        target_fps = input_fps

    print(f"Input: {total_input_frames} frames @ {input_fps:.2f} FPS")
    print(f"Output FPS: {target_fps:.2f}")
    print(f"Speed range: {normal_speed:.1f}x to {peak_speed:.1f}x")

    # Generate speed curve
    speeds = generate_speed_curve(total_input_frames, normal_speed, peak_speed)

    # Calculate which input frames to use for each output frame
    # We need to integrate the speed curve to get frame positions
    cumulative_time = 0
    output_frame_indices = []

    # Time per output frame (in input frame units)
    dt = 1.0

    for i in range(total_input_frames):
        # At this output frame, what's the input time position?
        input_frame_idx = cumulative_time

        if input_frame_idx < total_input_frames:
            output_frame_indices.append(input_frame_idx)

        # Current speed determines how much we advance in input frames
        if i < len(speeds):
            cumulative_time += speeds[i]

    output_frame_count = len(output_frame_indices)
    print(f"Output: {output_frame_count} frames (duration: {output_frame_count/target_fps:.2f}s)")

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, target_fps, (width, height))

    # Read all input frames (for random access)
    print("Reading input frames...")
    frames = []
    pbar = tqdm(total=total_input_frames, desc="Loading frames", unit="frame")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        pbar.update(1)
    pbar.close()
    cap.release()

    print(f"Loaded {len(frames)} frames")

    # Write output frames according to curve
    print("Applying speed curve...")
    pbar = tqdm(total=output_frame_count, desc="Writing frames", unit="frame")

    for frame_pos in output_frame_indices:
        # Get frame index and interpolation factor
        frame_idx = int(np.floor(frame_pos))
        alpha = frame_pos - frame_idx

        # Clamp to valid range
        frame_idx = max(0, min(frame_idx, len(frames) - 1))
        next_idx = min(frame_idx + 1, len(frames) - 1)

        # Linear interpolation between frames for smooth motion
        if alpha > 0.001 and frame_idx != next_idx:
            frame = cv2.addWeighted(frames[frame_idx], 1 - alpha, frames[next_idx], alpha, 0)
        else:
            frame = frames[frame_idx]

        out.write(frame)
        pbar.update(1)

    pbar.close()
    out.release()

    print(f"\n✓ Speed curve applied successfully!")
    print(f"Output saved to: {output_video}")

    # Merge audio if provided
    if audio_file and os.path.exists(audio_file):
        print("\nMerging audio...")
        temp_video = output_video.replace('.mp4', '_noaudio.mp4')
        os.rename(output_video, temp_video)

        # Get video duration to determine if we need to loop audio
        import subprocess
        result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                               '-of', 'default=noprint_wrappers=1:nokey=1', temp_video],
                              capture_output=True, text=True)
        video_duration = float(result.stdout.strip())

        # Get audio duration
        result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                               '-of', 'default=noprint_wrappers=1:nokey=1', audio_file],
                              capture_output=True, text=True)
        audio_duration = float(result.stdout.strip())

        if video_duration > audio_duration:
            print(f"  Video duration ({video_duration:.2f}s) > Audio duration ({audio_duration:.2f}s)")
            print(f"  Looping audio to match video duration...")
            cmd = f'ffmpeg -y -i "{temp_video}" -i "{audio_file}" -c:v copy -filter:a "aloop=loop=-1:size=2e+09" -shortest "{output_video}"'
        else:
            print(f"  Copying audio as-is...")
            cmd = f'ffmpeg -y -i "{temp_video}" -i "{audio_file}" -c:v copy -c:a copy -shortest "{output_video}"'

        os.system(cmd)

        if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
            os.remove(temp_video)
            print("✓ Audio merged successfully")
        else:
            os.rename(temp_video, output_video)
            print("⚠ Audio merge failed, kept video without audio")

def visualize_curve(total_frames, normal_speed, peak_speed):
    """Print a visualization of the speed curve"""
    speeds = generate_speed_curve(total_frames, normal_speed, peak_speed)

    print("\nSpeed Curve Visualization:")
    print("=" * 60)

    # Sample points for visualization
    sample_points = min(20, total_frames)
    indices = np.linspace(0, total_frames - 1, sample_points, dtype=int)

    max_speed = max(speeds)
    for idx in indices:
        speed = speeds[idx]
        bar_length = int((speed / max_speed) * 40)
        bar = "█" * bar_length
        print(f"Frame {idx:4d}: {bar} {speed:.2f}x")

    print("=" * 60)
    print(f"Average speed: {np.mean(speeds):.2f}x")
    print(f"Min speed: {np.min(speeds):.2f}x")
    print(f"Max speed: {np.max(speeds):.2f}x")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply smooth speed curve to interpolated video')
    parser.add_argument('--input', required=True, help='Input video (highly interpolated)')
    parser.add_argument('--output', required=True, help='Output video path')
    parser.add_argument('--normal-speed', type=float, default=1.0, help='Normal playback speed (default: 1.0)')
    parser.add_argument('--peak-speed', type=float, default=4.0, help='Peak playback speed (default: 4.0)')
    parser.add_argument('--fps', type=float, default=None, help='Output FPS (default: same as input)')
    parser.add_argument('--audio', type=str, default=None, help='Audio file to merge')
    parser.add_argument('--visualize', action='store_true', help='Show speed curve visualization')

    args = parser.parse_args()

    if args.visualize:
        # Quick visualization
        cap = cv2.VideoCapture(args.input)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        visualize_curve(total_frames, args.normal_speed, args.peak_speed)

    apply_speed_curve(
        args.input,
        args.output,
        normal_speed=args.normal_speed,
        peak_speed=args.peak_speed,
        target_fps=args.fps,
        audio_file=args.audio
    )
