#!/usr/bin/env python
"""
live_camera.py
--------------
Live CrossguardVision pipeline on the Raspberry Pi camera.

Captures frames from the Pi camera (Picamera2), runs the YOLO-Seg model plus
the spatial_reasoning geometry on every frame, and overlays each person's risk
(high / medium / low). Because the Pi is usually driven head-less over SSH, the
annotated video is served as an MJPEG stream:

    open  http://<pi-ip>:8000/  in a browser on the same network.

All geometry + drawing is imported from spatial_reasoning.py UNCHANGED -- this
file only adds frame capture, an FPS/risk HUD, and the stream/display sinks.

Examples
--------
    # Pi camera -> MJPEG stream on :8000 (CPU, imgsz 640):
    python live_camera.py --weights yolo11n-seg-crossguard.pt

    # show on a monitor wired to the Pi instead of streaming:
    python live_camera.py --weights yolo11n-seg-crossguard.pt --display

    # dry-run the loop on a video file / USB cam (no Pi camera needed):
    python live_camera.py --weights yolo11n-seg-crossguard.pt --source clip.mp4
    python live_camera.py --weights yolo11n-seg-crossguard.pt --source 0
"""
import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2

# Reuse the audited geometry + annotation from the batch tool (same folder).
from spatial_reasoning import (
    crosswalk_mask_from_result, person_boxes_from_result,
    crossing_zone_from_mask, classify_people, annotate, _counts,
    HIGH_RISK, MEDIUM_RISK, LOW_RISK,
)


# ----------------------------------------------------------------------------
# Per-frame pipeline (mirrors spatial_reasoning.predict_states but for an
# in-memory BGR frame instead of a file path).
# ----------------------------------------------------------------------------
def process_frame(model, frame, conf, imgsz, crossing_k, waiting_k, min_h_frac,
                  use_hull, min_arms, hull_erode_frac, device):
    result = model.predict(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
    cw_mask = crosswalk_mask_from_result(result, frame.shape)
    boxes = person_boxes_from_result(result)
    zone, outline = (crossing_zone_from_mask(cw_mask, min_arms=min_arms, erode_frac=hull_erode_frac)
                     if use_hull else (None, None))
    people = classify_people(boxes, cw_mask, crossing_k, waiting_k, min_h_frac,
                             use_hull=use_hull, crossing_zone=zone)
    out = annotate(frame, cw_mask, people, hull=outline)
    return out, people


def draw_hud(img, fps, infer_ms, counts):
    txt = (f"FPS {fps:4.1f}  {infer_ms:4.0f}ms   "
           f"HIGH {counts[HIGH_RISK]}  MED {counts[MEDIUM_RISK]}  LOW {counts[LOW_RISK]}")
    cv2.rectangle(img, (0, 0), (8 + 11 * len(txt), 30), (0, 0, 0), -1)
    cv2.putText(img, txt, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ----------------------------------------------------------------------------
# Frame sources
# ----------------------------------------------------------------------------
def picamera2_frames(width, height):
    """Yield BGR frames from the Pi camera. NOTE: Picamera2 'RGB888' returns an
    array whose channel order is actually B,G,R, so it is already cv2-compatible."""
    try:
        from picamera2 import Picamera2
    except ImportError as e:
        raise SystemExit(
            "picamera2 not importable inside the venv.\n"
            "Fix: set 'include-system-site-packages = true' in venv/pyvenv.cfg, e.g.\n"
            "  sed -i 's/include-system-site-packages = false/include-system-site-packages = true/'"
            " ~/crossguard/venv/pyvenv.cfg\n"
            f"(original error: {e})")
    picam2 = Picamera2()
    cfg = picam2.create_video_configuration(main={"format": "RGB888", "size": (width, height)})
    picam2.configure(cfg)
    picam2.start()
    try:
        while True:
            yield picam2.capture_array()
    finally:
        picam2.stop()


def opencv_frames(source):
    """Yield BGR frames from a video file or a USB/V4L2 camera index."""
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"could not open source: {source}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


# ----------------------------------------------------------------------------
# MJPEG streaming server (multipart/x-mixed-replace)
# ----------------------------------------------------------------------------
class _Stream:
    def __init__(self):
        self.cond = threading.Condition()
        self.jpeg = None

    def publish(self, jpeg):
        with self.cond:
            self.jpeg = jpeg
            self.cond.notify_all()

    def wait(self):
        with self.cond:
            self.cond.wait()
            return self.jpeg


STREAM = _Stream()
PAGE = (b"<html><head><title>CrossguardVision live</title></head>"
        b"<body style='margin:0;background:#111'>"
        b"<img src='/stream' style='width:100%;height:auto'></body></html>")


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while True:
                    jpeg = STREAM.wait()
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--FRAME\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def log_message(self, *args):  # silence per-request console spam
        pass


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_stream_server(port):
    srv = _ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="CrossguardVision live camera pipeline")
    ap.add_argument("--weights", required=True, help="YOLO-Seg weights (best.pt)")
    ap.add_argument("--source", default="picamera2",
                    help="'picamera2' (default), a video file path, or a camera index like 0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="model input size; 896=trained accuracy, 640=balanced, 480=faster")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--port", type=int, default=8000, help="MJPEG stream port")
    ap.add_argument("--display", action="store_true",
                    help="cv2.imshow on a locally-attached monitor instead of streaming")
    ap.add_argument("--jpeg-quality", type=int, default=80)
    # spatial knobs (same defaults/semantics as spatial_reasoning.py)
    ap.add_argument("--crossing-k", type=float, default=0.10)
    ap.add_argument("--waiting-k", type=float, default=0.50)
    ap.add_argument("--min-h-frac", type=float, default=0.05)
    ap.add_argument("--no-hull", action="store_true")
    ap.add_argument("--min-arms", type=int, default=2)
    ap.add_argument("--hull-erode-frac", type=float, default=0.0)
    args = ap.parse_args()

    from ultralytics import YOLO
    print(f"loading {args.weights} (device={args.device}) ...")
    model = YOLO(args.weights)

    # frame source
    if args.source == "picamera2":
        frames = picamera2_frames(args.width, args.height)
    else:
        frames = opencv_frames(args.source)

    # warm up (first inference is much slower than steady state)
    first = next(frames)
    process_frame(model, first, args.conf, args.imgsz, args.crossing_k, args.waiting_k,
                  args.min_h_frac, not args.no_hull, args.min_arms, args.hull_erode_frac, args.device)

    srv = None
    if not args.display:
        srv = start_stream_server(args.port)
        print(f"streaming -> http://<this-pi-ip>:{args.port}/   (Ctrl+C to stop)")
    else:
        print("showing local window (Ctrl+C or 'q' to stop)")

    fps, last_log = 0.0, 0.0
    try:
        for frame in frames:
            t0 = time.time()
            out, people = process_frame(
                model, frame, args.conf, args.imgsz, args.crossing_k, args.waiting_k,
                args.min_h_frac, not args.no_hull, args.min_arms, args.hull_erode_frac, args.device)
            dt = time.time() - t0
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else (1.0 / dt)
            counts = _counts(people)
            out = draw_hud(out, fps, dt * 1000, counts)

            if args.display:
                cv2.imshow("CrossguardVision", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                if ok:
                    STREAM.publish(buf.tobytes())

            if t0 - last_log >= 1.0:
                print(f"  {fps:4.1f} FPS  {dt*1000:4.0f} ms/frame   "
                      f"persons {len(people)}  H/M/L "
                      f"{counts[HIGH_RISK]}/{counts[MEDIUM_RISK]}/{counts[LOW_RISK]}")
                last_log = t0
    except KeyboardInterrupt:
        print("\nstopping ...")
    finally:
        if srv is not None:
            srv.shutdown()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
