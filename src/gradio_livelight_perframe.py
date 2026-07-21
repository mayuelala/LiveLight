"""Gradio V4: Record-then-Render demo for LightLive.

Upload video → 3D depth-mesh scene with draggable light →
Record button plays video while user drags light, capturing per-frame trajectory →
Render button runs offline rendering with 12-frame-offset trajectory →
Result: input / relit videos play at paper's 18.x fps, perfectly synced.

Launch:
    python gradio_livelight_perframe.py --port 7864
"""

from __future__ import annotations

import argparse
import base64
import gc
import io
import json as _json
import multiprocessing as _mp
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import socket as _socket
_orig_create_connection = _socket.create_connection

def _fast_fail_connection(address, *args, **kwargs):
    host = address[0] if isinstance(address, tuple) else address
    if isinstance(host, str) and host not in ("127.0.0.1", "localhost", "0.0.0.0", "::1"):
        raise OSError(f"[offline-guard] blocked outbound connection to {host}")
    return _orig_create_connection(address, *args, **kwargs)

_socket.create_connection = _fast_fail_connection

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

from src.livelight_wrapper_perframe_ref import (
    FrameState,
    LightParams,
    RelightPerFrameRefLive,
    pack_light,
    read_anchor_depth,
)

# ---------------------------------------------------------------------------
# Shared state for light position (multiprocessing-safe)
# ---------------------------------------------------------------------------
_light_array = _mp.Array('d', [0.5, 0.5, 1.0])

TRAJECTORY_PATH = "/tmp/_lightlive_trajectory.json"

def get_current_light():
    return _light_array[0], _light_array[1], _light_array[2]

def set_current_light(u, v, z):
    _light_array[0] = u
    _light_array[1] = v
    _light_array[2] = z

# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, max_frames: int = 0) -> Tuple[List[Image.Image], float]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
        if max_frames > 0 and len(frames) >= max_frames:
            break
    cap.release()
    return frames, fps


def frames_to_video(frames: List[Image.Image], fps: float, output_path: str):
    if not frames:
        return
    first = frames[0]
    if isinstance(first, Image.Image):
        w, h = first.size
    else:
        arr0 = np.array(first)
        h, w = arr0.shape[:2]
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "18",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for frame in frames:
        if isinstance(frame, Image.Image):
            rgb = np.array(frame.convert("RGB"), dtype=np.uint8)
        else:
            rgb = np.array(frame, dtype=np.uint8)
        proc.stdin.write(rgb.tobytes())
    proc.stdin.close()
    proc.wait()


def reencode_video_fps(input_path: str, target_fps: float, output_path: str):
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-r", str(target_fps),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "18",
        output_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------------------------------------------------------------------------
# Encode scene data for Three.js
# ---------------------------------------------------------------------------

def encode_scene_data(first_pil: Image.Image, depth_map: np.ndarray,
                      intrinsics: dict, anchor_depth: float) -> str:
    buf = io.BytesIO()
    first_pil.save(buf, format="PNG")
    rgb_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    depth_f32 = depth_map.astype(np.float32)
    depth_b64 = base64.b64encode(depth_f32.tobytes()).decode("ascii")

    return _json.dumps({
        "rgb": rgb_b64,
        "depth": depth_b64,
        "depth_h": depth_map.shape[0],
        "depth_w": depth_map.shape[1],
        "fx": intrinsics["fx"],
        "fy": intrinsics["fy"],
        "cx": intrinsics["cx"],
        "cy": intrinsics["cy"],
        "anchor_depth": float(anchor_depth),
    })


# ---------------------------------------------------------------------------
# Offline render with pre-recorded trajectory
# ---------------------------------------------------------------------------

LIGHT_OFFSET_FRAMES = 12

@torch.inference_mode()
def render_with_trajectory(
    runner: RelightPerFrameRefLive,
    frame_pils: List[Image.Image],
    depth_map: np.ndarray,
    ref_latents: torch.Tensor,
    clip_embeds: torch.Tensor,
    output_fps: float,
    trajectory: List[Tuple[float, float, float]],
    intensity: float, r: float, g: float, b: float,
    status_callback=None,
):
    tw = runner.temporal_window_size
    num_frames = len(frame_pils)
    runner.reset()
    all_preds: List[Image.Image] = []

    def _get_offset_light(frame_idx):
        idx = min(frame_idx, len(trajectory) - 1)
        return trajectory[idx]

    def _build_packed_lights_for_range(start_frame, count):
        lights = []
        for i in range(count):
            u, v, z = _get_offset_light(start_frame + i)
            params = LightParams(
                light_u=u, light_v=v, light_z_rel=z,
                light_intensity=intensity, light_r=r, light_g=g, light_b=b,
            )
            mpli, _ = runner._render_single_frame_mpli(depth_map, params)
            lights.append(pack_light(mpli))
        return lights

    def _report(msg):
        if status_callback:
            status_callback(msg)
        print(f"[render] {msg}", flush=True)

    # --- First chunk ---
    first_end = min(num_frames, tw)
    first_count = first_end

    packed_first = _build_packed_lights_for_range(0, first_count)

    first_states = runner._build_frame_states_batch(
        frame_images=frame_pils[:first_end],
        packed_lights=packed_first,
        precomputed_ref_latents=ref_latents[:first_end],
        precomputed_clip_embeds=clip_embeds[:first_end],
    )

    warmup_chunks = runner.temporal_adaptive_step - 1
    pad_state = first_states[0]
    for _ in range(warmup_chunks * tw):
        runner.frame_history.append(pad_state)

    ref_latent_single = pad_state.ref_latent
    padding_count = warmup_chunks * tw
    init_latents = ref_latent_single.unsqueeze(2).repeat(1, 1, padding_count, 1, 1).to(
        device=runner.device, dtype=runner.dtype
    )
    init_noise = torch.randn(
        init_latents.shape, generator=runner.generator, device=runner.device, dtype=runner.dtype
    )
    init_timesteps = list(reversed(runner.timesteps_list))
    init_timesteps_vec = torch.tensor(
        init_timesteps, device=runner.device, dtype=torch.long
    ).repeat_interleave(tw, dim=0)[:padding_count]
    noisy_init = runner.scheduler.add_noise(
        init_latents.permute(0, 2, 1, 3, 4).reshape(-1, init_latents.shape[1], init_latents.shape[3], init_latents.shape[4]),
        init_noise.permute(0, 2, 1, 3, 4).reshape(-1, init_noise.shape[1], init_noise.shape[3], init_noise.shape[4]),
        init_timesteps_vec,
    )
    noisy_init = noisy_init.reshape(1, padding_count, init_latents.shape[1], init_latents.shape[3], init_latents.shape[4]).permute(0, 2, 1, 3, 4)
    runner.stage3_chunk_latents = torch.cat(
        [noisy_init.to(dtype=runner.dtype), runner._sample_stage3_noise(tw)], dim=2
    )

    for state in first_states:
        runner.frame_history.append(state)
        runner.real_frame_count += 1
    if first_count < tw:
        for _ in range(tw - first_count):
            runner.frame_history.append(first_states[-1])

    chunk_preds, _ = runner._render_current_chunk(tw)
    all_preds.extend(chunk_preds[:first_count])
    _report(f"Chunk 0/{num_frames // tw}")

    # --- Subsequent chunks ---
    chunk_idx = 1
    for chunk_start in range(tw, num_frames, tw):
        chunk_end = min(num_frames, chunk_start + tw)
        real_count = chunk_end - chunk_start

        packed_chunk = _build_packed_lights_for_range(chunk_start, real_count)

        chunk_states = runner._build_frame_states_batch(
            frame_images=frame_pils[chunk_start:chunk_end],
            packed_lights=packed_chunk,
            precomputed_ref_latents=ref_latents[chunk_start:chunk_end],
            precomputed_clip_embeds=clip_embeds[chunk_start:chunk_end],
        )

        for state in chunk_states:
            runner.frame_history.append(state)
            runner.real_frame_count += 1

        if real_count < tw:
            for _ in range(tw - real_count):
                runner.frame_history.append(chunk_states[-1])

        chunk_preds, _ = runner._render_current_chunk(tw)
        all_preds.extend(chunk_preds[:real_count])
        _report(f"Chunk {chunk_idx}/{(num_frames + tw - 1) // tw}")
        chunk_idx += 1

    # Encode final video - trim first 12 warmup frames
    _report("Encoding video...")
    trimmed_preds = all_preds[LIGHT_OFFSET_FRAMES:num_frames]
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    frames_to_video(trimmed_preds, output_fps, tmp.name)
    _report(f"Output: {len(trimmed_preds)} frames (trimmed {LIGHT_OFFSET_FRAMES} warmup)")

    runner.reset()
    gc.collect()
    torch.cuda.empty_cache()
    _report("Done")
    return tmp.name


def _pred_to_np(pred) -> np.ndarray:
    if isinstance(pred, Image.Image):
        return np.array(pred.convert("RGB"), dtype=np.uint8)
    return np.array(pred, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Three.js scene HTML
# ---------------------------------------------------------------------------

THREEJS_CONTAINER = '<div id="threejs-container" style="width:100%;height:480px;position:relative;background:#e8e8ee;border-radius:8px;"><div id="scene-status" style="position:absolute;top:10px;left:10px;color:#333;font:14px monospace;z-index:10;background:rgba(255,255,255,0.7);padding:4px 8px;border-radius:4px;">Upload a video to load 3D scene</div><div id="light-info" style="position:absolute;bottom:10px;left:10px;color:#885500;font:13px monospace;z-index:10;background:rgba(255,255,255,0.7);padding:4px 8px;border-radius:4px;"></div></div>'

INIT_THREEJS_JS = """
() => {
    function loadScript(url, cb) {
        if (window[url]) { cb(); return; }
        var s = document.createElement('script');
        s.src = url;
        s.onload = function() { window[url] = true; cb(); };
        s.onerror = function() { console.error('Failed to load', url); };
        document.head.appendChild(s);
    }

    function waitForContainer(cb) {
        var el = document.getElementById('threejs-container');
        if (el) { cb(el); return; }
        setTimeout(function() { waitForContainer(cb); }, 300);
    }

    var cdnBase = 'https://cdn.jsdelivr.net/npm/three@0.128.0/';
    loadScript(cdnBase + 'build/three.min.js', function() {
    loadScript(cdnBase + 'examples/js/controls/OrbitControls.js', function() {
    loadScript(cdnBase + 'examples/js/controls/TransformControls.js', function() {
        waitForContainer(function(container) { initRelightScene(container); });
    }); }); });

    function initRelightScene(container) {
        var oldCanvas = container.querySelector('canvas');
        if (oldCanvas) oldCanvas.remove();

        var statusEl = document.getElementById('scene-status');
        var lightInfoEl = document.getElementById('light-info');
        statusEl.textContent = 'Three.js loaded. Upload a video.';

        var renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.appendChild(renderer.domElement);

        var scene = new THREE.Scene();
        scene.background = new THREE.Color(0xe8e8ee);

        var camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.01, 50000);
        camera.position.set(0, 0, 0.01);

        var orbit = new THREE.OrbitControls(camera, renderer.domElement);
        orbit.enableDamping = true;
        orbit.dampingFactor = 0.1;

        var lightGroup = new THREE.Group();
        var bulbGeo = new THREE.SphereGeometry(1, 16, 16);
        var bulbMat = new THREE.MeshBasicMaterial({ color: 0xffee44 });
        lightGroup.add(new THREE.Mesh(bulbGeo, bulbMat));
        var baseGeo = new THREE.CylinderGeometry(0.45, 0.55, 0.7, 12);
        var baseMat = new THREE.MeshBasicMaterial({ color: 0x888888 });
        var baseMesh = new THREE.Mesh(baseGeo, baseMat);
        baseMesh.position.y = -1.1;
        lightGroup.add(baseMesh);
        lightGroup.visible = false;
        scene.add(lightGroup);

        var lineMat = new THREE.LineBasicMaterial({ color: 0xffdd00, transparent: true, opacity: 0.4 });
        var lineGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
        var lightLine = new THREE.Line(lineGeo, lineMat);
        lightLine.visible = false;
        scene.add(lightLine);

        var transformCtrl = new THREE.TransformControls(camera, renderer.domElement);
        transformCtrl.setMode('translate');
        transformCtrl.setSize(0.8);
        transformCtrl.addEventListener('dragging-changed', function(e) { orbit.enabled = !e.value; });
        scene.add(transformCtrl);

        var sceneParams = null;
        var meshCenter = new THREE.Vector3();

        transformCtrl.addEventListener('change', function() {
            if (!sceneParams) return;
            updateLightInfo();
        });

        function posToUVZ(pos) {
            if (!sceneParams) return { u: 0.5, v: 0.5, z: 1.0 };
            var d = -pos.z;
            if (d <= 0) return { u: 0.5, v: 0.5, z: 1.0 };
            var px = pos.x * sceneParams.fx / d + sceneParams.cx;
            var py = -pos.y * sceneParams.fy / d + sceneParams.cy;
            return {
                u: Math.max(0, Math.min(1, px / Math.max(sceneParams.depth_w - 1, 1))),
                v: Math.max(0, Math.min(1, py / Math.max(sceneParams.depth_h - 1, 1))),
                z: Math.max(0.1, Math.min(3.0, d / sceneParams.anchor_depth))
            };
        }

        function uvzToPos(u, v, z_rel) {
            if (!sceneParams) return new THREE.Vector3(0, 0, -100);
            var d = z_rel * sceneParams.anchor_depth;
            var px = u * (sceneParams.depth_w - 1);
            var py = v * (sceneParams.depth_h - 1);
            return new THREE.Vector3(
                (px - sceneParams.cx) / sceneParams.fx * d,
                -(py - sceneParams.cy) / sceneParams.fy * d,
                -d
            );
        }

        function updateLightInfo() {
            var uvz = posToUVZ(lightGroup.position);
            lightInfoEl.textContent = 'Light: u=' + uvz.u.toFixed(3) + ' v=' + uvz.v.toFixed(3) + ' z=' + uvz.z.toFixed(3);
            var positions = lightLine.geometry.attributes.position;
            if (positions) {
                positions.setXYZ(0, lightGroup.position.x, lightGroup.position.y, lightGroup.position.z);
                positions.setXYZ(1, meshCenter.x, meshCenter.y, meshCenter.z);
                positions.needsUpdate = true;
            }
        }

        // --- Light position sync (for live preview, sendBeacon) ---
        var _lightUrl = 'http://' + window.location.hostname + ':' + (parseInt(window.location.port) + 1) + '/light';
        var _lastSentU = -1, _lastSentV = -1, _lastSentZ = -1;
        setInterval(function() {
            if (!sceneParams || !lightGroup.visible) return;
            var uvz = posToUVZ(lightGroup.position);
            if (Math.abs(uvz.u - _lastSentU) < 0.001 && Math.abs(uvz.v - _lastSentV) < 0.001 && Math.abs(uvz.z - _lastSentZ) < 0.001) return;
            _lastSentU = uvz.u; _lastSentV = uvz.v; _lastSentZ = uvz.z;
            navigator.sendBeacon(_lightUrl, JSON.stringify(uvz));
        }, 100);

        // --- Trajectory recording ---
        window._recordedTrajectory = [];
        window._isRecording = false;
        window._recordFps = 18.0;

        window._startRecording = function(fps) {
            window._recordFps = fps || 18.0;
            window._recordedTrajectory = [];
            window._isRecording = true;
            window._lastRecordedFrame = -1;

            // Find the video element in the input component
            var videoEl = document.querySelector('#input_video video');
            if (!videoEl) {
                console.error('No video element found');
                window._isRecording = false;
                return;
            }
            videoEl.currentTime = 0;
            videoEl.play();

            // Capture Three.js canvas as video
            var canvasEl = document.querySelector('#threejs-container canvas');
            if (canvasEl) {
                try {
                    var stream = canvasEl.captureStream(30);
                    var opts = {};
                    if (MediaRecorder.isTypeSupported('video/webm;codecs=vp9')) opts.mimeType = 'video/webm;codecs=vp9';
                    else if (MediaRecorder.isTypeSupported('video/webm')) opts.mimeType = 'video/webm';
                    window._mediaRecorder = new MediaRecorder(stream, opts);
                    window._canvasChunks = [];
                    window._mediaRecorder.ondataavailable = function(e) {
                        if (e.data && e.data.size > 0) window._canvasChunks.push(e.data);
                    };
                    window._mediaRecorder.onstop = function() {
                        var blob = new Blob(window._canvasChunks, {type: window._mediaRecorder.mimeType || 'video/webm'});
                        if (window._canvasBlobUrl) URL.revokeObjectURL(window._canvasBlobUrl);
                        window._canvasBlobUrl = URL.createObjectURL(blob);
                        var vid = document.getElementById('canvas_video_el');
                        if (vid) {
                            vid.src = window._canvasBlobUrl;
                            vid.style.display = 'block';
                            vid.load();
                        }
                        console.log('Canvas video captured:', window._canvasChunks.length, 'chunks, blobUrl:', window._canvasBlobUrl);
                    };
                    window._mediaRecorder.start(100);
                    console.log('MediaRecorder started on canvas');
                } catch(e) { console.warn('Canvas capture not supported:', e); }
            }

            function recordLoop() {
                if (!window._isRecording) return;
                if (videoEl.ended || videoEl.paused) {
                    window._isRecording = false;
                    // Stop canvas recording
                    if (window._mediaRecorder && window._mediaRecorder.state !== 'inactive') {
                        window._mediaRecorder.stop();
                    }
                    // Send trajectory to server
                    var trajUrl = 'http://' + window.location.hostname + ':' + (parseInt(window.location.port) + 1) + '/trajectory';
                    fetch(trajUrl, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(window._recordedTrajectory)
                    }).then(function() {
                        console.log('Trajectory uploaded:', window._recordedTrajectory.length, 'frames');
                    }).catch(function(e) { console.error('Trajectory upload failed', e); });
                    return;
                }
                var frameIdx = Math.floor(videoEl.currentTime * window._recordFps);
                if (frameIdx > window._lastRecordedFrame) {
                    var uvz = posToUVZ(lightGroup.position);
                    // Fill any skipped frames
                    for (var fi = window._lastRecordedFrame + 1; fi <= frameIdx; fi++) {
                        window._recordedTrajectory.push([uvz.u, uvz.v, uvz.z]);
                    }
                    window._lastRecordedFrame = frameIdx;
                }
                requestAnimationFrame(recordLoop);
            }
            requestAnimationFrame(recordLoop);
        };

        window._stopRecording = function() {
            window._isRecording = false;
        };

        // --- Play All: sync three videos ---
        window._playAll = function() {
            // Hide 3D editor, show canvas video
            var container = document.getElementById('threejs-container');
            if (container) container.style.display = 'none';

            var canvasVid = document.getElementById('canvas_video_el');
            console.log('PlayAll: canvasVid=', canvasVid, 'blobUrl=', window._canvasBlobUrl);
            if (canvasVid && window._canvasBlobUrl) {
                canvasVid.src = window._canvasBlobUrl;
                canvasVid.style.display = 'block';
            }

            var v1 = document.querySelector('#input_video video');
            var v2 = (canvasVid && window._canvasBlobUrl) ? canvasVid : null;
            var v3 = document.querySelector('#output_video video');
            if (v3) v3.style.visibility = 'visible';
            console.log('PlayAll: v1=', v1, 'v2=', v2, 'v3=', v3);
            var all = [v1, v2, v3].filter(Boolean);

            // Canvas trim: skip first N frames (tune this number to align light)
            var CANVAS_TRIM_FRAMES = 6;
            var canvasOffset = CANVAS_TRIM_FRAMES / (window._recordFps || 18.18);
            console.log('PlayAll:', all.length, 'videos, canvas skip', CANVAS_TRIM_FRAMES, 'frames =', canvasOffset.toFixed(3) + 's');

            // Stop previous sync
            if (window._syncRAF) { cancelAnimationFrame(window._syncRAF); window._syncRAF = null; }

            // Seek: input & output to 0, canvas skips ahead
            all.forEach(function(v) { v.muted = true; v.pause(); });
            if (v1) v1.currentTime = 0.001;
            if (v3) v3.currentTime = 0.001;
            if (v2) v2.currentTime = canvasOffset;

            setTimeout(function() {
                all.forEach(function(v) { v.play().catch(function(e){ console.warn('play err', e); }); });

                // Sync loop: keep videos aligned
                var master = v3 || v1;
                if (master && all.length > 1) {
                    function syncLoop() {
                        if (master.ended) { window._syncRAF = null; return; }
                        var t = master.currentTime;
                        var slaves = all.filter(function(x){ return x !== master; });
                        slaves.forEach(function(s) {
                            if (s === v2 && canvasOffset > 0) {
                                if (Math.abs(s.currentTime - (t + canvasOffset)) > 0.04) s.currentTime = t + canvasOffset;
                            } else {
                                if (Math.abs(s.currentTime - t) > 0.04) s.currentTime = t;
                            }
                        });
                        window._syncRAF = requestAnimationFrame(syncLoop);
                    }
                    window._syncRAF = requestAnimationFrame(syncLoop);
                }

                // Simple timer: video duration + 0.3s -> fire done
                var dur = (master && master.duration > 0) ? master.duration : 10;
                console.log('PlayAll: video duration=' + dur.toFixed(2) + 's, done in ' + (dur + 0.3).toFixed(2) + 's');
                setTimeout(function() {
                    all.forEach(function(v){ v.pause(); });
                    // Update status text and show rendered video directly
                    console.log('PlayAll: timer fired, updating DOM directly');
                    var statusEl = document.querySelector('#status_text textarea, #status_text input');
                    if (statusEl) { statusEl.value = 'Done'; statusEl.dispatchEvent(new Event('input', {bubbles:true})); }
                    var demoVid = document.querySelector('#demo_output_video');
                    if (demoVid) { demoVid.style.display = 'block'; }
                    console.log('PlayAll: done, showed demo_output_video:', demoVid);
                }, (dur + 0.3) * 1000);
            }, 300);
        };

        window._showEditor = function() {
            var container = document.getElementById('threejs-container');
            if (container) container.style.display = '';
            var canvasVid = document.getElementById('canvas_video_el');
            if (canvasVid) { canvasVid.pause(); canvasVid.style.display = 'none'; }
        };

        function buildScene(data) {
            var old = scene.getObjectByName('depthMesh');
            if (old) { scene.remove(old); old.geometry.dispose(); old.material.dispose(); }

            sceneParams = data;
            var W = data.depth_w, H = data.depth_h;
            var fx = data.fx, fy = data.fy, cx = data.cx, cy = data.cy;

            var depthBytes = Uint8Array.from(atob(data.depth), function(c) { return c.charCodeAt(0); });
            var depthF32 = new Float32Array(depthBytes.buffer);

            var GRID = 192;
            var stepX = (W - 1) / (GRID - 1);
            var stepY = (H - 1) / (GRID - 1);

            var positions = new Float32Array(GRID * GRID * 3);
            var uvs = new Float32Array(GRID * GRID * 2);
            var sumX = 0, sumY = 0, sumZ = 0, cnt = 0;

            for (var gy = 0; gy < GRID; gy++) {
                for (var gx = 0; gx < GRID; gx++) {
                    var px = gx * stepX;
                    var py = gy * stepY;
                    var ix = Math.min(Math.floor(px), W - 2);
                    var iy = Math.min(Math.floor(py), H - 2);
                    var fx_ = px - ix, fy_ = py - iy;
                    var d00 = depthF32[iy * W + ix];
                    var d10 = depthF32[iy * W + ix + 1];
                    var d01 = depthF32[(iy + 1) * W + ix];
                    var d11 = depthF32[(iy + 1) * W + ix + 1];
                    var d = d00*(1-fx_)*(1-fy_) + d10*fx_*(1-fy_) + d01*(1-fx_)*fy_ + d11*fx_*fy_;

                    var x3d = (px - cx) / fx * d;
                    var y3d = -(py - cy) / fy * d;
                    var z3d = -d;

                    var idx = gy * GRID + gx;
                    positions[idx*3] = x3d;
                    positions[idx*3+1] = y3d;
                    positions[idx*3+2] = z3d;
                    uvs[idx*2] = gx / (GRID - 1);
                    uvs[idx*2+1] = 1.0 - gy / (GRID - 1);

                    sumX += x3d; sumY += y3d; sumZ += z3d; cnt++;
                }
            }

            meshCenter.set(sumX/cnt, sumY/cnt, sumZ/cnt);

            var indices = [];
            var depthThresh = 0.15;
            function zAt(idx) { return -positions[idx*3+2]; }
            for (var gy2 = 0; gy2 < GRID - 1; gy2++) {
                for (var gx2 = 0; gx2 < GRID - 1; gx2++) {
                    var a = gy2 * GRID + gx2;
                    var b = a + 1;
                    var c = a + GRID;
                    var dv = c + 1;
                    var da = zAt(a), db = zAt(b), dc = zAt(c), dd = zAt(dv);
                    var mn = Math.min(da, db, dc, dd);
                    if (mn < 1e-6) continue;
                    var mx = Math.max(da, db, dc, dd);
                    if ((mx - mn) / mn > depthThresh) continue;
                    indices.push(a, b, c);
                    indices.push(b, dv, c);
                }
            }

            var geo = new THREE.BufferGeometry();
            geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
            geo.setIndex(indices);
            geo.computeVertexNormals();

            var img = new Image();
            img.onload = function() {
                var tex = new THREE.Texture(img);
                tex.needsUpdate = true;
                tex.encoding = THREE.sRGBEncoding;
                mesh.material.map = tex;
                mesh.material.needsUpdate = true;
            };
            img.src = 'data:image/png;base64,' + data.rgb;

            var mat = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide });
            var mesh = new THREE.Mesh(geo, mat);
            mesh.name = 'depthMesh';
            scene.add(mesh);

            var minX=1e9, maxX=-1e9, minY=1e9, maxY=-1e9;
            for (var pi = 0; pi < positions.length; pi += 3) {
                if (positions[pi] < minX) minX = positions[pi];
                if (positions[pi] > maxX) maxX = positions[pi];
                if (positions[pi+1] < minY) minY = positions[pi+1];
                if (positions[pi+1] > maxY) maxY = positions[pi+1];
            }
            var meshExtent = Math.max(maxX - minX, maxY - minY, 0.001);

            camera.position.set(0, 0, 0.01);
            camera.near = meshExtent * 0.001;
            camera.far = meshExtent * 50;
            camera.updateProjectionMatrix();
            orbit.target.copy(meshCenter);
            orbit.update();

            var lightPos = uvzToPos(0.5, 0.5, 1.0);
            lightGroup.position.copy(lightPos);
            lightGroup.visible = true;
            lightLine.visible = true;
            transformCtrl.attach(lightGroup);

            updateLightInfo();
            addGrid(meshExtent, meshCenter);
            statusEl.textContent = 'Scene loaded: ' + GRID + 'x' + GRID + ' mesh';
        }

        scene.add(new THREE.AmbientLight(0xffffff, 1.0));

        var gridHelper = null;
        function addGrid(extent, center) {
            if (gridHelper) scene.remove(gridHelper);
            var gridSize = extent * 4;
            var divisions = 40;
            gridHelper = new THREE.GridHelper(gridSize, divisions, 0xbbbbcc, 0xd0d0dd);
            gridHelper.position.set(center.x, center.y - extent * 0.6, center.z);
            scene.add(gridHelper);
        }

        var LIGHT_SCREEN_PX = 12;
        function animate() {
            requestAnimationFrame(animate);
            orbit.update();
            if (lightGroup.visible) {
                var dist = camera.position.distanceTo(lightGroup.position);
                var vFOV = camera.fov * Math.PI / 180;
                var worldPerPx = 2 * dist * Math.tan(vFOV / 2) / renderer.domElement.clientHeight;
                var s = worldPerPx * LIGHT_SCREEN_PX;
                lightGroup.scale.setScalar(s);
            }
            renderer.render(scene, camera);
        }
        animate();

        var ro = new ResizeObserver(function() {
            var w = container.clientWidth, h = container.clientHeight;
            if (w > 0 && h > 0) {
                renderer.setSize(w, h);
                camera.aspect = w / h;
                camera.updateProjectionMatrix();
            }
        });
        ro.observe(container);

        // --- Trajectory replay for demo mode ---
        window._setLightUVZ = function(u, v, z) {
            if (!sceneParams) return;
            var pos = uvzToPos(u, v, z);
            lightGroup.position.copy(pos);
            updateLightInfo();
        };

        window._replayTrajectory = function(trajectory, fps) {
            var idx = 0;
            if (window._replayInterval) clearInterval(window._replayInterval);
            window._replayInterval = setInterval(function() {
                if (idx >= trajectory.length) {
                    clearInterval(window._replayInterval);
                    window._replayInterval = null;
                    return;
                }
                var t = trajectory[idx];
                window._setLightUVZ(t[0], t[1], t[2]);
                idx++;
            }, 1000.0 / fps);
        };

        window._stopReplay = function() {
            if (window._replayInterval) { clearInterval(window._replayInterval); window._replayInterval = null; }
        };

        window._buildRelightScene = buildScene;
    }
}
"""


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config-path",
        default="./configs/prompts/relight_perframe_ref_step19000_oldest_eval.yaml",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7864)
    p.add_argument(
        "--acceleration",
        default="xformers",
        choices=["none", "xformers"],
    )
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--share", action="store_true")
    p.add_argument("--demo-fps", type=float, default=18.18)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui(runner: RelightPerFrameRefLive, max_frames: int, demo_fps: float):

    with gr.Blocks(title="LiveLight Demo", css="""
        .main-row > .gr-column { flex: 1 1 0% !important; min-width: 0 !important; }
        video { border: none !important; outline: none !important; }
        .gr-video, .gr-video > div, .gr-video video,
        div[data-testid="video"] > div,
        div[data-testid="video"] {
            border: none !important; outline: none !important;
            box-shadow: none !important; transition: none !important;
        }
        #output_video .download, #output_video button[aria-label*="download"],
        #output_video button[aria-label*="Download"],
        #output_video .icon-buttons { display: none !important; }
        .dummy_never_match {
        }
    """) as demo:
        gr.Markdown("# 💡 LiveLight Demo")

        # -- state --
        video_frames_state = gr.State([])
        video_fps_state = gr.State(demo_fps)
        depth_map_state = gr.State(None)
        ref_latents_state = gr.State(None)
        clip_embeds_state = gr.State(None)
        frame_pils_state = gr.State([])
        output_path_state = gr.State(None)

        # ---- TOP: 3 columns (Input Video | Light Editor | Output Video) ----
        with gr.Row(elem_classes="main-row"):
            with gr.Column(scale=1):
                gr.Markdown("### Input Video")
                input_video = gr.Video(
                    label="Input",
                    interactive=False,
                    height=480,
                    elem_id="input_video",
                )
                source_frame = gr.Image(
                    label="Source Frame",
                    interactive=False,
                    height=480,
                    visible=False,
                )

            with gr.Column(scale=1):
                gr.Markdown("### Light Editor")
                threejs_block = gr.HTML(
                    value=THREEJS_CONTAINER,
                    elem_id="threejs_block",
                )
                scene_data = gr.Textbox(value="", visible=False, elem_id="scene_data")
                gr.HTML('<video id="canvas_video_el" style="display:none;width:100%;height:480px;border-radius:8px;background:#111;object-fit:contain;" playsinline></video>')

            with gr.Column(scale=1):
                gr.Markdown("### Streaming Video")
                streaming_preview = gr.Image(
                    label="Streaming Preview",
                    interactive=False,
                    height=480,
                    visible=False,
                )
                output_video = gr.Video(
                    label="Output",
                    interactive=False,
                    height=480,
                    elem_id="output_video",
                )

        # ---- BOTTOM: Upload, controls ----
        with gr.Row():
            with gr.Column(scale=1):
                video_upload = gr.Video(label="Upload Video", height=150)
                prep_status = gr.Textbox(label="Prep", value="—", interactive=False)

            with gr.Column(scale=2):
                with gr.Row():
                    intensity = gr.Slider(0.0, 2.5, value=1.0, step=0.01, label="Intensity")
                    light_color = gr.ColorPicker(label="Light Color", value="#ffffff")
                with gr.Row():
                    presets = [
                        ("\u2600\ufe0f Warm", "#ffe4c4"),
                        ("\u2744\ufe0f Cool", "#e0e8ff"),
                        ("\U0001f305 Sunset", "#ff8c42"),
                        ("\U0001f56f\ufe0f Candle", "#ff9329"),
                        ("\U0001f499 Blue", "#4a7dff"),
                        ("\U0001f49c Neon", "#ff44cc"),
                    ]
                    preset_btns = []
                    for plabel, phex in presets:
                        preset_btns.append((gr.Button(plabel, size="sm", min_width=60), phex))
                with gr.Row():
                    record_btn = gr.Button("Record", variant="secondary", size="lg")
                    render_btn = gr.Button("Render", variant="primary", size="lg")
                    play_all_btn = gr.Button("Play All", variant="secondary", size="lg")
                    demo_btn = gr.Button("Demo Mode", visible=False, variant="secondary", size="lg")
                with gr.Row():
                    generate_btn = gr.Button("\U0001f680 Generate", visible=False, variant="primary", size="lg")
                status_text = gr.Textbox(label="Status", value="Ready", interactive=False, elem_id="status_text")

            with gr.Column(scale=1):
                demo_output_video = gr.Video(label="Rendered Video", height=250, visible=True, elem_id="demo_output_video")

        # -- callbacks --

        def _gpu_mb():
            return torch.cuda.memory_allocated() / 1024 / 1024

        def on_video_upload(video_path):
            if video_path is None:
                return [], demo_fps, "—", None, None, None, [], "", None

            print(f"[upload] enter, GPU={_gpu_mb():.0f}MB", flush=True)
            runner.reset()
            gc.collect()
            torch.cuda.empty_cache()

            frames, orig_fps = extract_frames(video_path, max_frames=max_frames)
            if not frames:
                return [], demo_fps, "No frames", None, None, None, [], "", None

            max_sec = 11
            max_by_time = int(orig_fps * max_sec)
            if len(frames) > max_by_time:
                frames = frames[:max_by_time]

            # Resample to demo_fps
            orig_frame_count = len(frames)
            target_frame_count = int(orig_frame_count * demo_fps / orig_fps)
            if target_frame_count < 1:
                target_frame_count = 1
            resampled = []
            for i in range(target_frame_count):
                src_idx = min(int(i * orig_fps / demo_fps), orig_frame_count - 1)
                resampled.append(frames[src_idx])
            frames = resampled

            num = len(frames)
            msg = "Done"
            print(f"[upload] {msg}", flush=True)

            first_pil = frames[0].convert("RGB").resize(
                (runner.width, runner.height), Image.BILINEAR
            )
            depth_map = runner._predict_depth_once(first_pil)
            depth_map = runner._resize_depth_map(depth_map)

            anchor_depth = read_anchor_depth(depth_map, runner.anchor_u, runner.anchor_v, runner.depth_patch_radius)

            frame_pils = [f.convert("RGB").resize((runner.width, runner.height), Image.BILINEAR)
                          for f in frames]

            encode_bs = 8
            all_rl, all_ce = [], []
            with torch.inference_mode():
                for bi in range(0, len(frame_pils), encode_bs):
                    batch = frame_pils[bi : bi + encode_bs]
                    rl, ce = runner._encode_frames_batch(batch)
                    all_rl.append(rl.cpu())
                    all_ce.append(ce.cpu())
                    del rl, ce
                    torch.cuda.empty_cache()
            ref_latents = torch.cat(all_rl, dim=0)
            clip_embeds = torch.cat(all_ce, dim=0)
            torch.cuda.empty_cache()

            scene_json = encode_scene_data(first_pil, depth_map, runner.intrinsics, anchor_depth)

            # Create input video from frame_pils (same resolution as output) for perfect sync
            input_vid_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            frames_to_video(frame_pils, demo_fps, input_vid_tmp.name)
            print(f"[upload] input video: {len(frame_pils)} frames @ {demo_fps}fps", flush=True)

            return (frames, demo_fps, msg, depth_map, ref_latents, clip_embeds,
                    frame_pils, scene_json, input_vid_tmp.name)

        def hex_to_rgb(hex_str):
            hex_str = (hex_str or "#ffffff").lstrip("#")
            if len(hex_str) != 6:
                hex_str = "ffffff"
            return tuple(int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4))

        def on_record():
            return "Recording... drag the light while video plays"

        def on_render(frames, fps, frame_pils, depth_map, ref_latents, clip_embeds,
                      inten, light_color_hex):
            if not frames or depth_map is None:
                return None, None, "Upload video first", None, gr.update(visible=False), gr.update()

            # Load trajectory
            if not os.path.exists(TRAJECTORY_PATH):
                return None, None, "Record a trajectory first", None, gr.update(visible=False), gr.update()
            with open(TRAJECTORY_PATH) as f:
                trajectory = _json.load(f)

            num_frames = len(frame_pils)
            if len(trajectory) < num_frames:
                last = trajectory[-1] if trajectory else [0.5, 0.5, 1.0]
                while len(trajectory) < num_frames:
                    trajectory.append(last)
            trajectory = [(t[0], t[1], t[2]) for t in trajectory[:num_frames]]

            r, g, b = hex_to_rgb(light_color_hex)

            gc.collect()
            torch.cuda.empty_cache()

            video_path = render_with_trajectory(
                runner, frame_pils, depth_map, ref_latents, clip_embeds,
                fps, trajectory, inten, r, g, b,
            )

            runner.reset()
            gc.collect()
            torch.cuda.empty_cache()

            # Create trimmed input video (remove last 12 frames) to match output
            trim = LIGHT_OFFSET_FRAMES
            trimmed_input_pils = frame_pils[:max(1, len(frame_pils) - trim)]
            input_trimmed_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            frames_to_video(trimmed_input_pils, fps, input_trimmed_tmp.name)
            print(f"[render] Input trimmed: {len(trimmed_input_pils)} frames, Output trimmed: first {trim} removed", flush=True)

            return video_path, input_trimmed_tmp.name, "Done", video_path, gr.update(visible=True), gr.update(value=video_path)

        # -- wire --

        video_upload.change(
            fn=on_video_upload,
            inputs=[video_upload],
            outputs=[video_frames_state, video_fps_state, prep_status,
                     depth_map_state, ref_latents_state, clip_embeds_state,
                     frame_pils_state, scene_data, input_video],
        ).then(
            fn=None,
            inputs=[scene_data],
            outputs=[],
            js="(val) => { try { if (val && window._buildRelightScene) window._buildRelightScene(JSON.parse(val)); } catch(e) { console.error('scene build error', e); } }",
        )

        record_btn.click(
            fn=on_record,
            inputs=[],
            outputs=[status_text],
        ).then(
            fn=None,
            inputs=[video_fps_state],
            outputs=[],
            js="(fps) => { window._startRecording(fps); }",
        )

        render_btn.click(
            fn=on_render,
            inputs=[video_frames_state, video_fps_state, frame_pils_state,
                    depth_map_state, ref_latents_state, clip_embeds_state,
                    intensity, light_color],
            outputs=[output_video, input_video, status_text, output_path_state, demo_btn, demo_output_video],
        )

        # --- Demo mode ---
        def on_enter_demo():
            return (
                gr.update(visible=False),  # record_btn
                gr.update(visible=False),  # render_btn
                gr.update(visible=False),  # play_all_btn
                gr.update(visible=False),  # demo_btn
                gr.update(visible=True),   # generate_btn
                "Click 🚀 Generate to start",
                gr.update(),  # demo_output_video (JS controls visibility)
            )


        demo_btn.click(
            fn=on_enter_demo,
            inputs=[],
            outputs=[record_btn, render_btn, play_all_btn, demo_btn, generate_btn, status_text, demo_output_video],
        ).then(
            fn=None,
            inputs=[],
            outputs=[],
            js="() => { window._showEditor(); var d=document.querySelector('#demo_output_video'); if(d) d.style.display='none'; var ov=document.querySelector('#output_video video'); if(ov) ov.style.visibility='hidden'; }",
        )

        generate_btn.click(
            fn=lambda: "Generating...",
            inputs=[],
            outputs=[status_text],
        ).then(
            fn=None,
            inputs=[],
            outputs=[],
            js="() => { if (window._playAll) window._playAll(); }",
        )


        play_all_btn.click(
            fn=None,
            inputs=[],
            outputs=[],
            js="() => { if (window._playAll) window._playAll(); }",
        )

        for btn, hex_val in preset_btns:
            btn.click(fn=lambda h=hex_val: h, outputs=[light_color])

        demo.load(fn=None, js=INIT_THREEJS_JS)

    return demo


def main():
    args = parse_args()

    print(f"[init] Loading model from {args.config_path} ...")
    runner = RelightPerFrameRefLive(args, device="cuda:0")
    print("[init] Model loaded.")

    _socket.create_connection = _orig_create_connection

    import gradio.networking as _gnet
    _gnet.url_ok = lambda *a, **kw: True

    import gradio_client.utils as _gcutils
    _orig_json_schema_to_python_type = _gcutils._json_schema_to_python_type
    def _safe_json_schema_to_python_type(schema, defs=None):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_json_schema_to_python_type(schema, defs)
    _gcutils._json_schema_to_python_type = _safe_json_schema_to_python_type

    demo = build_ui(runner, max_frames=args.max_frames, demo_fps=args.demo_fps)

    # Light + trajectory HTTP server in separate process
    from http.server import HTTPServer, BaseHTTPRequestHandler

    LIGHT_PORT = args.port + 1

    def _run_light_server(shared_arr, port):
        class LightHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                path = self.path.split("?")[0]

                if path == "/light":
                    data = _json.loads(body)
                    shared_arr[0] = float(data.get("u", 0.5))
                    shared_arr[1] = float(data.get("v", 0.5))
                    shared_arr[2] = float(data.get("z", 1.0))
                    self._ok('{"ok":true}')

                elif path == "/trajectory":
                    with open(TRAJECTORY_PATH, "w") as f:
                        f.write(body.decode())
                    traj = _json.loads(body)
                    print(f"[trajectory] Received {len(traj)} frames", flush=True)
                    self._ok('{"ok":true}')

                else:
                    self.send_error(404)

            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/light":
                    self._ok(_json.dumps({"u": shared_arr[0], "v": shared_arr[1], "z": shared_arr[2]}))
                elif path == "/trajectory":
                    try:
                        with open("/tmp/_lightlive_trajectory.json") as f:
                            self._ok(f.read())
                    except FileNotFoundError:
                        self._ok("[]")
                else:
                    self.send_error(404)

            def do_OPTIONS(self):
                self.send_response(200)
                self._cors()
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def _ok(self, body_str):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._cors()
                self.end_headers()
                self.wfile.write(body_str.encode())

            def _cors(self):
                self.send_header("Access-Control-Allow-Origin", "*")

            def log_message(self, format, *a):
                pass

        server = HTTPServer(("0.0.0.0", port), LightHandler)
        server.serve_forever()

    light_proc = _mp.Process(target=_run_light_server, args=(_light_array, LIGHT_PORT), daemon=True)
    light_proc.start()
    print(f"[init] Light+trajectory server on port {LIGHT_PORT}", flush=True)

    demo.queue()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_api=False,
    )


if __name__ == "__main__":
    main()
