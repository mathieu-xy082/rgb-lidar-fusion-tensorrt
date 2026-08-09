#!/usr/bin/env python3
"""Render a KITTI onboarding sequence and generate a keyboard-navigable gallery.

This script intentionally orchestrates the existing small helpers and prints each
underlying command before running it so developers can follow the pipeline.

Examples:
  pdm run python docs/onboarding/scripts/render_kitti_sequence.py --count 3
  pdm run python docs/onboarding/scripts/render_kitti_sequence.py --ids 000000,000001 --serve

Gallery controls:
  ArrowRight / ArrowDown / Space / n : next frame
  ArrowLeft / ArrowUp / p           : previous frame
  1, 2, 3, 4                       : focus a panel
"""
from __future__ import annotations

import argparse
import html
import http.server
import json
import shlex
import socketserver
import subprocess
import sys
from pathlib import Path

PANELS = [
    ("Camera", "image_{id}.ppm"),
    ("LiDAR overlay", "overlay_{id}.ppm"),
    ("Sparse depth overlay", "visualizations/sparse_depth_overlay.ppm"),
    ("Splatted depth overlay", "visualizations/splatted_depth_overlay.ppm"),
]


def parse_ids(ids: str | None, count: int) -> list[str]:
    if ids:
        return [item.strip() for item in ids.split(",") if item.strip()]
    return [f"{idx:06d}" for idx in range(count)]


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run(command: list[str]) -> None:
    print(f"COMMAND: {command_text(command)}", flush=True)
    subprocess.run(command, check=True)


def render_sample(sample_id: str, data_root: Path, output_root: Path, alpha: float) -> dict[str, object]:
    sample_out = output_root / f"kitti_{sample_id}"
    sample_out.mkdir(parents=True, exist_ok=True)

    image_png = data_root / "image_2" / f"{sample_id}.png"
    velodyne = data_root / "velodyne" / f"{sample_id}.bin"
    calib = data_root / "calib" / f"{sample_id}.txt"
    image_ppm = sample_out / f"image_{sample_id}.ppm"
    overlay_ppm = sample_out / f"overlay_{sample_id}.ppm"
    sparse_npz = sample_out / f"sparse_maps_{sample_id}.npz"
    splatted_npz = sample_out / f"splatted_maps_{sample_id}.npz"
    visual_dir = sample_out / "visualizations"

    for path in (image_png, velodyne, calib):
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run download_kitti_samples.py first")

    run([
        sys.executable,
        "docs/onboarding/scripts/png_to_ppm.py",
        str(image_png),
        str(image_ppm),
    ])
    run([
        sys.executable,
        "scripts/smoke_project_lidar.py",
        "--calib-file",
        str(calib),
        "--velodyne-file",
        str(velodyne),
        "--image-file",
        str(image_ppm),
        "--overlay-output",
        str(overlay_ppm),
        "--sparse-output",
        str(sparse_npz),
        "--splatted-output",
        str(splatted_npz),
        "--splat-radius-px",
        "2",
        "--splat-sigma-px",
        "1.0",
    ])
    run([
        sys.executable,
        "docs/onboarding/scripts/render_splatting_ppm.py",
        "--splatted-npz",
        str(splatted_npz),
        "--background-ppm",
        str(image_ppm),
        "--output-dir",
        str(visual_dir),
        "--alpha",
        f"{alpha:.3f}",
    ])

    return {
        "id": sample_id,
        "dir": str(sample_out),
        "panels": [
            {"title": title, "path": str((sample_out / pattern.format(id=sample_id)).relative_to(output_root))}
            for title, pattern in PANELS
        ],
    }


def write_gallery(output_root: Path, frames: list[dict[str, object]]) -> Path:
    gallery = output_root / "kitti_gallery.html"
    frames_json = json.dumps(frames, ensure_ascii=False)
    escaped_json = html.escape(frames_json, quote=False)
    gallery.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>KITTI RGB-LiDAR onboarding gallery</title>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #111; color: #eee; }}
    header {{ padding: 12px 16px; background: #1d1d1d; position: sticky; top: 0; z-index: 2; }}
    .hint {{ color: #bbb; font-size: 14px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; padding: 12px; }}
    .panel {{ background: #1a1a1a; border: 2px solid #333; border-radius: 8px; padding: 8px; }}
    .panel.focus {{ border-color: #ffcc33; }}
    .panel h2 {{ margin: 0 0 8px; font-size: 16px; }}
    canvas {{ width: 100%; height: auto; background: #000; image-rendering: auto; }}
    code {{ color: #9cdcfe; }}
  </style>
</head>
<body>
  <header>
    <div><strong>KITTI RGB-LiDAR onboarding gallery</strong> — frame <code id="frame-id"></code> (<span id="frame-index"></span>)</div>
    <div class="hint">←/→ previous/next frame · 1/2/3/4 focus panel · serve with <code>python -m http.server 8000</code> from <code>{html.escape(str(output_root))}</code></div>
  </header>
  <main class="grid" id="grid"></main>
  <script id="frames-json" type="application/json">{escaped_json}</script>
  <script>
    const frames = JSON.parse(document.getElementById('frames-json').textContent);
    let index = 0;
    let focusPanel = -1;

    function parsePPM(text) {{
      text = text.replace(/#[^\\n]*/g, ' ');
      const tokens = text.trim().split(/\\s+/);
      if (tokens[0] !== 'P3') throw new Error('Only P3 PPM is supported in this gallery');
      const width = Number(tokens[1]);
      const height = Number(tokens[2]);
      const max = Number(tokens[3]);
      if (max !== 255) throw new Error('Only maxval=255 is supported');
      const data = new Uint8ClampedArray(width * height * 4);
      let ti = 4;
      for (let i = 0; i < width * height; i++) {{
        data[i * 4] = Number(tokens[ti++]);
        data[i * 4 + 1] = Number(tokens[ti++]);
        data[i * 4 + 2] = Number(tokens[ti++]);
        data[i * 4 + 3] = 255;
      }}
      return new ImageData(data, width, height);
    }}

    async function drawPPM(canvas, path) {{
      const response = await fetch(path);
      if (!response.ok) throw new Error(`${{response.status}} ${{path}}`);
      const image = parsePPM(await response.text());
      canvas.width = image.width;
      canvas.height = image.height;
      canvas.getContext('2d').putImageData(image, 0, 0);
    }}

    async function showFrame() {{
      const frame = frames[index];
      document.getElementById('frame-id').textContent = frame.id;
      document.getElementById('frame-index').textContent = `${{index + 1}} / ${{frames.length}}`;
      const grid = document.getElementById('grid');
      grid.innerHTML = '';
      for (let i = 0; i < frame.panels.length; i++) {{
        const panel = frame.panels[i];
        const el = document.createElement('section');
        el.className = 'panel' + (i === focusPanel ? ' focus' : '');
        el.innerHTML = `<h2>${{i + 1}}. ${{panel.title}}</h2><canvas></canvas><div class="hint"><code>${{panel.path}}</code></div>`;
        grid.appendChild(el);
        try {{
          await drawPPM(el.querySelector('canvas'), panel.path);
        }} catch (error) {{
          el.querySelector('.hint').textContent = error.message;
        }}
      }}
    }}

    document.addEventListener('keydown', (event) => {{
      if (['ArrowRight', 'ArrowDown', ' ', 'n'].includes(event.key)) {{
        index = (index + 1) % frames.length;
        showFrame();
      }} else if (['ArrowLeft', 'ArrowUp', 'p'].includes(event.key)) {{
        index = (index + frames.length - 1) % frames.length;
        showFrame();
      }} else if (['1', '2', '3', '4'].includes(event.key)) {{
        focusPanel = Number(event.key) - 1;
        showFrame();
      }}
    }});

    showFrame();
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return gallery


def serve(output_root: Path, port: int) -> None:
    print(f"COMMAND: cd {output_root} && python -m http.server {port}")
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        print(f"SERVING http://127.0.0.1:{port}/kitti_gallery.html")
        import os

        os.chdir(output_root)
        httpd.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", help="Comma-separated KITTI frame IDs, e.g. 000000,000001")
    parser.add_argument("--count", type=int, default=1, help="Render IDs 000000..count-1 when --ids is absent")
    parser.add_argument("--data-root", type=Path, default=Path("data/kitti/training"))
    parser.add_argument("--output-root", type=Path, default=Path("results/onboarding"))
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--serve", action="store_true", help="Serve the generated gallery over localhost")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    sample_ids = parse_ids(args.ids, args.count)
    print(f"sample_ids={','.join(sample_ids)}", flush=True)
    print(f"data_root={args.data_root}", flush=True)
    print(f"output_root={args.output_root}", flush=True)

    frames = [render_sample(sample_id, args.data_root, args.output_root, args.alpha) for sample_id in sample_ids]
    gallery = write_gallery(args.output_root, frames)
    print(f"GALLERY: {gallery}")
    print(f"OPEN: http://127.0.0.1:{args.port}/{gallery.name}")
    print(f"COMMAND: cd {args.output_root} && python -m http.server {args.port}")

    if args.serve:
        serve(args.output_root, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
