
"""
TRÓPICO — Tropical SST Anomaly Animation

NOAA Coral Reef Watch v3.1
Monthly mean SST anomalies, 5 km resolution

Research Group on Tropical Climatology,
Hydrology, and Meteorology — UNAM

Output: Silent MP4 background video.
"""

from pathlib import Path
import subprocess

import numpy as np
import requests
import xarray as xr
import matplotlib.pyplot as plt
from PIL import Image


# ==========================================
# 1. Configuration
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "crw_monthly"
FRAMES = DATA / "frames"
OUTPUT = ROOT / "assets" / "climate"

DATA.mkdir(parents=True, exist_ok=True)
FRAMES.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

# Animation period
START_MONTH = "2025-10"
N_MONTHS = 12

# Geographic domain (longitude: 0–360)
LON_MIN, LON_MAX = 100, 300
LAT_MIN, LAT_MAX = -35, 35

# Use every third grid point (~0.15 degrees)
STRIDE = 3

# Map and video
WIDTH = 1600
HEIGHT = 560
FPS = 24
FRAMES_PER_MONTH = 24

# Fixed SST anomaly scale (degrees Celsius)
VMIN = -4
VMAX = 4

# NOAA STAR monthly files
BASE_URL = (
    "https://www.star.nesdis.noaa.gov/pub/"
    "socd/mecb/crw/data/5km/v3.1_op/"
    "nc/v1.0/monthly"
)

VIDEO = OUTPUT / "tropico_sst_background.mp4"
POSTER = OUTPUT / "tropico_sst_poster.webp"


# ==========================================
# 2. Download and plot monthly anomalies
# ==========================================

months = [
    str(
        np.datetime64(START_MONTH, "M")
        + np.timedelta64(i, "M")
    )
    for i in range(N_MONTHS)
]

images = []

for month in months:

    year = month[:4]
    yyyymm = month.replace("-", "")

    filename = f"ct5km_ssta-mean_v3.1_{yyyymm}.nc"
    nc_file = DATA / filename
    png_file = FRAMES / f"ssta_{month}.png"

    print(f"Processing {month}...", flush=True)

    # --------------------------------------
    # Download the original NOAA monthly file
    # --------------------------------------

    if not nc_file.exists():

        url = f"{BASE_URL}/{year}/{filename}"
        temporary = DATA / (filename + ".part")

        temporary.unlink(missing_ok=True)

        print(f"  Downloading {filename}", flush=True)

        with requests.get(
            url,
            stream=True,
            timeout=(20, 90)
        ) as response:

            response.raise_for_status()

            expected = response.headers.get(
                "Content-Length"
            )

            downloaded = 0

            with temporary.open("wb") as file:

                for chunk in response.iter_content(
                    chunk_size=128 * 1024
                ):

                    if chunk:
                        file.write(chunk)
                        downloaded += len(chunk)

        if expected and downloaded != int(expected):
            raise RuntimeError(
                f"Incomplete download for {month}"
            )

        # Check the NetCDF before accepting it
        with xr.open_dataset(temporary) as ds:
            if not ds.data_vars:
                raise ValueError("Empty NetCDF file")

        temporary.replace(nc_file)

    # --------------------------------------
    # Read the SST anomaly and plot
    # --------------------------------------

    if not png_file.exists():

        with xr.open_dataset(nc_file) as ds:

            # Locate the SST anomaly variable
            names = [
                name for name in ds.data_vars
                if (
                    "ssta" in name.lower()
                    or "anomaly" in name.lower()
                )
            ]

            if not names:
                raise ValueError(
                    f"SST anomaly variable not found. "
                    f"Available: {list(ds.data_vars)}"
                )

            field = ds[names[0]].squeeze(drop=True)

            lat_name = next(
                name for name in field.dims
                if "lat" in name.lower()
            )

            lon_name = next(
                name for name in field.dims
                if "lon" in name.lower()
            )

            lat = ds[lat_name].values
            lon = np.mod(ds[lon_name].values, 360)

            # Select the tropical domain
            iy = np.where(
                (lat >= LAT_MIN) & (lat <= LAT_MAX)
            )[0]

            ix = np.where(
                (lon >= LON_MIN) & (lon <= LON_MAX)
            )[0]

            # Sort coordinates and reduce resolution
            iy = iy[np.argsort(lat[iy])][::STRIDE]
            ix = ix[np.argsort(lon[ix])][::STRIDE]

            values = field.isel({
                lat_name: iy,
                lon_name: ix
            }).transpose(
                lat_name, lon_name
            ).values.astype("float32")

        values[
            ~np.isfinite(values)
            | (np.abs(values) > 15)
        ] = np.nan

        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#142c3b")

        fig = plt.figure(
            figsize=(WIDTH / 100, HEIGHT / 100),
            dpi=100,
            facecolor="#142c3b"
        )

        ax = fig.add_axes([0, 0, 1, 1])

        ax.imshow(
            values,
            origin="lower",
            extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
            cmap=cmap,
            vmin=VMIN,
            vmax=VMAX,
            interpolation="bilinear",
            aspect="auto"
        )

        ax.axis("off")

        fig.savefig(
            png_file,
            dpi=100,
            facecolor=fig.get_facecolor()
        )

        plt.close(fig)

    images.append(png_file)


# ==========================================
# 3. Create a static poster
# ==========================================

with Image.open(images[-1]) as img:

    img.convert("RGB").save(
        POSTER,
        "WEBP",
        quality=85
    )


# ==========================================
# 4. Create the MP4 video
# ==========================================

print("Creating MP4 video...", flush=True)

command = [
    "ffmpeg", "-y",
    "-hide_banner",
    "-loglevel", "error",
    "-f", "rawvideo",
    "-pixel_format", "rgb24",
    "-video_size", f"{WIDTH}x{HEIGHT}",
    "-framerate", str(FPS),
    "-i", "-",
    "-an",
    "-c:v", "libx264",
    "-preset", "medium",
    "-crf", "25",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    str(VIDEO)
]

process = subprocess.Popen(
    command,
    stdin=subprocess.PIPE
)

for i in range(len(images)):

    with Image.open(images[i]) as img:
        current = img.convert("RGB")

    with Image.open(images[(i + 1) % len(images)]) as img:
        following = img.convert("RGB")

    for j in range(FRAMES_PER_MONTH):

        phase = j / FRAMES_PER_MONTH
        alpha = (1 - np.cos(np.pi * phase)) / 2

        frame = Image.blend(
            current,
            following,
            float(alpha)
        )

        process.stdin.write(frame.tobytes())

    print(
        f"Encoded {i + 1}/{len(images)}",
        flush=True
    )

process.stdin.close()

if process.wait() != 0:
    raise RuntimeError("FFmpeg encoding failed")


# ==========================================
# 5. Finished
# ==========================================

print("\nTRÓPICO animation completed!")
print(f"Video: {VIDEO}")
print(f"Poster: {POSTER}")
print(f"Size: {VIDEO.stat().st_size / 1e6:.1f} MB")
