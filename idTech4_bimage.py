"""
idTech 4 Binary Image (.bimage) Import Addon for Blender.

idTech 4 BFG (Doom 3 BFG) converts every source image it loads into a
compiled "generated/images/**.bimage" cache: a small header plus one or
more mip levels of GPU-ready pixel data (DXT1/DXT5 block-compressed, or
raw RGB565/RGBA8/etc), stored in several encoder-specific channel
layouts (plain RGBA, a YCoCg-in-DXT5 trick for diffuse maps, DXT5-alpha
normal maps, and a "mask value packed into green" trick for DXT1).

This script parses that binary format directly and decodes it into an
ordinary Blender image datablock. 
( See DOOM-3-BFG-VR/neo/renderer/BinaryImage.cpp, DXT/DXTDecoder.cpp,
RenderProgs_embedded.h for referemce) 

The PS4 build of D3BFG reuses two format codes for BC5/BC7.
this module doesn't implement a BC5/BC7 decoder of its own - those two
formats are decoded by round-tripping through a throwaway temp DDS file
and Blender's own bundled image loader instead (see decode_bc5/decode_bc7). 
"""

bl_info = {
    "name": "idTech4 Binary Image (.bimage)",
    "author": "Samson & Claude Sonnet",
    "version": (1, 0, 0),
    "blender": (4, 5, 0),
    "location": "File > Import > idTech4 Binary Image (.bimage)",
    "description": "Load idTech 4 / Doom 3 BFG .bimage caches as Blender images",
    "category": "Import-Export",
}

import bpy
import os
import struct
import tempfile
import numpy as np
from bpy.props import StringProperty, IntProperty, EnumProperty, BoolProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper

# ---------------------------------------------------------------------------
# Format constants (renderer/BinaryImageData.h, renderer/ImageOpts.h)
# ---------------------------------------------------------------------------

BIMAGE_MAGIC = (ord('B') << 0) | (ord('I') << 8) | (ord('M') << 16) | (10 << 24)

TT_DISABLED, TT_2D, TT_CUBIC, TT_2D_ARRAY = range(4)
TEXTURE_TYPE_NAMES = {TT_DISABLED: "DISABLED", TT_2D: "2D", TT_CUBIC: "CUBIC", TT_2D_ARRAY: "2D_ARRAY"}

(FMT_NONE, FMT_RGBA8, FMT_XRGB8, FMT_ALPHA, FMT_L8A8, FMT_LUM8, FMT_INT8,
 FMT_DXT1, FMT_DXT5, FMT_DEPTH, FMT_X16, FMT_Y16_X16, FMT_RGB565,
 FMT_ETC1_RGB8_OES, FMT_SHADOW_ARRAY) = range(15)

# The PS4 build of this engine uses these two numeric format codes
# for a different purposes: format 13 (PC: FMT_ETC1_RGB8_OES, a
# mobile-only format ) is actually BC5, and format 14 (PC: FMT_SHADOW_ARRAY)
# is actually BC7. As the PC-meaning values essentially never occur in practice, 
# these codes are unconditionally treated as BC5/BC7.

FMT_BC5 = FMT_ETC1_RGB8_OES
FMT_BC7 = FMT_SHADOW_ARRAY

FORMAT_NAMES = {
    FMT_NONE: "NONE", FMT_RGBA8: "RGBA8", FMT_XRGB8: "XRGB8", FMT_ALPHA: "ALPHA",
    FMT_L8A8: "L8A8", FMT_LUM8: "LUM8", FMT_INT8: "INT8", FMT_DXT1: "DXT1",
    FMT_DXT5: "DXT5", FMT_DEPTH: "DEPTH", FMT_X16: "X16", FMT_Y16_X16: "Y16_X16",
    FMT_RGB565: "RGB565", FMT_ETC1_RGB8_OES: "BC5 (PS4)", FMT_SHADOW_ARRAY: "BC7 (PS4)",
}

UNSUPPORTED_FORMATS = {FMT_DEPTH, FMT_X16, FMT_Y16_X16}

CFM_DEFAULT, CFM_NORMAL_DXT5, CFM_YCOCG_DXT5, CFM_GREEN_ALPHA, CFM_YCOCG_RGBA8 = range(5)

COLOR_FORMAT_NAMES = {
    CFM_DEFAULT: "DEFAULT", CFM_NORMAL_DXT5: "NORMAL_DXT5", CFM_YCOCG_DXT5: "YCOCG_DXT5",
    CFM_GREEN_ALPHA: "GREEN_ALPHA", CFM_YCOCG_RGBA8: "YCOCG_RGBA8",
}

CUBE_FACE_NAMES = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]


class BimageError(Exception):
    pass


class BimageLevel:
    __slots__ = ("level", "dest_z", "width", "height", "data")

    def __init__(self, level, dest_z, width, height, data):
        self.level = level
        self.dest_z = dest_z
        self.width = width
        self.height = height
        self.data = data


class BimageFile:
    __slots__ = ("texture_type", "format", "color_format", "width", "height",
                 "num_levels", "source_file_time", "levels")


# ---------------------------------------------------------------------------
# Header / level parsing (renderer/BinaryImage.cpp: LoadFromGeneratedFile)
# ---------------------------------------------------------------------------

def parse_bimage(filepath):
    """Parse a .bimage file's header and every stored mip/face level.

    The whole header (and every per-level header) is stored big-endian
    regardless of host platform; each level's pixel payload follows
    immediately as raw bytes.
    """
    with open(filepath, 'rb') as f:
        data = f.read()

    if len(data) < 36:
        raise BimageError(f"File too small to be a .bimage: {filepath}")

    source_file_time, header_magic, texture_type, fmt, color_format, width, height, num_levels = \
        struct.unpack_from('>q7i', data, 0)

    if header_magic != BIMAGE_MAGIC:
        raise BimageError(
            f"Not a .bimage file (bad magic 0x{header_magic & 0xFFFFFFFF:08X}, "
            f"expected 0x{BIMAGE_MAGIC:08X}): {filepath}")

    bf = BimageFile()
    bf.texture_type = texture_type
    bf.format = fmt
    bf.color_format = color_format
    bf.width = width
    bf.height = height
    bf.num_levels = num_levels
    bf.source_file_time = source_file_time
    bf.levels = []

    offset = 36
    num_images = num_levels * (6 if texture_type == TT_CUBIC else 1)
    for _ in range(num_images):
        if offset + 20 > len(data):
            raise BimageError(f"Truncated .bimage (level header past EOF): {filepath}")
        level, dest_z, w, h, data_size = struct.unpack_from('>iiiii', data, offset)
        offset += 20
        if offset + data_size > len(data):
            raise BimageError(f"Truncated .bimage (level data past EOF): {filepath}")
        bf.levels.append(BimageLevel(level, dest_z, w, h, data[offset:offset + data_size]))
        offset += data_size

    return bf


def get_level(bf, mip_level, dest_z=0):
    for lvl in bf.levels:
        if lvl.level == mip_level and lvl.dest_z == dest_z:
            return lvl
    raise BimageError(f"mip level {mip_level} (face {dest_z}) not present in file "
                       f"(has {bf.num_levels} level(s))")


# ---------------------------------------------------------------------------
# DXT1 / DXT5 block decode 
# ---------------------------------------------------------------------------

def _color565_to_rgb(c):
    c = c.astype(np.uint32)
    r5 = (c >> 11) & 0x1F
    g6 = (c >> 5) & 0x3F
    b5 = c & 0x1F
    r = ((r5 << 3) | (r5 >> 2)).astype(np.uint8)
    g = ((g6 << 2) | (g6 >> 4)).astype(np.uint8)
    b = ((b5 << 3) | (b5 >> 2)).astype(np.uint8)
    return r, g, b


def decode_dxt1(data, width, height):
    padded_w = (width + 3) & ~3
    padded_h = (height + 3) & ~3
    nbx, nby = padded_w // 4, padded_h // 4
    nblocks = nbx * nby

    raw = np.frombuffer(data, dtype=np.uint8, count=nblocks * 8).reshape(nblocks, 8)
    c0 = raw[:, 0].astype(np.uint32) | (raw[:, 1].astype(np.uint32) << 8)
    c1 = raw[:, 2].astype(np.uint32) | (raw[:, 3].astype(np.uint32) << 8)
    idx = (raw[:, 4].astype(np.uint32) | (raw[:, 5].astype(np.uint32) << 8) |
           (raw[:, 6].astype(np.uint32) << 16) | (raw[:, 7].astype(np.uint32) << 24))

    r0, g0, b0 = _color565_to_rgb(c0)
    r1, g1, b1 = _color565_to_rgb(c1)
    four_color = c0 > c1

    r0i, g0i, b0i = r0.astype(np.int32), g0.astype(np.int32), b0.astype(np.int32)
    r1i, g1i, b1i = r1.astype(np.int32), g1.astype(np.int32), b1.astype(np.int32)

    r2 = np.where(four_color, (2 * r0i + r1i) // 3, (r0i + r1i) // 2).astype(np.uint8)
    g2 = np.where(four_color, (2 * g0i + g1i) // 3, (g0i + g1i) // 2).astype(np.uint8)
    b2 = np.where(four_color, (2 * b0i + b1i) // 3, (b0i + b1i) // 2).astype(np.uint8)
    r3 = np.where(four_color, (r0i + 2 * r1i) // 3, 0).astype(np.uint8)
    g3 = np.where(four_color, (g0i + 2 * g1i) // 3, 0).astype(np.uint8)
    b3 = np.where(four_color, (b0i + 2 * b1i) // 3, 0).astype(np.uint8)
    a3 = np.where(four_color, 255, 0).astype(np.uint8)
    a_full = np.full(nblocks, 255, dtype=np.uint8)

    colors_r = np.stack([r0, r1, r2, r3], axis=1)
    colors_g = np.stack([g0, g1, g2, g3], axis=1)
    colors_b = np.stack([b0, b1, b2, b3], axis=1)
    colors_a = np.stack([a_full, a_full, a_full, a3], axis=1)

    out = np.empty((nby, nbx, 4, 4, 4), dtype=np.uint8)
    idx2d = idx.reshape(nby, nbx)
    block_range = np.arange(nblocks).reshape(nby, nbx)

    for t in range(16):
        row, col = divmod(t, 4)
        sel = (idx2d >> (2 * t)) & 3
        out[:, :, row, col, 0] = colors_r[block_range, sel]
        out[:, :, row, col, 1] = colors_g[block_range, sel]
        out[:, :, row, col, 2] = colors_b[block_range, sel]
        out[:, :, row, col, 3] = colors_a[block_range, sel]

    img = out.transpose(0, 2, 1, 3, 4).reshape(nby * 4, nbx * 4, 4)
    return img[:height, :width, :]


def decode_dxt5(data, width, height):
    padded_w = (width + 3) & ~3
    padded_h = (height + 3) & ~3
    nbx, nby = padded_w // 4, padded_h // 4
    nblocks = nbx * nby

    raw = np.frombuffer(data, dtype=np.uint8, count=nblocks * 16).reshape(nblocks, 16)
    a0 = raw[:, 0].astype(np.int32)
    a1 = raw[:, 1].astype(np.int32)
    aidx_lo = (raw[:, 2].astype(np.uint32) | (raw[:, 3].astype(np.uint32) << 8) | (raw[:, 4].astype(np.uint32) << 16))
    aidx_hi = (raw[:, 5].astype(np.uint32) | (raw[:, 6].astype(np.uint32) << 8) | (raw[:, 7].astype(np.uint32) << 16))
    c0 = raw[:, 8].astype(np.uint32) | (raw[:, 9].astype(np.uint32) << 8)
    c1 = raw[:, 10].astype(np.uint32) | (raw[:, 11].astype(np.uint32) << 8)
    cidx = (raw[:, 12].astype(np.uint32) | (raw[:, 13].astype(np.uint32) << 8) |
            (raw[:, 14].astype(np.uint32) << 16) | (raw[:, 15].astype(np.uint32) << 24))

    a_gt = a0 > a1
    alphas = np.zeros((nblocks, 8), dtype=np.int32)
    alphas[:, 0] = a0
    alphas[:, 1] = a1
    alphas[:, 2] = np.where(a_gt, (6 * a0 + 1 * a1) // 7, (4 * a0 + 1 * a1) // 5)
    alphas[:, 3] = np.where(a_gt, (5 * a0 + 2 * a1) // 7, (3 * a0 + 2 * a1) // 5)
    alphas[:, 4] = np.where(a_gt, (4 * a0 + 3 * a1) // 7, (2 * a0 + 3 * a1) // 5)
    alphas[:, 5] = np.where(a_gt, (3 * a0 + 4 * a1) // 7, (1 * a0 + 4 * a1) // 5)
    alphas[:, 6] = np.where(a_gt, (2 * a0 + 5 * a1) // 7, 0)
    alphas[:, 7] = np.where(a_gt, (1 * a0 + 6 * a1) // 7, 255)
    alphas = alphas.astype(np.uint8).reshape(nby, nbx, 8)

    r0, g0, b0 = _color565_to_rgb(c0)
    r1, g1, b1 = _color565_to_rgb(c1)
    r0i, g0i, b0i = r0.astype(np.int32), g0.astype(np.int32), b0.astype(np.int32)
    r1i, g1i, b1i = r1.astype(np.int32), g1.astype(np.int32), b1.astype(np.int32)
    # DXT5 color part has no punch-through mode: always the 4-way linear ramp.
    r2 = ((2 * r0i + r1i) // 3).astype(np.uint8)
    g2 = ((2 * g0i + g1i) // 3).astype(np.uint8)
    b2 = ((2 * b0i + b1i) // 3).astype(np.uint8)
    r3 = ((r0i + 2 * r1i) // 3).astype(np.uint8)
    g3 = ((g0i + 2 * g1i) // 3).astype(np.uint8)
    b3 = ((b0i + 2 * b1i) // 3).astype(np.uint8)

    colors_r = np.stack([r0, r1, r2, r3], axis=1)
    colors_g = np.stack([g0, g1, g2, g3], axis=1)
    colors_b = np.stack([b0, b1, b2, b3], axis=1)

    out = np.empty((nby, nbx, 4, 4, 4), dtype=np.uint8)
    cidx2d = cidx.reshape(nby, nbx)
    aidx_lo2d = aidx_lo.reshape(nby, nbx)
    aidx_hi2d = aidx_hi.reshape(nby, nbx)
    block_range = np.arange(nblocks).reshape(nby, nbx)
    row_idx = np.arange(nby)[:, None]
    col_idx = np.arange(nbx)[None, :]

    for t in range(16):
        row, col = divmod(t, 4)
        csel = (cidx2d >> (2 * t)) & 3
        out[:, :, row, col, 0] = colors_r[block_range, csel]
        out[:, :, row, col, 1] = colors_g[block_range, csel]
        out[:, :, row, col, 2] = colors_b[block_range, csel]
        asel = (aidx_lo2d >> (3 * t)) & 7 if t < 8 else (aidx_hi2d >> (3 * (t - 8))) & 7
        out[:, :, row, col, 3] = alphas[row_idx, col_idx, asel]

    img = out.transpose(0, 2, 1, 3, 4).reshape(nby * 4, nbx * 4, 4)
    return img[:height, :width, :]


def decode_rgb565(data, width, height):
    # Stored as manually packed big-endian uint16 (renderer/BinaryImage.cpp
    # writes the high byte first), NOT the native-endian block colors used
    # inside DXT payloads.
    raw = np.frombuffer(data, dtype='>u2', count=width * height).reshape(height, width)
    raw = raw.astype(np.uint32)
    r5 = (raw >> 11) & 0x1F
    g6 = (raw >> 5) & 0x3F
    b5 = raw & 0x1F
    r = ((r5 << 3) | (r5 >> 2)).astype(np.uint8)
    g = ((g6 << 2) | (g6 >> 4)).astype(np.uint8)
    b = ((b5 << 3) | (b5 >> 2)).astype(np.uint8)
    a = np.full((height, width), 255, dtype=np.uint8)
    return np.stack([r, g, b, a], axis=-1)


_DDS_MAGIC = b'DDS '
_DXGI_BC5_UNORM = 83
_DXGI_BC7_UNORM = 98


def _make_minimal_dx10_dds(dxgi_format, width, height, block_bytes, block_size=16):
    """The smallest possible single-mip DX10 DDS wrapper around already-
    extracted BC5/BC7 block bytes - just enough for Blender's own bundled
    image loader (OpenImageIO) to recognise and decode them, since this
    module has no native BC5/BC7 decoder of its own. """
    flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000  # CAPS|HEIGHT|WIDTH|PIXELFORMAT|LINEARSIZE
    blocks_w = max(1, (width + 3) // 4)
    blocks_h = max(1, (height + 3) // 4)
    pitch = blocks_w * blocks_h * block_size
    header = struct.pack('<7I', 124, flags, height, width, pitch, 0, 1)
    header += b'\x00' * 44  # dwReserved1[11]
    header += struct.pack('<2I4s5I', 32, 0x4, b'DX10', 0, 0, 0, 0, 0)  # DDS_PIXELFORMAT (FourCC 'DX10')
    header += struct.pack('<5I', 0x1000, 0, 0, 0, 0)  # caps, caps2/3/4, reserved2
    dx10_header = struct.pack('<5I', dxgi_format, 3, 0, 1, 0)  # format, DIM_TEXTURE2D, miscFlag, arraySize, miscFlags2
    return _DDS_MAGIC + header + dx10_header + block_bytes


def _decode_via_blender_dds(dxgi_format, width, height, block_bytes):
    """Decode a block-compressed format via Blender's own bundled decoder
    by round-tripping through a throwaway temp DDS file (bpy.data.images.load
    needs a real file path, no in-memory loading available). Returns a
    top-to-bottom (H,W,4) uint8 array, matching the other decode_* functions
    - Blender's own Image.pixels are bottom-to-top, so this flips once to
    match; load_bimage_pixels flips again at the very end for the final
    Blender assignment, same as every other format."""
    dds_bytes = _make_minimal_dx10_dds(dxgi_format, width, height, block_bytes)
    fd, path = tempfile.mkstemp(suffix='.dds', prefix='idtech4_bimage_')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(dds_bytes)
        img = bpy.data.images.load(path, check_existing=False)
        try:
            flat = np.empty(width * height * 4, dtype=np.float32)
            img.pixels.foreach_get(flat)
        finally:
            bpy.data.images.remove(img)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    rgba = (np.clip(flat, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8).reshape(height, width, 4)
    return np.flipud(rgba)


def decode_bc5(data, width, height):
    return _decode_via_blender_dds(_DXGI_BC5_UNORM, width, height, data)


def decode_bc7(data, width, height):
    return _decode_via_blender_dds(_DXGI_BC7_UNORM, width, height, data)


def decode_uncompressed(data, width, height, fmt):
    n = width * height
    if fmt == FMT_RGBA8:
        return np.frombuffer(data, dtype=np.uint8, count=n * 4).reshape(height, width, 4).copy()
    if fmt == FMT_XRGB8:
        px = np.frombuffer(data, dtype=np.uint8, count=n * 4).reshape(height, width, 4).copy()
        px[:, :, 3] = 255
        return px
    if fmt == FMT_ALPHA:
        a = np.frombuffer(data, dtype=np.uint8, count=n).reshape(height, width)
        return np.stack([a, a, a, a], axis=-1)
    if fmt == FMT_L8A8:
        px = np.frombuffer(data, dtype=np.uint8, count=n * 2).reshape(height, width, 2)
        lum, alpha = px[:, :, 0], px[:, :, 1]
        return np.stack([lum, lum, lum, alpha], axis=-1)
    if fmt == FMT_LUM8:
        lum = np.frombuffer(data, dtype=np.uint8, count=n).reshape(height, width)
        a = np.full((height, width), 255, dtype=np.uint8)
        return np.stack([lum, lum, lum, a], axis=-1)
    if fmt == FMT_INT8:
        v = np.frombuffer(data, dtype=np.uint8, count=n).reshape(height, width)
        return np.stack([v, v, v, v], axis=-1)
    raise BimageError(f"Unsupported uncompressed format {FORMAT_NAMES.get(fmt, fmt)}")


def decode_level(level, fmt):
    w, h = level.width, level.height
    if fmt == FMT_DXT1:
        return decode_dxt1(level.data, w, h)
    if fmt == FMT_DXT5:
        return decode_dxt5(level.data, w, h)
    if fmt == FMT_RGB565:
        return decode_rgb565(level.data, w, h)
    if fmt == FMT_BC5:
        return decode_bc5(level.data, w, h)
    if fmt == FMT_BC7:
        return decode_bc7(level.data, w, h)
    if fmt in UNSUPPORTED_FORMATS:
        raise BimageError(f"Format {FORMAT_NAMES.get(fmt, fmt)} is a render-target/special "
                           f"format, not a storable texture format - can't decode it here.")
    return decode_uncompressed(level.data, w, h, fmt)


# ---------------------------------------------------------------------------
# Color-format post-processing (renderer/BinaryImage.cpp encode side +
# RenderProgs_embedded.h ConvertYCoCgToRGB / DXT5nm shader decode - these
# undo the channel-packing tricks baked into the stored bytes)
# ---------------------------------------------------------------------------

def apply_color_format(rgba_u8, color_format):
    """rgba_u8: (H,W,4) uint8 as decoded straight from the pixel format.
    Returns float32 (H,W,4) in [0,1]."""
    f = rgba_u8.astype(np.float32) / 255.0

    if color_format in (CFM_DEFAULT,):
        return f

    if color_format in (CFM_YCOCG_DXT5, CFM_YCOCG_RGBA8):
        r, g, b, a = f[..., 0], f[..., 1], f[..., 2], f[..., 3]
        z = (b * 31.875) + 1.0
        z = 1.0 / z
        co = r * z
        cg = g * z
        y = a
        out_r = co - cg + y
        out_g = cg - 0.50196078 * z + y
        out_b = -co - cg + 1.00392156 * z + y
        out = np.stack([out_r, out_g, out_b, np.ones_like(y)], axis=-1)
        return np.clip(out, 0.0, 1.0)

    if color_format == CFM_NORMAL_DXT5:
        nx = f[..., 3] * 2.0 - 1.0
        ny = f[..., 1] * 2.0 - 1.0
        nz = np.sqrt(np.clip(1.0 - nx * nx - ny * ny, 0.0, 1.0))
        out = np.stack([nx * 0.5 + 0.5, ny * 0.5 + 0.5, nz * 0.5 + 0.5, np.ones_like(nx)], axis=-1)
        return out

    if color_format == CFM_GREEN_ALPHA:
        g = f[..., 1]
        return np.stack([g, g, g, g], axis=-1)

    return f


def _decode_and_finish(level, fmt, color_format):
    """decode_level() plus the appropriate colorFormat post-processing -
    except for BC5/BC7, which already come back from decode_level() as
    complete, final RGBA (Blender's own DDS/OIIO round-trip did all the
    channel work) rather than raw DXT-packed data. The colorFormat byte
    idTech4 still stores alongside them describes the OLD DXT5-packing
    scheme (e.g. CFM_NORMAL_DXT5's "X in alpha, Y in green" trick) that
    doesn't apply once the format is genuinely native like BC5 - re-running
    that formula on already-finished data would corrupt it."""
    raw = decode_level(level, fmt)
    if fmt in (FMT_BC5, FMT_BC7):
        return raw.astype(np.float32) / 255.0
    return apply_color_format(raw, color_format)


# ---------------------------------------------------------------------------
# High-level: bimage -> Blender image
# ---------------------------------------------------------------------------

def _pixels_to_blender_flat(rgba_float):
    # idTech4 pixel buffers are top-to-bottom; Blender's Image.pixels are
    # bottom-to-top, so flip vertically before handing off.
    return np.ascontiguousarray(np.flipud(rgba_float)).ravel()


def load_bimage_pixels(filepath, mip_level=0, cube_face='0'):
    """Parse+decode a .bimage file. Returns (width, height, flat_float32_pixels, info_dict)."""
    bf = parse_bimage(filepath)

    if bf.format in UNSUPPORTED_FORMATS:
        raise BimageError(f"'{os.path.basename(filepath)}' uses format "
                           f"{FORMAT_NAMES.get(bf.format, bf.format)}, which idTech4 only ever "
                           f"uses for render targets, not saved textures - nothing to import.")

    if bf.texture_type == TT_CUBIC:
        if cube_face == 'STRIP':
            faces = [_decode_and_finish(get_level(bf, mip_level, z), bf.format, bf.color_format)
                     for z in range(6)]
            combined = np.concatenate(faces, axis=1)  # side by side, left to right
            width, height = combined.shape[1], combined.shape[0]
            flat = _pixels_to_blender_flat(combined)
        else:
            z = int(cube_face)
            level = get_level(bf, mip_level, z)
            px = _decode_and_finish(level, bf.format, bf.color_format)
            width, height = level.width, level.height
            flat = _pixels_to_blender_flat(px)
    elif bf.texture_type == TT_2D:
        level = get_level(bf, mip_level, 0)
        px = _decode_and_finish(level, bf.format, bf.color_format)
        width, height = level.width, level.height
        flat = _pixels_to_blender_flat(px)
    else:
        raise BimageError(f"Unsupported texture type "
                           f"{TEXTURE_TYPE_NAMES.get(bf.texture_type, bf.texture_type)}")

    info = dict(
        texture_type=TEXTURE_TYPE_NAMES.get(bf.texture_type, str(bf.texture_type)),
        format=FORMAT_NAMES.get(bf.format, str(bf.format)),
        color_format=COLOR_FORMAT_NAMES.get(bf.color_format, str(bf.color_format)),
        base_width=bf.width, base_height=bf.height, num_levels=bf.num_levels,
        color_format_raw=bf.color_format,
    )
    return width, height, flat, info


def load_bimage_as_image(filepath, mip_level=0, cube_face='0', force_reload=False, colorspace=None):
    """Load a .bimage file as a bpy.data.images entry, reusing an existing
    datablock loaded from the same file/mip/face unless force_reload is set."""
    abs_path = os.path.normpath(bpy.path.abspath(filepath))
    tag = f"{abs_path}|{mip_level}|{cube_face}"

    if not force_reload:
        for img in bpy.data.images:
            if img.get("bimage_source_tag") == tag:
                return img, None

    width, height, flat, info = load_bimage_pixels(filepath, mip_level, cube_face)

    name = os.path.basename(filepath)
    img = bpy.data.images.new(name, width, height, alpha=True, float_buffer=False)
    img.pixels.foreach_set(flat)
    img.pack()  # embed the decoded pixels so the datablock survives without the source file
    img["bimage_source_tag"] = tag
    img["bimage_source_path"] = abs_path

    if colorspace is None:
        colorspace = 'Non-Color' if info['color_format_raw'] in (CFM_NORMAL_DXT5, CFM_GREEN_ALPHA) else 'sRGB'
    try:
        img.colorspace_settings.name = colorspace
    except TypeError:
        pass

    return img, info


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

class ImportFileGuardMixin:
    """Validate what the file browser actually handed back, before any
    import work happens.

    ImportHelper does no validation of its own. Click Import with no file
    selected and `filepath` is the browser's current *directory*, while
    `files` (on the multi-file importers) holds a single placeholder entry
    whose `.name` is "". Nothing downstream is built to notice either, so
    each importer used to fail its own awkward way: a raw "[Errno 2] No
    such file or directory" straight out of open(), an "Imported 0
    file(s)" that still returned {'FINISHED'}, or - worst - the Base
    Directory / Materials Source gate popping up, because a directory
    derives no base directory any more than a real model path outside a
    game tree does.

    So every import operator calls guard_input_files() as the FIRST
    statement of its execute(), above the sources gate and above any modal
    timer, and refuses with a message naming the actual mistake.

    Extensions are deliberately NOT policed here - only "nothing
    selected", "that's a folder" and "no such file" are refused. Every
    format here legitimately arrives under more than one extension
    (.ase/.base, .lwo/.blwo, .md5mesh/.bmd5mesh), and a hand-typed name is
    the user's business.

    Duplicated into each addon file rather than shared through an import,
    exactly like MD5_ImportScaleRotMixin and for the same reason: these
    are standalone single-file addons that have to work installed alone.
    """

    # Used only to build the error text ("No .ase file selected"). Falls
    # back to ImportHelper's own filename_ext, which every importer here
    # already sets.
    guard_label = ""

    def resolve_input_files(self, multi=False):
        """Return (paths, error). A non-None *error* is a ready-to-report
        message meaning the operator must refuse; *paths* is then empty.

        *multi*: read the browser's multi-selection (`files` + `directory`)
        instead of `filepath`, dropping the empty-named placeholder entry
        Blender always leaves in `files` when nothing is selected.

        A partial selection is not fatal: unusable entries are dropped and
        described in self._guard_warnings, and only a completely empty
        result is an error.
        """
        self._guard_warnings = []
        label = self.guard_label or getattr(self, 'filename_ext', '') or "file"

        candidates = []
        if multi and getattr(self, 'files', None):
            directory = getattr(self, 'directory', '') or ''
            candidates = [os.path.join(directory, f.name)
                          for f in self.files if f.name]
        # Falling back to filepath is not just the single-file path: with
        # nothing selected the `files` loop above yields nothing at all,
        # and filepath still holds the directory - which is what produces
        # the "that's a folder" message below rather than a bare "nothing
        # selected".
        if not candidates and self.filepath:
            candidates = [self.filepath]

        if not candidates:
            return [], (f"No {label} file selected - pick one in the file "
                        f"browser, then click Import.")

        paths = []
        for path in candidates:
            if os.path.isdir(path):
                name = os.path.basename(os.path.normpath(path))
                self._guard_warnings.append(
                    f"\"{name}\" is a folder, not a {label} file - open it "
                    f"and select a file inside.")
            elif not os.path.isfile(path):
                self._guard_warnings.append(f"File not found: {path}")
            else:
                paths.append(path)

        if not paths:
            # One bad path is the ordinary case (nothing selected at all),
            # so say what was wrong with it rather than summarising a
            # count the user would then have to go find.
            if len(self._guard_warnings) == 1:
                return [], self._guard_warnings[0]
            return [], ("Nothing usable in the selection: "
                        + " ".join(self._guard_warnings))
        return paths, None

    def guard_input_files(self, multi=False):
        """resolve_input_files() plus the reporting. Returns
        (paths, status): a non-None *status* has already been reported and
        is the operator's own return value.

        Always {'CANCELLED'}, never {'FINISHED'} - a refused import must
        not push an undo step, and a scripted caller has no other way to
        tell that nothing was imported.
        """
        paths, err = self.resolve_input_files(multi=multi)
        if err:
            self.report({'ERROR'}, err)
            return [], {'CANCELLED'}
        for msg in self._guard_warnings:
            self.report({'WARNING'}, msg)
        return paths, None


class IMPORT_OT_idtech4_bimage(Operator, ImportHelper, ImportFileGuardMixin):
    """Load an idTech4 .bimage cache file as a Blender image"""
    bl_idname = "idtech4.import_bimage"
    bl_label = "Import idTech4 .bimage"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".bimage"
    filter_glob: StringProperty(default="*.bimage", options={'HIDDEN'})

    mip_level: IntProperty(
        name="Mip Level",
        description="0 = full resolution (base) mip",
        default=0, min=0,
    )
    cube_face: EnumProperty(
        name="Cube Face",
        description="Which face to load, for cubemap .bimage files (ignored for ordinary 2D images)",
        items=[
            ('0', f"Face 0 ({CUBE_FACE_NAMES[0]})", ""),
            ('1', f"Face 1 ({CUBE_FACE_NAMES[1]})", ""),
            ('2', f"Face 2 ({CUBE_FACE_NAMES[2]})", ""),
            ('3', f"Face 3 ({CUBE_FACE_NAMES[3]})", ""),
            ('4', f"Face 4 ({CUBE_FACE_NAMES[4]})", ""),
            ('5', f"Face 5 ({CUBE_FACE_NAMES[5]})", ""),
            ('STRIP', "All 6 (horizontal strip)", "Lay all six faces out side by side"),
        ],
        default='STRIP',
    )
    force_reload: BoolProperty(
        name="Force Reload",
        description="Reload even if this file/mip/face was already imported",
        default=False,
    )

    def execute(self, context):
        _paths, status = self.guard_input_files()
        if status:
            return status
        try:
            img, info = load_bimage_as_image(
                self.filepath, self.mip_level, self.cube_face, self.force_reload)
        except BimageError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load '{self.filepath}': {e}")
            return {'CANCELLED'}

        if info is not None:
            self.report(
                {'INFO'},
                f"Loaded {img.name}: {info['texture_type']} {info['format']}/"
                f"{info['color_format']} {info['base_width']}x{info['base_height']} "
                f"({info['num_levels']} mip(s))")
        else:
            self.report({'INFO'}, f"Reused already-loaded image {img.name}")
        return {'FINISHED'}


def menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_idtech4_bimage.bl_idname, text="idTech4 Binary Image (.bimage)")


classes = (
    IMPORT_OT_idtech4_bimage,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
