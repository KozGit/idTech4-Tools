# SPDX-License-Identifier: GPL-3.0-or-later
"""
idTech4 renderbump for Blender
==============================

A bit-for-bit re-implementation of the `renderbump` console command from the
Doom 3 / Quake 4 / Prey engine (idTech4), including the exact model-loading
path the engine takes to get there.  Given the same low- and high-poly models
the engine was given, this produces the same tangent-space normal map, stored
as an internal Blender image.

This is a REPLICATION, not an improvement.  Every quirk of the original is
kept on purpose: the lazy 100-step ray march through the triangle hash, the
`int -= 0.001` epsilon in CreateTriHash that only ever decrements the lower
bin (and the typo that applies the second epsilon to the wrong array), the
farthest-hit-wins trace, the dupVerts normal merge that loses contributions
on 3-way shared vertices, and the truncating (not rounding) float->byte
conversion.

WHICH ENGINE CODE THIS MIRRORS
------------------------------
  renderer/Model_lwo.cpp        lwGetPolyNormals / lwGetVertNormals /
                                lwGetPointPolygons -- the LWO smoothing pass
  renderer/Model_ase.cpp        ASE_Parse and every ASE_Key* callback under
                                it, including the two bugs in
                                ASE_KeyMESH_CVERTLIST
  framework/FileSystem.cpp      OSPathToRelativePath, which is what turns an
                                .ase *BITMAP into a material name
  renderer/Model.cpp            ConvertLWOToModelSurfaces and
                                ConvertASEToModelSurfaces, each in both the
                                normal path (the low poly, via CheckModel)
                                and the fastLoad path (the high poly, via
                                PartialInitFromFile)
  renderer/tr_trisurf.cpp       R_CleanupTriangles and everything under it:
                                CreateSilIndexes, RemoveDegenerateTriangles,
                                DuplicateMirroredVertexes, CreateDupVerts,
                                DeriveFacePlanes, DeriveTangents
  idlib/math/Simd_Generic.cpp   DeriveTriPlanes / DeriveTangents /
                                NormalizeTangents
  tools/compilers/renderbump/renderbump.cpp
                                CreateTriHash, TraceToMeshFace,
                                SampleHighMesh, RasterizeTriangle,
                                RenderBumpTriangles, OutlineNormalMap,
                                WriteRenderBump

THREE THINGS WORTH KNOWING BEFORE YOU USE IT
--------------------------------------------
1.  The low-poly model's normals and tangents are NOT taken from the file.
    The engine throws away any explicit normals on a surface whose material
    carries a renderBump command ("completely ignore any explict normals on
    surfaces with a renderbump command, which will guarantee the best
    contours and least vertexes") and regenerates them.  So does this.  The
    smooth-shading you see in Blender on the low poly has no effect at all.

2.  The high-poly model's normals ARE taken from the file, but where they
    come from depends on the format.  A .lwo carries no usable normals at
    all: the engine recomputes LightWave's own smoothing (each polygon's
    flat normal, plus every adjacent polygon that shares the point, is in
    the same smoothing group, and sits within the surface's SMAN angle),
    which is not the rule Blender's shade-auto-smooth uses and is not what
    Blender's quantised custom split normals hold, so this addon recomputes
    it too.  Set "High Smoothing Angle" to the source surface's SMAN value
    (LightWave's default, and what the shipped Doom 3 models use, is 90).
    An .ase is the other way around -- it stores finished per-corner normals
    in *MESH_NORMALS and the engine uses them verbatim, so the smoothing
    angle does not apply and is hidden.

3.  Both formats can be read straight off disk, and that is the accurate
    way to use this.  Going through Blender costs you something either way:
    the .lwo importer welds points that round together, and the .ase
    importer deliberately undoes the winding and V flips the engine applies
    at parse time.  Reading the file skips all of it.

ACCURACY
--------
Against the reference `gizmo1_local.tga` shipped with the test pair, from the
same two .lwo files at `-size 1024 1024 -aa 0` (not the `-size 128 128 -aa 2`
the shipped mapobjects.mtr asks for -- the reference was rebaked at the
larger size): 99.25% of texels are bit-identical, 99.996% are within
one 8-bit step, and the traced/untraced coverage mask matches on all but one
texel in a million.  The residue is sub-LSB floating point noise: the
original is a 32-bit x86 build whose tangent and plane builders normalise
with the SSE `rsqrtps` instruction, a hardware approximation accurate to
about 1.5e-4 whose exact result is a property of the CPU's internal table and
is not reproducible in software.  Selecting "Generic SIMD (RSqrt)" in the
addon switches to idMath::RSqrt instead, which is what a build with the SIMD
assembly disabled (e.g. dhewm3) uses -- that scores 95.9% on the same pair,
so "Exact" is the right default for a stock Doom 3 build.
"""

bl_info = {
    "name": "idTech4 RenderBump",
    "author": "replicated from the Doom 3 GPL source",
    "version": (1, 0, 0),
    "blender": (3, 2, 0),
    "location": "View3D > Sidebar > idTech4 Renderbump",
    "description": "Generate a normal map exactly the way idTech4's renderbump does",
    "category": "Object",
}

import math
import os
import re
import struct
import time

import numpy as np

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, IntProperty,
                       PointerProperty, StringProperty)
from bpy.types import Operator, Panel, PropertyGroup

F32 = np.float32

# Every expression below that the original evaluates in float is evaluated in
# float here too; anything the original evaluates on the x87 stack (a 32-bit
# MSVC build promotes float expressions to 80-bit and only rounds on store)
# is evaluated in float64 and rounded once, at the store.


# =============================================================================
#  Scalar / vector helpers -- idlib/math
# =============================================================================

RSQRT_EXACT = 'EXACT'
RSQRT_QUAKE = 'QUAKE'


def rsqrt(x, mode=RSQRT_EXACT):
    """The reciprocal square root the SIMD tangent/plane builders use.

    EXACT models an SSE build (rsqrtps, ~1.5e-4 relative, unreproducible in
    software, so the exact value is the closest we can get).  QUAKE is
    idMath::RSqrt verbatim -- the 0x5f3759df trick plus one Newton step --
    which is what idSIMD_Generic uses when the SIMD assembly is compiled out.
    """
    x = np.asarray(x, F32)
    if mode == RSQRT_QUAKE:
        y = (x * F32(0.5)).astype(F32)
        i = x.view(np.int32)
        i = np.int32(0x5f3759df) - (i >> 1)
        r = i.view(F32)
        return (r * (F32(1.5) - (r * r).astype(F32) * y)).astype(F32)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = (1.0 / np.sqrt(x.astype(np.float64))).astype(F32)
    # RSqrt(0) blows up to a huge finite number and scales the zero vector
    # back to zero; the SSE path's FIX_DEGENERATE_TANGENT does the same via
    # SIMD_SP_tiny.  Either way the result is a zero vector.
    return np.where(np.isfinite(r), r, F32(0.0)).astype(F32)


def normalize_exact(v):
    """idVec3::Normalize -- idMath::InvSqrt is table + two Newton steps in
    double, i.e. exact to within a float ULP."""
    v = np.asarray(v, F32)
    sq = (v * v).astype(F32).sum(-1, dtype=np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv = (1.0 / np.sqrt(sq.astype(np.float64))).astype(F32)
    inv = np.where(np.isfinite(inv), inv, F32(0))
    return (v * inv[..., None]).astype(F32)


def fix_degenerate_normal(n):
    """idVec3::FixDegenerateNormal -- snap near-axial normals onto the axis.

    "LWO models aren't all that pretty when it comes down to the floating
    point values they store."
    """
    n = np.array(n, F32, copy=True)
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    m = (x == 0) & (y == 0)
    z[m & (z > 0)] = 1.0
    z[m & (z <= 0)] = -1.0
    m = (x == 0) & (y != 0) & (z == 0)
    y[m & (y > 0)] = 1.0
    y[m & (y <= 0)] = -1.0
    m = (x != 0) & (y == 0) & (z == 0)
    x[m & (x > 0)] = 1.0
    x[m & (x <= 0)] = -1.0
    axial = ((n[..., 0] == 0) & (n[..., 1] == 0)) | \
            ((n[..., 0] == 0) & (n[..., 2] == 0)) | \
            ((n[..., 1] == 0) & (n[..., 2] == 0))
    ax = np.abs(n[..., 0]) == 1.0
    ay = (~ax) & (np.abs(n[..., 1]) == 1.0)
    az = (~ax) & (~ay) & (np.abs(n[..., 2]) == 1.0)
    sel = (~axial) & ax
    n[sel, 1] = 0.0
    n[sel, 2] = 0.0
    sel = (~axial) & ay
    n[sel, 0] = 0.0
    n[sel, 2] = 0.0
    sel = (~axial) & az
    n[sel, 0] = 0.0
    n[sel, 1] = 0.0
    return n


def c_trunc(x):
    """C's float -> int conversion: truncate toward zero."""
    return np.trunc(x).astype(np.int64)


class HashIndex(object):
    """idHashIndex.  Chains are LIFO -- First() returns the most recently
    added index, which is what decides ties in CreateSilRemap and FindVector.
    """

    def __init__(self, hash_size):
        self.mask = hash_size - 1
        self.hash = [-1] * hash_size
        self.chain = {}

    def add(self, key, index):
        h = key & self.mask
        self.chain[index] = self.hash[h]
        self.hash[h] = index

    def walk(self, key):
        i = self.hash[key & self.mask]
        while i >= 0:
            yield i
            i = self.chain[i]


def vector_subset_remap(vecs, epsilon, box_hash_size=32):
    """idVectorSubset<type,dim>::FindVector, run over a whole list.

    Returns, for every entry, either its own index or the index of an earlier
    entry within `epsilon` on every axis.
    """
    vecs = np.asarray(vecs, F32)
    n, dim = vecs.shape
    if n == 0:
        return np.zeros(0, np.int64)
    expand = F32(2 * box_hash_size * epsilon)
    mins = (vecs.min(0) - expand).astype(F32)
    maxs = (vecs.max(0) + expand).astype(F32)
    box_size = ((maxs - mins) / F32(box_hash_size)).astype(F32)
    inv = (F32(1.0) / box_size).astype(F32)
    half = (box_size * F32(0.5)).astype(F32)
    h = HashIndex(box_hash_size ** dim)
    remap = np.arange(n, dtype=np.int64)
    eps = F32(epsilon)
    corners = 1 << dim
    for i in range(n):
        v = vecs[i]
        partial = [int(np.trunc((v[k] - mins[k] - half[k]) * inv[k])) for k in range(dim)]
        found = -1
        for corner in range(corners):
            key = 0
            for k in range(dim):
                key = key * box_hash_size + partial[k] + ((corner >> k) & 1)
            for j in h.walk(key):
                if np.all(np.abs(vecs[j] - v) <= eps):
                    found = j
                    break
            if found >= 0:
                break
        if found >= 0:
            remap[i] = found
            continue
        key = 0
        for k in range(dim):
            key = key * box_hash_size + int(np.trunc((v[k] - mins[k]) * inv[k]))
        h.add(key, i)
        remap[i] = i
    return remap


# =============================================================================
#  LWO2 / LWOB reader -- renderer/Model_lwo.cpp
# =============================================================================

def _u2(d, p):
    return struct.unpack_from('>H', d, p)[0], p + 2


def _f4(d, p):
    return struct.unpack_from('>f', d, p)[0], p + 4


def _vx(d, p):
    """LWO's variable-width index: 2 bytes unless the first byte is 0xFF."""
    if d[p] != 0xFF:
        return struct.unpack_from('>H', d, p)[0], p + 2
    return struct.unpack_from('>I', d, p)[0] & 0x00FFFFFF, p + 4


def _s0(d, p):
    e = d.index(b'\x00', p)
    s = d[p:e].decode('latin1')
    n = e + 1
    if (n - p) & 1:
        n += 1
    return s, n


class _Layer(object):
    pass


def parse_lwo(path):
    """lwGetObject().  Returns every layer, but only the first is ever used --
    the engine does `lwLayer *layer = lwo->layer;` and never walks the list.
    """
    with open(path, 'rb') as fp:
        d = fp.read()
    if d[:4] != b'FORM':
        raise ValueError('%s is not an IFF file' % os.path.basename(path))
    formtype = d[8:12]
    if formtype not in (b'LWO2', b'LWOB'):
        raise ValueError('%s: unsupported LightWave form %r'
                         % (os.path.basename(path), formtype))
    lwob = (formtype == b'LWOB')
    off = 12
    tags = []
    surfaces = []
    layers = []
    layer = None
    total = len(d)
    while off + 8 <= total:
        cid = d[off:off + 4]
        cksize, = struct.unpack_from('>I', d, off + 4)
        body = d[off + 8:off + 8 + cksize]
        off += 8 + cksize + (cksize & 1)
        if cid == b'LAYR':
            layer = _Layer()
            layer.points = np.zeros((0, 3), F32)
            layer.polys = []
            layer.poly_type = []
            layer.poly_surf_tag = []
            layer.poly_smgp = []
            layer.vmaps = []
            layer.name = ''
            if len(body) >= 16:
                layer.name, _ = _s0(body, 16)
            layers.append(layer)
        elif cid == b'PNTS':
            if layer is None:
                layer = _Layer()
                layer.polys = []; layer.poly_type = []; layer.poly_surf_tag = []
                layer.poly_smgp = []; layer.vmaps = []; layer.name = ''
                layers.append(layer)
            n = cksize // 12
            layer.points = np.frombuffer(body[:n * 12], dtype='>f4').astype(F32).reshape(n, 3)
        elif cid == b'POLS':
            ptype = body[:4] if not lwob else b'FACE'
            p = 4 if not lwob else 0
            while p < len(body):
                nv, p = _u2(body, p)
                nv &= 0x03FF
                idxs = []
                for _ in range(nv):
                    v, p = _vx(body, p)
                    idxs.append(v)
                if lwob:
                    # LWOB stores the surface index inline after the vertices
                    surf_i = struct.unpack_from('>h', body, p)[0]
                    p += 2
                    if surf_i < 0:      # negative == detail polygons follow
                        surf_i = -surf_i
                        _ndetail, p = _u2(body, p)
                    layer.poly_surf_tag.append(surf_i - 1)
                else:
                    layer.poly_surf_tag.append(-1)
                layer.polys.append(tuple(idxs))
                layer.poly_type.append(ptype)
                layer.poly_smgp.append(-1)
        elif cid == b'PTAG':
            sub = body[:4]
            p = 4
            while p < len(body):
                pi, p = _vx(body, p)
                tg, p = _u2(body, p)
                if pi < len(layer.polys):
                    if sub == b'SURF':
                        layer.poly_surf_tag[pi] = tg
                    elif sub == b'SMGP':
                        layer.poly_smgp[pi] = tg
        elif cid == b'TAGS' or (lwob and cid == b'SRFS'):
            p = 0
            while p < len(body):
                s, p = _s0(body, p)
                tags.append(s)
        elif cid in (b'VMAP', b'VMAD'):
            perpoly = (cid == b'VMAD')
            vtype = body[:4]
            p = 4
            dim, p = _u2(body, p)
            _name, p = _s0(body, p)
            vidx, pidx, vals = [], [], []
            while p < len(body):
                v, p = _vx(body, p)
                if perpoly:
                    q, p = _vx(body, p)
                    pidx.append(q)
                vidx.append(v)
                vals.append(struct.unpack_from('>%df' % dim, body, p))
                p += 4 * dim
            layer.vmaps.append(dict(
                type=vtype, dim=dim, perpoly=perpoly,
                vindex=np.array(vidx, np.int64),
                pindex=np.array(pidx, np.int64) if perpoly else None,
                val=(np.array(vals, F32).reshape(len(vidx), dim)
                     if vals else np.zeros((0, dim), F32)),
                nverts=len(vidx), offset=0))
        elif cid == b'SURF':
            name, p = _s0(body, 0)
            if not lwob:
                _src, p = _s0(body, p)
            smooth = 0.0
            color = (0.78431, 0.78431, 0.78431)
            while p + 6 <= len(body):
                sid = body[p:p + 4]
                ssz, _ = _u2(body, p + 4)
                sd = body[p + 6:p + 6 + ssz]
                if sid == b'COLR':
                    if lwob and ssz >= 4:
                        color = (sd[0] / 255.0, sd[1] / 255.0, sd[2] / 255.0)
                    elif ssz >= 12:
                        color = struct.unpack_from('>3f', sd, 0)
                elif sid == b'FLAG' and lwob and ssz >= 2:
                    flags, _ = _u2(sd, 0)
                    if flags & 4:
                        smooth = 1.56207        # lwGetSurface5's fixed angle
                elif sid == b'SMAN' and ssz >= 4:
                    smooth, _ = _f4(sd, 0)
                p += 6 + ssz + (ssz & 1)
            surfaces.append(dict(name=name, smooth=float(smooth), color=color))

    by_name = {}
    for s in surfaces:
        by_name.setdefault(s['name'], s)
    for lay in layers:
        lay.poly_smgp = [0 if g < 0 else g for g in lay.poly_smgp]
        names, smooth, colors, idx = [], [], [], []
        lut = {}
        for i in range(len(lay.polys)):
            t = lay.poly_surf_tag[i]
            nm = tags[t] if 0 <= t < len(tags) else 'default'
            if nm not in lut:
                s = by_name.get(nm, dict(name=nm, smooth=0.0,
                                         color=(0.78431, 0.78431, 0.78431)))
                lut[nm] = len(names)
                names.append(nm)
                smooth.append(float(s['smooth']))
                colors.append(tuple(s['color']))
            idx.append(lut[nm])
        lay.surf_names = names
        lay.surf_smooth = smooth
        lay.surf_color = colors
        lay.poly_surf = np.array(idx, np.int64)
    return layers



# =============================================================================
#  ASE reader -- renderer/Model_ase.cpp
# =============================================================================
#
# An .ase reaches renderbump through a completely separate door from a .lwo:
# ASE_Load, then ConvertASEToModelSurfaces.  The two differ in ways the bake
# can see, so nothing below is shared with the LWO path:
#
#   * normals are READ from the file (*MESH_NORMALS) rather than recomputed
#     from smoothing groups -- there is no SMAN angle, no lwGetVertNormals,
#     and *MESH_SMOOTHING is tokenised and thrown away
#   * every parsed normal is rotated by the object's *NODE_TM, then
#     normalised.  Positions are NOT transformed.
#   * v is inverted at parse time ("our OpenGL second texture axis is
#     inverted from MAX's sense"), which is the same convention
#     EngineSource.tv_list already carries for .lwo
#   * winding is flipped at parse time (A, C, B) for *MESH_FACE,
#     *MESH_TFACE and *MESH_CFACE alike
#   * positions and texcoords are indexed separately, per corner
#   * the vertex weld runs per *GEOMOBJECT rather than once for the whole
#     file, and the resulting surfaces are then merged by material
#   * *MESH_MTLID is tokenised and thrown away -- ConvertASEToModelSurfaces
#     reads exactly one material per object, via *MATERIAL_REF

BASE_GAMEDIR = 'base'

_ASE_WS = re.compile(r'[\x00-\x20]*')
_ASE_TOK = re.compile(r'[^\x00-\x20]*')
_ASE_LINE = re.compile(r'[^\n\r]*')
_ASE_ATOI = re.compile(r'[ \t\n\r\f\v]*[-+]?[0-9]+')
_ASE_ATOF = re.compile(r'[ \t\n\r\f\v]*[-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)'
                       r'(?:[eE][-+]?[0-9]+)?')


def _atoi(s):
    """C atoi: the leading integer, or 0."""
    m = _ASE_ATOI.match(s or '')
    return int(m.group(0)) if m else 0


def _atof(s):
    """C atof: the leading float, or 0."""
    m = _ASE_ATOF.match(s or '')
    return float(m.group(0)) if m else 0.0


def ase_os_path_to_relative(os_path, fs_game=''):
    """idFileSystemLocal::OSPathToRelativePath (framework/FileSystem.cpp).

    An .ase names its material with the *BITMAP path the art package wrote,
    e.g. "//Purgatory/purgatory/doom/base/models/x/y.tga".  The engine keeps
    everything after the first path SEGMENT equal to "base" -- or, failing
    that, to fs_game then fs_game_base -- and returns the empty string when
    nothing anchors, which is how a foreign .ase ends up with every object
    sharing one nameless material.

    The segment test wants a slash on both sides, so a path that begins
    "base/..." does not anchor: the engine reads the character before the
    match and requires it to be a slash.  The search is case sensitive,
    because strstr is.
    """
    p = os_path.replace('\\', '/')
    for anchor in (BASE_GAMEDIR, fs_game):
        if not anchor:
            continue
        i = p.find(anchor)
        while i != -1:
            c1 = p[i - 1] if i > 0 else '\0'
            after = i + len(anchor)
            c2 = p[after] if after < len(p) else '\0'
            if c1 == '/' and c2 == '/':
                s = p.find('/', i)
                return p[s + 1:] if s >= 0 else ''
            i = p.find(anchor, i + 1)
    return ''


class _AseLexer(object):
    """ASE_GetToken.

    Whitespace-delimited, where "whitespace" is anything <= 32, plus a
    rest-of-line mode.  Quotes are not special to it: a *BITMAP path
    containing a space tokenises into pieces and the engine keeps only the
    first.  The rest-of-line mode skips leading whitespace before it starts
    collecting, and a newline IS whitespace -- so asking for the rest of a
    line you have already consumed hands back the whole of the NEXT line.
    """

    __slots__ = ('s', 'n', 'p')

    def __init__(self, text):
        self.s = text
        self.n = len(text)
        self.p = 0

    def get(self, rest_of_line=False):
        n = self.n
        if self.p >= n:
            return None
        s = self.s
        p = _ASE_WS.match(s, self.p).end()
        m = (_ASE_LINE if rest_of_line else _ASE_TOK).match(s, p)
        e = m.end()
        # the delimiter itself is consumed
        self.p = e + 1 if e < n else n
        return m.group(0)


class _AseMesh(object):
    """aseMesh_t.

    ASE_ParseGeomObject memsets the whole aseObject_t, so a file with no
    *NODE_TM leaves `transform` as the ZERO matrix rather than as garbage --
    and every normal it rotates then comes out (0,0,0).  (The memset inside
    ASE_KeyGEOMOBJECT's *MESH branch really does size a POINTER instead of
    the struct, but the enclosing object was already cleared, so the bug
    never shows.)
    """

    __slots__ = ('num_vertexes', 'num_tvertexes', 'num_cvertexes',
                 'num_faces', 'num_tvfaces', 'num_cvfaces', 'transform',
                 'colors_parsed', 'normals_parsed', 'vertexes', 'tvertexes',
                 'cvertexes', 'face_vert', 'face_tvert', 'face_color',
                 'vertex_normals')

    def __init__(self):
        self.num_vertexes = 0
        self.num_tvertexes = 0
        self.num_cvertexes = 0
        self.num_faces = 0
        self.num_tvfaces = 0
        self.num_cvfaces = 0
        self.transform = np.zeros((4, 3), F32)
        self.colors_parsed = False
        self.normals_parsed = False
        self.vertexes = None
        self.tvertexes = None
        self.cvertexes = None
        self.face_vert = None
        self.face_tvert = None
        self.face_color = None
        self.vertex_normals = None


class _AseObject(object):
    """aseObject_t, minus the animation frames renderbump never reads."""

    __slots__ = ('name', 'material_ref', 'mesh')

    def __init__(self):
        self.name = ''
        self.material_ref = 0
        self.mesh = _AseMesh()


class AseModel(object):
    """aseModel_t."""

    __slots__ = ('materials', 'objects', 'notes')

    def __init__(self):
        self.materials = []
        self.objects = []
        self.notes = []


def _ase_rotate_normal(n, t):
    """The *NODE_TM rotation ASE_KeyMESH_NORMALS applies to every normal it
    parses, followed by idVec3::Normalize.

    out = n * T, with T the first three *TM_ROWs; *TM_ROW3 (the translation)
    is parsed and never used, and neither is any of it applied to positions.
    Normalize on a zero vector leaves it zero: idMath::InvSqrt(0) is a large
    finite number and scales the zero back to zero.
    """
    t = t.astype(np.float64)
    out = np.array([n[0] * t[0][c] + n[1] * t[1][c] + n[2] * t[2][c]
                    for c in range(3)], np.float64).astype(F32)
    sq = F32(float(out[0]) ** 2 + float(out[1]) ** 2 + float(out[2]) ** 2)
    if sq <= 0:
        return np.zeros(3, F32)
    inv = F32(1.0 / math.sqrt(float(sq)))
    return (out * inv).astype(F32)


class _AseParser(object):
    """ASE_Parse.  One method per ASE_Key* callback, dispatching on the same
    tokens in the same order, so an unknown token is ignored in exactly the
    places the original ignores it.
    """

    def __init__(self, text, fs_game=''):
        self.lex = _AseLexer(text)
        self.fs_game = fs_game
        self.model = AseModel()
        self.cur_object = None
        self.cur_mesh = None
        self.cur_face = 0
        self.cur_vertex = 0

    # -- ASE_ParseBracedBlock / ASE_SkipEnclosingBraces ------------------

    def braced(self, parser):
        indent = 0
        while True:
            t = self.lex.get()
            if t is None:
                return
            if t == '{':
                indent += 1
            elif t == '}':
                indent -= 1
                if indent == 0:
                    return
                if indent < 0:
                    raise ValueError("unexpected '}' in .ase")
            elif parser is not None:
                parser(t)

    def skip_braces(self):
        self.braced(None)

    # -- materials -------------------------------------------------------

    def key_map_diffuse(self, token):
        lex = self.lex
        if token == '*BITMAP':
            tok = lex.get() or ''
            # remove the quotes: skip the leading one, truncate at the next.
            # A path with a space in it was already cut short by the
            # tokenizer, and then there is no closing quote to find.
            body = tok[1:]
            q = body.find('"')
            if q >= 0:
                body = body[:q]
            self.model.materials[-1]['name'] = ase_os_path_to_relative(
                body.replace('\\', '/'), self.fs_game)
        elif token == '*UVW_U_OFFSET':
            self.model.materials[-1]['uOffset'] = _atof(lex.get())
        elif token == '*UVW_V_OFFSET':
            self.model.materials[-1]['vOffset'] = _atof(lex.get())
        elif token == '*UVW_U_TILING':
            self.model.materials[-1]['uTiling'] = _atof(lex.get())
        elif token == '*UVW_V_TILING':
            self.model.materials[-1]['vTiling'] = _atof(lex.get())
        elif token == '*UVW_ANGLE':
            self.model.materials[-1]['angle'] = _atof(lex.get())

    def key_material(self, token):
        # ASE_KeyMATERIAL knows only *MAP_DIFFUSE, and the braced walker
        # hands it every token at every depth -- so a *SUBMATERIAL's own
        # *MAP_DIFFUSE writes over the TOP-LEVEL material's name and texture
        # matrix.  With several submaterials the last one wins, and
        # *MATERIAL_NAME is never read at all.
        if token == '*MAP_DIFFUSE':
            self.braced(self.key_map_diffuse)

    def key_material_list(self, token):
        if token == '*MATERIAL':
            self.model.materials.append(
                dict(name='', uOffset=0.0, vOffset=0.0,
                     uTiling=1.0, vTiling=1.0, angle=0.0))
            self.braced(self.key_material)

    # -- node transform --------------------------------------------------

    def key_node_tm(self, token):
        for row in range(4):
            if token == '*TM_ROW%i' % row:
                for i in range(3):
                    self.cur_object.mesh.transform[row][i] = \
                        F32(_atof(self.lex.get()))
                return

    # -- mesh ------------------------------------------------------------

    def key_vertex_list(self, token):
        m = self.cur_mesh
        if token != '*MESH_VERTEX':
            raise ValueError("unknown token '%s' while parsing "
                             "MESH_VERTEX_LIST" % token)
        lex = self.lex
        lex.get()                                   # skip number
        i = self.cur_vertex
        if i >= m.num_vertexes:
            raise ValueError('ase.currentVertex >= pMesh->numVertexes')
        for k in range(3):
            m.vertexes[i][k] = F32(_atof(lex.get()))
        self.cur_vertex += 1

    def key_tvert_list(self, token):
        m = self.cur_mesh
        if token != '*MESH_TVERT':
            raise ValueError("unknown token '%s' while parsing "
                             "MESH_TVERTLIST" % token)
        lex = self.lex
        lex.get()                                   # skip number
        u = lex.get()
        v = lex.get()
        lex.get()                                   # w, read and discarded
        i = self.cur_vertex
        if i >= m.num_tvertexes:
            raise ValueError('ase.currentVertex > pMesh->numTVertexes')
        m.tvertexes[i][0] = F32(_atof(u))
        # "our OpenGL second texture axis is inverted from MAX's sense"
        m.tvertexes[i][1] = F32(1.0 - _atof(v))
        self.cur_vertex += 1

    def key_cvert_list(self, token):
        """ASE_KeyMESH_CVERTLIST, bug for bug.

        Two of them.  colorsParsed is set on entry to this callback rather
        than on the *MESH_CFACELIST that actually supplies the per-corner
        indices; and each component is read with `atof( token )` -- the
        FUNCTION PARAMETER, which is the literal string "*MESH_VERTCOL" --
        instead of `atof( ase.token )`, the number just tokenised.  atof of
        that is 0, so every vertex colour in every .ase the engine has ever
        loaded is (0, 0, 0).  The tokens are still consumed, so the parse
        stays in step; the only casualty is the data.
        """
        m = self.cur_mesh
        m.colors_parsed = True
        if token != '*MESH_VERTCOL':
            raise ValueError("unknown token '%s' while parsing "
                             "MESH_CVERTLIST" % token)
        lex = self.lex
        lex.get()                                   # skip number
        i = self.cur_vertex
        if i >= m.num_cvertexes:
            raise ValueError('ase.currentVertex > pMesh->numCVertexes')
        for k in range(3):
            lex.get()
            m.cvertexes[i][k] = F32(_atof(token))   # not _atof(lex.get())
        self.cur_vertex += 1

    def key_face_list(self, token):
        m = self.cur_mesh
        if token != '*MESH_FACE':
            raise ValueError("unknown token '%s' while parsing "
                             "MESH_FACE_LIST" % token)
        lex = self.lex
        j = self.cur_face
        if j >= m.num_faces:
            raise ValueError('more *MESH_FACE lines than *MESH_NUMFACES')
        lex.get()                                   # skip face number
        # "we are flipping the order here to change the front/back facing
        #  from 3DS to our standard (clockwise facing out)"
        for slot in (0, 2, 1):
            lex.get()                               # skip the A:/B:/C: label
            m.face_vert[j][slot] = _atoi(lex.get())
        # the AB:/BC:/CA: edge flags, *MESH_SMOOTHING and *MESH_MTLID all go
        # here.  *MESH_MTLID is parsed nowhere else either: the engine's own
        # handling of it is commented out, so a per-face material id has no
        # effect on anything.
        lex.get(True)
        self.cur_face += 1

    def key_tface_list(self, token):
        m = self.cur_mesh
        if token != '*MESH_TFACE':
            raise ValueError("unknown token '%s' in MESH_TFACE" % token)
        lex = self.lex
        j = self.cur_face
        if j >= m.num_faces:
            raise ValueError('more *MESH_TFACE lines than *MESH_NUMFACES')
        lex.get()                                   # skip face number
        a = _atoi(lex.get())
        c = _atoi(lex.get())
        b = _atoi(lex.get())
        m.face_tvert[j][0] = a
        m.face_tvert[j][1] = b
        m.face_tvert[j][2] = c
        self.cur_face += 1

    def key_cface_list(self, token):
        m = self.cur_mesh
        if token != '*MESH_CFACE':
            raise ValueError("unknown token '%s' in MESH_CFACE" % token)
        lex = self.lex
        j = self.cur_face
        if j >= m.num_faces:
            raise ValueError('more *MESH_CFACE lines than *MESH_NUMFACES')
        lex.get()                                   # skip face number
        remap = (0, 2, 1)                           # the same winding flip
        for i in range(3):
            a = _atoi(lex.get())
            # the original indexes cvertexes with no range check at all.
            # Every entry is zero regardless (see key_cvert_list), so
            # clamping changes no value and keeps a file that lies about its
            # counts from walking off the end.
            a = min(max(a, 0), m.num_cvertexes - 1)
            for ch in range(3):
                if a < 0:
                    break
                m.face_color[j][remap[i]][ch] = \
                    int(np.trunc(float(m.cvertexes[a][ch]) * 255.0)) & 0xFF
        self.cur_face += 1

    def key_mesh_normals(self, token):
        m = self.cur_mesh
        lex = self.lex
        m.normals_parsed = True
        if token == '*MESH_FACENORMAL':
            num = _atoi(lex.get())
            if num >= m.num_faces or num < 0:
                raise ValueError('MESH_NORMALS face index out of range: %i'
                                 % num)
            if num != self.cur_face:
                raise ValueError('MESH_NORMALS face index != currentFace')
            # the face normal is parsed and rotated, and then nothing in the
            # renderbump path ever reads it
            lex.get()
            lex.get()
            lex.get()
            self.cur_face += 1
        elif token == '*MESH_VERTEXNORMAL':
            num = _atoi(lex.get())
            if num >= m.num_vertexes or num < 0:
                raise ValueError('MESH_NORMALS vertex index out of range: %i'
                                 % num)
            j = self.cur_face - 1
            face = m.face_vert[j]
            for v in range(3):
                if num == face[v]:
                    break
            else:
                raise ValueError("MESH_NORMALS vertex index doesn't match "
                                 "face")
            n = (_atof(lex.get()), _atof(lex.get()), _atof(lex.get()))
            m.vertex_normals[j][v] = _ase_rotate_normal(n, m.transform)

    def key_mesh(self, token):
        m = self.cur_mesh
        lex = self.lex
        if token == '*MESH_NUMVERTEX':
            m.num_vertexes = _atoi(lex.get())
        elif token == '*MESH_NUMTVERTEX':
            m.num_tvertexes = _atoi(lex.get())
        elif token == '*MESH_NUMCVERTEX':
            m.num_cvertexes = _atoi(lex.get())
        elif token == '*MESH_NUMFACES':
            m.num_faces = _atoi(lex.get())
        elif token == '*MESH_NUMTVFACES':
            m.num_tvfaces = _atoi(lex.get())
            if m.num_tvfaces != m.num_faces:
                raise ValueError('MESH_NUMTVFACES != MESH_NUMFACES')
        elif token == '*MESH_NUMCVFACES':
            m.num_cvfaces = _atoi(lex.get())
            # yes, the original really does re-test numTVFaces here
            if m.num_tvfaces != m.num_faces:
                raise ValueError('MESH_NUMCVFACES != MESH_NUMFACES')
        elif token == '*MESH_VERTEX_LIST':
            m.vertexes = np.zeros((max(m.num_vertexes, 0), 3), F32)
            self.cur_vertex = 0
            self.braced(self.key_vertex_list)
        elif token == '*MESH_TVERTLIST':
            m.tvertexes = np.zeros((max(m.num_tvertexes, 0), 2), F32)
            self.cur_vertex = 0
            self.braced(self.key_tvert_list)
        elif token == '*MESH_CVERTLIST':
            m.cvertexes = np.zeros((max(m.num_cvertexes, 0), 3), F32)
            self.cur_vertex = 0
            self.braced(self.key_cvert_list)
        elif token == '*MESH_FACE_LIST':
            n = max(m.num_faces, 0)
            m.face_vert = np.zeros((n, 3), np.int64)
            m.face_tvert = np.zeros((n, 3), np.int64)
            # aseFace_t comes out of Mem_Alloc, so the alpha byte of a
            # parsed vertex colour is whatever was in the heap -- *MESH_CFACE
            # only ever writes r, g and b.  Zero is what a fresh page holds,
            # and it is the only self-consistent choice: the byte takes part
            # in the vertex match, so it has to be the SAME for every corner
            # or the merge itself would go non-deterministic.  It reaches the
            # output in the alpha channel of the colour map, and nowhere else.
            m.face_color = np.zeros((n, 3, 4), np.uint8)
            m.vertex_normals = np.zeros((n, 3, 3), F32)
            self.cur_face = 0
            self.braced(self.key_face_list)
        elif token == '*MESH_TFACELIST':
            if m.face_vert is None:
                raise ValueError('*MESH_TFACELIST before *MESH_FACE_LIST')
            self.cur_face = 0
            self.braced(self.key_tface_list)
        elif token == '*MESH_CFACELIST':
            if m.face_vert is None:
                raise ValueError('*MESH_CFACELIST before *MESH_FACE_LIST')
            self.cur_face = 0
            self.braced(self.key_cface_list)
        elif token == '*MESH_NORMALS':
            if m.face_vert is None:
                self.model.notes.append('*MESH_NORMALS before *MESH_FACE_LIST')
                self.skip_braces()
                return
            self.cur_face = 0
            self.braced(self.key_mesh_normals)

    def key_mesh_animation(self, token):
        # A frame mesh is parsed into its own aseMesh_t and appended to the
        # object's frame list, which renderbump never looks at.  Note that
        # this leaves currentMesh pointing at the LAST frame, exactly as the
        # original does.
        if token == '*MESH':
            self.cur_mesh = _AseMesh()
            self.braced(self.key_mesh)
        else:
            raise ValueError("unknown token '%s' while parsing "
                             "MESH_ANIMATION" % token)

    def key_geomobject(self, token):
        lex = self.lex
        obj = self.cur_object
        if token == '*NODE_NAME':
            obj.name = lex.get(True) or ''
        elif token == '*NODE_PARENT':
            lex.get(True)
        elif token == '*NODE_TM' or token == '*TM_ANIMATION':
            self.braced(self.key_node_tm)
        elif token == '*MESH':
            self.cur_mesh = obj.mesh
            self.braced(self.key_mesh)
        elif token == '*MATERIAL_REF':
            obj.material_ref = _atoi(lex.get())
        elif token == '*MESH_ANIMATION':
            self.braced(self.key_mesh_animation)
        elif token in ('*PROP_MOTIONBLUR', '*PROP_CASTSHADOW',
                       '*PROP_RECVSHADOW'):
            lex.get(True)

    def parse_geom_object(self):
        self.cur_object = _AseObject()
        self.model.objects.append(self.cur_object)
        self.braced(self.key_geomobject)

    def key_group(self, token):
        if token == '*GEOMOBJECT':
            self.parse_geom_object()

    def run(self):
        lex = self.lex
        while True:
            t = lex.get()
            if t is None:
                break
            if t == '*3DSMAX_ASCIIEXPORT' or t == '*COMMENT':
                lex.get(True)
            elif t == '*SCENE':
                self.skip_braces()
            elif t == '*GROUP':
                lex.get()                           # group name
                self.braced(self.key_group)
            elif t == '*SHAPEOBJECT' or t == '*CAMERAOBJECT':
                self.skip_braces()
            elif t == '*MATERIAL_LIST':
                self.braced(self.key_material_list)
            elif t == '*GEOMOBJECT':
                self.parse_geom_object()
        return self.model


def parse_ase(path, fs_game=''):
    """ASE_Load.

    The engine hands ASE_Parse a NUL-terminated buffer and measures it with
    strlen, so anything after an embedded NUL byte is not part of the file
    as far as the parser is concerned.
    """
    with open(path, 'rb') as fp:
        data = fp.read()
    z = data.find(b'\x00')
    if z >= 0:
        data = data[:z]
    return _AseParser(data.decode('latin1'), fs_game).run()


# =============================================================================
#  EngineSource -- the common shape a .lwo file and a Blender mesh both
#  reduce to before the engine's own model conversion runs
# =============================================================================

class EngineSource(object):
    """Everything ConvertLWOToModelSurfaces reads, in idTech4 model space.

    points      (N,3) float32, already axis-swapped: id.x = lwo.x,
                id.y = lwo.z, id.z = lwo.y
    polys       list of point-index tuples, in FILE order (which is the
                reverse of Blender's loop order -- see below)
    tv_list     (T,2) float32, the texcoord value table, t already inverted
    corner_tv   (C,) int64, one index into tv_list per polygon corner
    poly_surf   (P,) int64 index into surf_*
    poly_smgp   (P,) int64 smoothing group

    An .ase does not reduce to this at all -- ConvertASEToModelSurfaces
    keeps positions and texcoords in separate index streams, welds per
    *GEOMOBJECT rather than per file, and reads its normals off the disk --
    so an ASE source carries `kind` = 'ASE' and hangs the parsed aseModel_t
    off `ase`, leaving every field below unset.  Only `name` and
    `skipped_polys` are common to both.
    """

    __slots__ = ('points', 'polys', 'tv_list', 'corner_tv', 'poly_surf',
                 'poly_smgp', 'surf_names', 'surf_smooth', 'surf_color',
                 'corner_start', 'poly_nverts', 'corner_point', 'name',
                 'skipped_polys', 'kind', 'ase')

    def finish(self):
        self.poly_nverts = np.array([len(p) for p in self.polys], np.int64)
        self.corner_start = np.zeros(len(self.polys) + 1, np.int64)
        np.cumsum(self.poly_nverts, out=self.corner_start[1:])
        if len(self.polys):
            self.corner_point = np.concatenate(
                [np.asarray(p, np.int64) for p in self.polys])
        else:
            self.corner_point = np.zeros(0, np.int64)
        return self


def source_from_lwo(path, layer_index=0):
    """A .lwo straight off disk, read the way the engine reads it."""
    layers = parse_lwo(path)
    if not layers:
        raise ValueError('%s has no layers' % os.path.basename(path))
    lay = layers[layer_index]
    if lay.points is None or len(lay.points) == 0:
        raise ValueError('%s: layer has bad or missing vertex data'
                         % os.path.basename(path))

    src = EngineSource()
    src.kind = 'LWO'
    src.ase = None
    src.name = os.path.basename(path)
    src.skipped_polys = int(sum(1 for q in lay.polys if len(q) != 3))
    pts = lay.points.astype(F32)
    src.points = np.stack([pts[:, 0], pts[:, 2], pts[:, 1]], 1).astype(F32)
    src.polys = lay.polys
    src.poly_surf = lay.poly_surf
    src.poly_smgp = np.array(lay.poly_smgp, np.int64)
    src.surf_names = lay.surf_names
    src.surf_smooth = lay.surf_smooth
    src.surf_color = lay.surf_color
    src.finish()

    # tvList: every TXUV chunk appended in file order, t inverted
    uvmaps = [vm for vm in lay.vmaps if vm['type'] == b'TXUV']
    ntv = sum(vm['nverts'] for vm in uvmaps)
    if ntv:
        tv = np.empty((ntv, 2), F32)
        off = 0
        for vm in uvmaps:
            vm['offset'] = off
            k = vm['nverts']
            tv[off:off + k, 0] = vm['val'][:, 0]
            tv[off:off + k, 1] = (F32(1.0) - vm['val'][:, 1]).astype(F32)
            off += k
    else:
        tv = np.zeros((1, 2), F32)
    src.tv_list = tv

    # Per corner: every continuous VMAP first (last chunk wins), then every
    # discontinuous VMAD on top.  A corner no chunk covers falls back to 0.
    total = int(src.corner_start[-1])
    ctv = np.zeros(total, np.int64)
    npoly = len(src.polys)
    for vm in uvmaps:
        if vm['perpoly']:
            continue
        lut = np.full(len(src.points), -1, np.int64)
        lut[vm['vindex']] = np.arange(vm['nverts'], dtype=np.int64) + vm['offset']
        hit = lut[src.corner_point]
        ctv = np.where(hit >= 0, hit, ctv)
    for vm in uvmaps:
        if not vm['perpoly']:
            continue
        pk = vm['pindex']
        vk = vm['vindex']
        ok = (pk >= 0) & (pk < npoly)
        pk, vk = pk[ok], vk[ok]
        srcidx = (np.arange(vm['nverts'], dtype=np.int64) + vm['offset'])[ok]
        base = src.corner_start[pk]
        cnt = src.poly_nverts[pk]
        pos = np.full(len(pk), -1, np.int64)
        for j in range(int(cnt.max()) if len(cnt) else 0):
            m = (pos < 0) & (j < cnt) & \
                (src.corner_point[np.minimum(base + j, max(total - 1, 0))] == vk)
            pos[m] = base[m] + j
        good = pos >= 0
        ctv[pos[good]] = srcidx[good]
    src.corner_tv = ctv
    return src


def source_from_ase(path, fs_game=''):
    """An .ase straight off disk, read the way the engine reads it.

    There is nothing to flatten here the way source_from_lwo flattens a
    layer: ConvertASEToModelSurfaces works object by object off the parsed
    aseModel_t, so that is what gets carried.  *MESH_FACE is triangles only,
    so the "make sure you triplet it down" skip that the LWO path counts
    cannot arise.
    """
    model = parse_ase(path, fs_game)
    if not model.objects:
        raise ValueError('%s has no *GEOMOBJECT' % os.path.basename(path))
    src = EngineSource()
    src.kind = 'ASE'
    src.ase = model
    src.name = os.path.basename(path)
    src.skipped_polys = 0
    return src


def source_from_file(path, fs_game=''):
    """Pick the reader by extension, the way idRenderModelStatic::InitFromFile
    picks a converter."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.ase':
        return source_from_ase(path, fs_game)
    if ext == '.lwo':
        return source_from_lwo(path)
    raise ValueError('%s: renderbump reads .lwo and .ase models'
                     % os.path.basename(path))


def source_from_object(obj, use_world=True, smooth_angle=math.pi / 2,
                       depsgraph=None):
    """A Blender mesh object, put back into idTech4 model space.

    Blender's .lwo importer reverses each polygon's winding relative to the
    file (an axis swap alone has determinant -1, but LightWave's own
    polygon-normal formula does not cancel it out), so the corner order is
    reversed back here to recover the order ConvertLWOToModelSurfaces would
    have seen.  V is flipped because the engine's tvList inverts t and the
    importer does not.

    Polygons that are not triangles are skipped rather than triangulated,
    because that is what the engine does: "model %s has too many verts for a
    poly! Make sure you triplet it down".
    """
    if depsgraph is not None:
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        owner, temp = eval_obj, True
    else:
        mesh, owner, temp = obj.data, None, False
    try:
        nv = len(mesh.vertices)
        co = np.empty(nv * 3, np.float32)
        mesh.vertices.foreach_get('co', co)
        co = co.reshape(nv, 3)
        if use_world:
            m = np.array(obj.matrix_world, np.float64)
            if not np.allclose(m, np.eye(4)):
                co = (co.astype(np.float64) @ m[:3, :3].T + m[:3, 3]).astype(F32)

        npoly = len(mesh.polygons)
        ls = np.empty(npoly, np.int32)
        mesh.polygons.foreach_get('loop_start', ls)
        lt = np.empty(npoly, np.int32)
        mesh.polygons.foreach_get('loop_total', lt)
        mi = np.empty(npoly, np.int32)
        mesh.polygons.foreach_get('material_index', mi)
        tri_only = lt == 3
        skipped = int((~tri_only).sum())
        if not tri_only.any():
            raise ValueError('%s has no triangles (renderbump needs a '
                             'triangulated mesh)' % obj.name)
        ls, mi = ls[tri_only], mi[tri_only]

        nl = len(mesh.loops)
        lv = np.empty(nl, np.int32)
        mesh.loops.foreach_get('vertex_index', lv)
        # Blender loop order -> engine (file) order is a reversal
        tl = np.stack([ls + 2, ls + 1, ls + 0], 1)
        corner_loop = tl.reshape(-1)

        src = EngineSource()
        # a scene object goes down the LWO conversion, whatever it was
        # imported from
        src.kind = 'LWO'
        src.ase = None
        src.name = obj.name
        src.skipped_polys = skipped
        src.points = co.astype(F32)
        tri_verts = lv[tl].astype(np.int64)
        src.polys = [tuple(int(x) for x in t) for t in tri_verts]
        src.finish()
        src.corner_point = tri_verts.reshape(-1)

        uvl = mesh.uv_layers.active
        if uvl is None:
            raise ValueError('%s has no UV map' % obj.name)
        uv = np.empty(nl * 2, np.float32)
        uvl.uv.foreach_get('vector', uv)
        uv = uv.reshape(nl, 2)[corner_loop]
        # the engine's tvList holds t already inverted; Blender's V is not
        src.tv_list = np.stack([uv[:, 0], (F32(1.0) - uv[:, 1]).astype(F32)],
                               1).astype(F32)
        src.corner_tv = np.arange(len(src.tv_list), dtype=np.int64)

        mats = [(m.name if m else 'default') for m in obj.data.materials] or ['default']
        mi = np.clip(mi, 0, len(mats) - 1)
        used, inv = np.unique(mi, return_inverse=True)
        src.poly_surf = inv.astype(np.int64).reshape(-1)
        src.surf_names = [mats[u] for u in used]
        src.surf_smooth = [float(smooth_angle)] * len(used)
        src.surf_color = [(1.0, 1.0, 1.0)] * len(used)
        src.poly_smgp = np.zeros(len(src.polys), np.int64)
        return src
    finally:
        if temp:
            owner.to_mesh_clear()


# =============================================================================
#  LWO smoothing -- lwGetPolyNormals + lwGetVertNormals
# =============================================================================

def lw_corner_normals(src):
    """Per polygon-corner normals, in idTech4 space.

    lwGetPolyNormals takes each polygon's normal as cross(v[1]-v[0],
    v[last]-v[0]) on the RAW LightWave points.  Evaluated on the axis-swapped
    points that becomes cross(v[last]-v[0], v[1]-v[0]) -- the swap is a parity
    flip, so the cross product picks up a sign that exactly cancels it.  That
    is also the convention R_DeriveFacePlanes uses, so the two agree.

    lwGetVertNormals then sums, for each corner, every other polygon sharing
    that POINT that is in the same smoothing group and whose normal is within
    the surface's smoothing angle, and renormalises.
    """
    pts = src.points
    npoly = len(src.polys)
    nv = src.poly_nverts
    i0 = np.array([p[0] if len(p) else 0 for p in src.polys], np.int64)
    i1 = np.array([p[1] if len(p) > 1 else 0 for p in src.polys], np.int64)
    il = np.array([p[-1] if len(p) else 0 for p in src.polys], np.int64)
    p0 = pts[i0]
    v1 = (pts[i1] - p0).astype(F32)
    v2 = (pts[il] - p0).astype(F32)
    n = np.cross(v2, v1).astype(F32)             # parity-corrected, see above
    ln = np.sqrt((n.astype(np.float64) ** 2).sum(1)).astype(F32)
    ok = (nv >= 3) & (ln > 0)
    poly_norm = np.zeros((npoly, 3), F32)
    poly_norm[ok] = (n[ok] / ln[ok, None]).astype(F32)

    # lwGetPointPolygons
    flat_pt = src.corner_point
    flat_poly = np.repeat(np.arange(npoly, dtype=np.int64), nv)
    order = np.argsort(flat_pt, kind='stable')
    spoly = flat_poly[order]
    counts = np.bincount(flat_pt, minlength=len(pts))
    starts = np.zeros(len(pts) + 1, np.int64)
    np.cumsum(counts, out=starts[1:])

    smg = src.poly_smgp
    smooth = np.array([src.surf_smooth[s] for s in src.poly_surf], np.float64)

    ncorner = len(flat_pt)
    acc = poly_norm[flat_poly].astype(np.float64)
    active = (smooth[flat_poly] > 0) & (nv[flat_poly] >= 3)
    if active.any():
        ca = np.nonzero(active)[0]
        pt_a = flat_pt[ca]
        deg = counts[pt_a]
        rep = np.repeat(ca, deg)
        base = np.repeat(starts[pt_a], deg)
        run = np.arange(len(rep), dtype=np.int64) - np.repeat(
            np.concatenate(([0], np.cumsum(deg)[:-1])), deg)
        nbr = spoly[base + run]
        own = flat_poly[rep]
        keep = (nbr != own) & (smg[nbr] == smg[own])
        rep, nbr = rep[keep], nbr[keep]
        if len(rep):
            pn64 = poly_norm.astype(np.float64)
            dv = np.einsum('ij,ij->i', pn64[flat_poly[rep]], pn64[nbr])
            np.clip(dv, -1.0, 1.0, out=dv)
            keep = np.arccos(dv) <= smooth[flat_poly[rep]]
            rep, nbr = rep[keep], nbr[keep]
            if len(rep):
                contrib = poly_norm[nbr].astype(np.float64)
                for c in range(3):
                    acc[:, c] += np.bincount(rep, weights=contrib[:, c],
                                             minlength=ncorner)
    accf = acc.astype(F32)
    r = np.sqrt((accf.astype(np.float64) ** 2).sum(1))
    out = accf.copy()
    good = active & (r > 0)
    out[good] = (accf[good] / r[good, None].astype(F32)).astype(F32)
    out[~active] = poly_norm[flat_poly[~active]]
    return fix_degenerate_normal(out)


# =============================================================================
#  ConvertLWOToModelSurfaces
# =============================================================================

class Tri(object):
    """The parts of srfTriangles_t that renderbump touches."""
    __slots__ = ('xyz', 'st', 'normal', 'color', 'indexes', 'sil', 'tangents',
                 'dup_verts', 'face_planes', 'num_mirrored')


SLOP_VERTEX = 0.01          # r_slopVertex
SLOP_TEXCOORD = 0.001       # r_slopTexCoord


def build_low_surfaces(src, progress=None):
    """The full (non-fastLoad) model load, for the low poly.

    Every low-poly surface renderbump ever sees is by definition a surface
    whose material carries a renderBump command, so normalsParsed is always
    false: the file's normals are discarded and matching texcoords alone are
    enough to merge two corners onto one vertex.
    """
    vlist = src.points
    tvlist = src.tv_list
    if progress:
        progress('welding vertices')
    vremap = vector_subset_remap(vlist, SLOP_VERTEX, 32)
    if progress:
        progress('welding texcoords')
    tvremap = vector_subset_remap(tvlist, SLOP_TEXCOORD, 32)

    out = []
    merge_map = {}
    npoly = len(src.polys)
    for si, name in enumerate(src.surf_names):
        col = np.array([int(src.surf_color[si][0] * 255),
                        int(src.surf_color[si][1] * 255),
                        int(src.surf_color[si][2] * 255), 255], np.uint8)
        sel = np.nonzero((src.poly_surf == si) & (src.poly_nverts == 3))[0]

        mv_v, mv_tv, indexes = [], [], []
        mv_hash = {}
        for j in sel:
            base = int(src.corner_start[j])
            for k in range(3):
                c = base + k
                v = int(vremap[src.corner_point[c]])
                tv = int(tvremap[src.corner_tv[c]])
                found = -1
                for mi in mv_hash.get(v, ()):
                    if mv_tv[mi] == tv:
                        found = mi          # normals ignored, see docstring
                        break
                if found < 0:
                    found = len(mv_v)
                    mv_v.append(v)
                    mv_tv.append(tv)
                    mv_hash.setdefault(v, []).append(found)
                indexes.append(found)

        if not indexes:
            continue
        tri = Tri()
        tri.xyz = vlist[np.array(mv_v, np.int64)]
        tri.st = tvlist[np.array(mv_tv, np.int64)]
        tri.normal = np.zeros((len(mv_v), 3), F32)
        tri.color = np.tile(col, (len(mv_v), 1))
        tri.indexes = np.array(indexes, np.int64)
        tri.sil = tri.tangents = tri.dup_verts = tri.face_planes = None
        tri.num_mirrored = 0

        if name in merge_map:
            out[merge_map[name]] = merge_tris(out[merge_map[name]], tri)
        else:
            merge_map[name] = len(out)
            out.append(tri)
    return out


def merge_tris(a, b):
    """R_MergeTriangles."""
    t = Tri()
    t.xyz = np.concatenate([a.xyz, b.xyz])
    t.st = np.concatenate([a.st, b.st])
    t.normal = np.concatenate([a.normal, b.normal])
    t.color = np.concatenate([a.color, b.color])
    t.indexes = np.concatenate([a.indexes, b.indexes + len(a.xyz)])
    t.sil = t.tangents = t.dup_verts = t.face_planes = None
    t.num_mirrored = 0
    return t


def build_high_mesh(src, corner_normals):
    """PartialInitFromFile: the fastLoad path, then CombineModelSurfaces.

    Under fastLoad the vertex and texcoord remaps are both the identity and
    normalEpsilon is 1.0, so two corners can only ever merge when their
    normals are bit-identical -- which makes the merge a no-op on the
    geometry.  So the mesh is simply every corner of every triangle, in
    surface-then-polygon order, which is what CombineModelSurfaces would
    have produced anyway.
    """
    npoly = len(src.polys)
    corner_poly = np.repeat(np.arange(npoly, dtype=np.int64), src.poly_nverts)
    kept = np.nonzero(src.poly_nverts[corner_poly] == 3)[0]
    # surfaces are visited in order, so the corners come out grouped by surface
    sel = kept[np.argsort(src.poly_surf[corner_poly[kept]], kind='stable')]

    surf_col = np.array([[int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), 255]
                         for c in src.surf_color], F32)
    colors = surf_col[src.poly_surf[corner_poly[sel]]]
    return dict(xyz=src.points[src.corner_point[sel]].astype(F32),
                normals=corner_normals[sel].astype(F32),
                colors=colors.astype(F32),
                indexes=np.arange(len(sel), dtype=np.int64))



# =============================================================================
#  ConvertASEToModelSurfaces
# =============================================================================

SLOP_NORMAL = 0.02          # r_slopNormal


def _ase_material(model, obj):
    """ase->materials[object->materialRef].

    The original does that read unchecked; a *MATERIAL_REF past the end of
    the material list walks off the array.  Clamping is the closest
    well-defined thing.
    """
    if not model.materials:
        return None
    i = obj.material_ref
    if i < 0 or i >= len(model.materials):
        i = min(max(i, 0), len(model.materials) - 1)
    return model.materials[i]


def _ase_surface_plan(model):
    """The mergeTo pass at the top of ConvertASEToModelSurfaces.

    r_mergeModelSurfaces defaults on, so objects resolving to one material
    share one surface, and the surfaces come out in the order their
    materials first appear.  With no *MATERIAL_LIST at all the engine drops
    everything into a single default surface.

    Two things it does that are out of reach here: it compares resolved decl
    POINTERS rather than names, and it refuses to merge a discrete material
    (a flare or autosprite).  Both need the decl system.  Comparing the name
    ConvertASEToModelSurfaces would have handed FindMaterial, and treating
    nothing as discrete, is what the .lwo path here already settles for.
    """
    if not model.materials:
        return [0] * len(model.objects), ['<default>']
    merge_to, names = [], []
    for obj in model.objects:
        name = _ase_material(model, obj)['name']
        if name in names:
            merge_to.append(names.index(name))
        else:
            merge_to.append(len(names))
            names.append(name)
    return merge_to, names


def _ase_object_tri(model, obj, fast_load, normals_parsed):
    """One *GEOMOBJECT through the body of ConvertASEToModelSurfaces.

    The weld is per object, not per file: an .ase tracks positions and
    texcoords in separate lists, so this is where the two get unified into
    the single vertex stream the renderer wants.  Under fastLoad both remaps
    are the identity, "renderbump doesn't care about vertex count".

    normalEpsilon is 1 - r_slopNormal here whether or not fastLoad is set.
    That is not an oversight on this side: the LWO branch of the same
    function forces it to 1 under fastLoad ("don't merge unless completely
    exact") and the ASE branch simply does not, so a fastLoad .ase really
    does weld corners whose normals merely agree to within 0.02, and the
    first corner into a slot keeps its normal for all of them.
    """
    mesh = obj.mesh
    nf = mesh.num_faces
    if nf <= 0 or mesh.face_vert is None or mesh.vertexes is None:
        return None
    nverts = len(mesh.vertexes)
    if nverts == 0:
        return None

    if fast_load:
        vremap = np.arange(nverts, dtype=np.int64)
    else:
        vremap = vector_subset_remap(mesh.vertexes, SLOP_VERTEX, 32)

    have_tv = (mesh.num_tvfaces == nf and mesh.num_tvertexes != 0
               and mesh.tvertexes is not None)
    if not have_tv:
        tvremap = np.zeros(0, np.int64)
    elif fast_load:
        tvremap = np.arange(len(mesh.tvertexes), dtype=np.int64)
    else:
        tvremap = vector_subset_remap(mesh.tvertexes, SLOP_TEXCOORD, 32)

    # This loop is per corner and cannot be vectorised -- the match table is
    # order dependent, and so is which corner's normal a merged vertex ends
    # up with -- so everything it touches goes to plain Python lists first.
    # The values are unchanged: tolist() on a float32 array hands back the
    # exact float32 values widened to Python floats, which is the same
    # widening the x87 stack does to them anyway.
    fv = mesh.face_vert.tolist()
    ft = mesh.face_tvert.tolist()
    fc = mesh.face_color.tolist()
    vn = mesh.vertex_normals.tolist()
    vremap = vremap.tolist()
    tvremap = tvremap.tolist()
    colors_parsed = mesh.colors_parsed
    normal_eps = float(F32(1.0) - F32(SLOP_NORMAL))

    # normal, color and tv are initialised once, outside the loop, and only
    # reassigned when the file actually carries that channel
    normal = [0.0, 0.0, 0.0]
    color = (255, 255, 255, 255)                    # identityColor
    tv = 0

    mv_v, mv_tv, mv_col, mv_nrm = [], [], [], []
    mv_hash = {}
    indexes = []
    for j in range(nf):
        face_v = fv[j]
        face_t = ft[j]
        for k in range(3):
            v = face_v[k]
            if v < 0 or v >= nverts:
                raise ValueError('ConvertASEToModelSurfaces: bad vertex '
                                 'index in ASE file')
            v = vremap[v]
            if have_tv:
                t = face_t[k]
                if t < 0 or t >= mesh.num_tvertexes:
                    raise ValueError('ConvertASEToModelSurfaces: bad tex '
                                     'coord index in ASE file')
                tv = tvremap[t]
            if normals_parsed:
                normal = vn[j][k]
            if colors_parsed:
                c = fc[j][k]
                color = (c[0], c[1], c[2], c[3])
            found = -1
            for mi in mv_hash.get(v, ()):
                if mv_tv[mi] != tv:
                    continue
                if mv_col[mi] != color:
                    continue
                if not normals_parsed:
                    # "if we are going to create the normals, just
                    #  matching texcoords is enough"
                    found = mi
                    break
                o = mv_nrm[mi]
                d = float(F32(o[0] * normal[0] + o[1] * normal[1]
                              + o[2] * normal[2]))
                if d > normal_eps:
                    found = mi
                    break
            if found < 0:
                found = len(mv_v)
                mv_v.append(v)
                mv_tv.append(tv)
                mv_col.append(color)
                mv_nrm.append(normal)
                mv_hash.setdefault(v, []).append(found)
            indexes.append(found)

    if not indexes:
        return None

    nv = len(mv_v)
    tri = Tri()
    tri.xyz = mesh.vertexes[np.array(mv_v, np.int64)].astype(F32)
    tri.normal = np.array(mv_nrm, F32).reshape(nv, 3)
    tri.color = np.array(mv_col, np.uint8).reshape(nv, 4)
    tri.indexes = np.array(indexes, np.int64)
    tri.sil = tri.tangents = tri.dup_verts = tri.face_planes = None
    tri.num_mirrored = 0

    # "an ASE allows the texture coordinates to be scaled, translated, and
    # rotated".  Note the sign: uOffset is negated, vOffset is not.  This
    # runs AFTER the weld, on the material the OBJECT points at -- which is
    # why it cannot be folded into the texcoord table up front, since
    # tvRemap's slop welding sees the untransformed values.
    mat = _ase_material(model, obj)
    if mat is None:
        u_off = v_off = 0.0
        u_til = v_til = 1.0
        t_sin, t_cos = 0.0, 1.0
    else:
        u_off = float(F32(-mat['uOffset']))
        v_off = float(F32(mat['vOffset']))
        u_til = float(F32(mat['uTiling']))
        v_til = float(F32(mat['vTiling']))
        t_sin = float(F32(math.sin(mat['angle'])))
        t_cos = float(F32(math.cos(mat['angle'])))
    st = np.zeros((nv, 2), F32)
    if have_tv:
        raw = mesh.tvertexes[np.array(mv_tv, np.int64)].astype(np.float64)
        u = (raw[:, 0] * u_til + u_off).astype(F32).astype(np.float64)
        v = (raw[:, 1] * v_til + v_off).astype(F32).astype(np.float64)
        st[:, 0] = (u * t_cos + v * t_sin).astype(F32)
        st[:, 1] = (u * -t_sin + v * t_cos).astype(F32)
    tri.st = st
    return tri


def build_low_surfaces_ase(src, progress=None):
    """The full (non-fastLoad) model load of an .ase, for the low poly.

    Same reasoning as the .lwo path, reached by a different route: every
    low-poly surface renderbump sees carries a renderBump command, and
    ConvertASEToModelSurfaces answers that by forcing normalsParsed false --
    "completely ignore any explict normals on surfaces with a renderbump
    command" -- so the file's own *MESH_NORMALS are thrown away and matching
    texcoords alone are enough to merge two corners onto one vertex.
    """
    model = src.ase
    merge_to, names = _ase_surface_plan(model)
    out = [None] * len(names)
    n = len(model.objects)
    for oi, obj in enumerate(model.objects):
        if progress:
            progress('welding object %i/%i' % (oi + 1, n))
        tri = _ase_object_tri(model, obj, False, False)
        if tri is None:
            continue
        k = merge_to[oi]
        out[k] = tri if out[k] is None else merge_tris(out[k], tri)
    return [t for t in out if t is not None]


def build_high_mesh_ase(src):
    """PartialInitFromFile on an .ase, then CombineModelSurfaces.

    Unlike the .lwo high-poly path this is not a pass-through -- see
    _ase_object_tri on normalEpsilon -- so the vertex count really does
    shrink and the surviving normals really are the first-come ones.
    """
    model = src.ase
    merge_to, names = _ase_surface_plan(model)
    surfaces = [None] * len(names)
    for oi, obj in enumerate(model.objects):
        tri = _ase_object_tri(model, obj, True, obj.mesh.normals_parsed)
        if tri is None:
            continue
        k = merge_to[oi]
        surfaces[k] = tri if surfaces[k] is None \
            else merge_tris(surfaces[k], tri)
    kept = [t for t in surfaces if t is not None]
    if not kept:
        raise ValueError('%s has no triangles' % src.name)

    xyz, nrm, col, idx = [], [], [], []
    base = 0
    for t in kept:
        xyz.append(t.xyz)
        nrm.append(t.normal)
        col.append(t.color.astype(F32))
        idx.append(t.indexes + base)
        base += len(t.xyz)
    return dict(xyz=np.concatenate(xyz).astype(F32),
                normals=np.concatenate(nrm).astype(F32),
                colors=np.concatenate(col).astype(F32),
                indexes=np.concatenate(idx).astype(np.int64))


def ase_source_notes(src, tag):
    """Everything about an .ase that changes the bake and that the file
    cannot tell us on its own."""
    notes = []
    model = src.ase
    for n in model.notes:
        notes.append('WARNING: %s %s: %s' % (tag, src.name, n))

    if not model.materials:
        notes.append('note: %s %s has no *MATERIAL_LIST, so the engine puts '
                     'every object into one default surface with an identity '
                     'texture matrix' % (tag, src.name))
    else:
        blank = sum(1 for o in model.objects
                    if not _ase_material(model, o)['name'])
        if blank:
            notes.append(
                'note: %s %s has %i object(s) whose *BITMAP path does not sit '
                'under a "base" directory. OSPathToRelativePath returns an '
                'empty material name for those, so the engine merges them all '
                'onto one surface' % (tag, src.name, blank))

    parsed = [o for o in model.objects if o.mesh.normals_parsed]
    zeroed = [o for o in parsed if not o.mesh.transform[:3].any()]
    if zeroed:
        notes.append(
            'WARNING: %s %s has %i object(s) with *MESH_NORMALS but no '
            '*NODE_TM. The engine rotates every parsed normal by that matrix, '
            'and an absent one is all zeros, so every normal on those objects '
            'comes out (0,0,0) -- in the engine too, not just here'
            % (tag, src.name, len(zeroed)))
    if tag == 'high':
        if parsed:
            notes.append(
                'note: the high poly\'s *MESH_NORMALS are being used as read. '
                'The engine would discard them and leave the normals at zero '
                'if that surface\'s own material also carried a renderBump '
                'command, which needs a .mtr lookup to know')
        if len(parsed) < len(model.objects):
            notes.append(
                'WARNING: %s %s has %i object(s) with no *MESH_NORMALS at '
                'all. Nothing regenerates them on this path -- fastLoad '
                'returns from FinishSurfaces before R_CleanupTriangles gets '
                'to run -- so the engine traces those against (0,0,0) '
                'normals too, and the bake will be wrong in the same way'
                % (tag, src.name, len(model.objects) - len(parsed)))
    return notes


# =============================================================================
#  R_CleanupTriangles and friends -- renderer/tr_trisurf.cpp
# =============================================================================

def create_sil_indexes(tri):
    """R_CreateSilIndexes via R_CreateSilRemap -- weld on EXACT xyz only.

    "Uniquing vertexes only on xyz before creating sil edges reduces the edge
    count by about 20% on Q3 models."
    """
    n = len(tri.xyz)
    h = HashIndex(1024)
    remap = np.arange(n, dtype=np.int64)
    xyz = tri.xyz
    keys = ((c_trunc(xyz[:, 0]) + c_trunc(xyz[:, 1]) + c_trunc(xyz[:, 2]))
            & 1023)
    for i in range(n):
        v1 = xyz[i]
        found = -1
        for j in h.walk(int(keys[i])):
            if xyz[j][0] == v1[0] and xyz[j][1] == v1[1] and xyz[j][2] == v1[2]:
                found = j
                break
        if found >= 0:
            remap[i] = found
        else:
            h.add(int(keys[i]), i)
    tri.sil = remap[tri.indexes]
    return tri.sil


def remove_degenerate_triangles(tri):
    """R_RemoveDegenerateTriangles -- silIndexes, not indexes."""
    a, b, c = tri.sil[0::3], tri.sil[1::3], tri.sil[2::3]
    keep = ~((a == b) | (a == c) | (b == c))
    if keep.all():
        return 0
    m = np.repeat(keep, 3)
    tri.indexes = tri.indexes[m]
    tri.sil = tri.sil[m]
    return int((~keep).sum())


def duplicate_mirrored_vertexes(tri):
    """R_DuplicateMirroredVertexes -- split any vertex used by both texture
    polarities so tangent smoothing at that vertex does not degenerate."""
    a = tri.st[tri.indexes[0::3]]
    b = tri.st[tri.indexes[1::3]]
    c = tri.st[tri.indexes[2::3]]
    d0 = (b - a).astype(F32)
    d1 = (c - a).astype(F32)
    area = ((d0[:, 0] * d1[:, 1]).astype(F32)
            - (d0[:, 1] * d1[:, 0]).astype(F32)).astype(F32)
    polarity = (area < 0).astype(np.int64)

    nverts = len(tri.xyz)
    used = np.zeros((nverts, 2), bool)
    idx3 = tri.indexes.reshape(-1, 3)
    for p in (0, 1):
        m = polarity == p
        if m.any():
            used[np.unique(idx3[m].ravel()), p] = True

    both = used[:, 0] & used[:, 1]
    n_extra = int(both.sum())
    tri.num_mirrored = n_extra
    if not n_extra:
        return
    negative_remap = np.zeros(nverts, np.int64)
    negative_remap[both] = np.arange(nverts, nverts + n_extra, dtype=np.int64)
    srcv = np.nonzero(both)[0]
    for field in ('xyz', 'st', 'normal', 'color'):
        arr = getattr(tri, field)
        setattr(tri, field, np.concatenate([arr, arr[srcv]]))
    neg = np.repeat(polarity == 1, 3)
    swap = neg & (negative_remap[tri.indexes] != 0)
    tri.indexes = np.where(swap, negative_remap[tri.indexes], tri.indexes)


def create_dup_verts(tri):
    """R_CreateDupVerts."""
    n = len(tri.xyz)
    remap = np.arange(n, dtype=np.int64)
    remap[tri.indexes] = tri.sil          # duplicate writes: last one wins
    dup = np.nonzero(remap != np.arange(n))[0]
    tri.dup_verts = (np.stack([dup, remap[dup]], 1) if len(dup)
                     else np.zeros((0, 2), np.int64))


def derive_face_planes(tri, mode=RSQRT_EXACT):
    """R_DeriveFacePlanes -> idSIMD::DeriveTriPlanes.

    Note the winding: n = (c-a) x (b-a), the opposite of Blender's own face
    normal from the same index order.  Combined with the LWO axis swap's
    parity flip this comes out pointing the same way as the file's normals.
    """
    a = tri.xyz[tri.indexes[0::3]]
    b = tri.xyz[tri.indexes[1::3]]
    c = tri.xyz[tri.indexes[2::3]]
    d0 = (b - a).astype(F32)
    d1 = (c - a).astype(F32)
    n = np.empty((len(a), 3), F32)
    n[:, 0] = (d1[:, 1] * d0[:, 2]).astype(F32) - (d1[:, 2] * d0[:, 1]).astype(F32)
    n[:, 1] = (d1[:, 2] * d0[:, 0]).astype(F32) - (d1[:, 0] * d0[:, 2]).astype(F32)
    n[:, 2] = (d1[:, 0] * d0[:, 1]).astype(F32) - (d1[:, 1] * d0[:, 0]).astype(F32)
    f = rsqrt((n * n).astype(F32).sum(1, dtype=np.float32), mode)
    n = (n * f[:, None]).astype(F32)
    d = (-(n * a).astype(F32).sum(1, dtype=np.float32)).astype(F32)
    tri.face_planes = (n, d)
    return tri.face_planes


def derive_tangents(tri, mode=RSQRT_EXACT):
    """R_DeriveTangents -> idSIMD::DeriveTangents + NormalizeTangents.

    The per-face tangents are flipped by the sign of the texture-space area
    (the `signBit` xor in the generic implementation, USE_INVA in the
    reference version), summed onto every vertex of the face, and finally
    projected onto the vertex normal's plane.
    """
    i0, i1, i2 = tri.indexes[0::3], tri.indexes[1::3], tri.indexes[2::3]
    a, b, c = tri.xyz[i0], tri.xyz[i1], tri.xyz[i2]
    sa, sb, sc = tri.st[i0], tri.st[i1], tri.st[i2]
    d0 = np.concatenate([(b - a).astype(F32), (sb - sa).astype(F32)], 1)
    d1 = np.concatenate([(c - a).astype(F32), (sc - sa).astype(F32)], 1)

    n = np.empty((len(a), 3), F32)
    n[:, 0] = (d1[:, 1] * d0[:, 2]).astype(F32) - (d1[:, 2] * d0[:, 1]).astype(F32)
    n[:, 1] = (d1[:, 2] * d0[:, 0]).astype(F32) - (d1[:, 0] * d0[:, 2]).astype(F32)
    n[:, 2] = (d1[:, 0] * d0[:, 1]).astype(F32) - (d1[:, 1] * d0[:, 0]).astype(F32)
    n = (n * rsqrt((n * n).astype(F32).sum(1, dtype=np.float32), mode)[:, None]).astype(F32)

    area = ((d0[:, 3] * d1[:, 4]).astype(F32)
            - (d0[:, 4] * d1[:, 3]).astype(F32)).astype(F32)
    sign = np.signbit(area)

    tang = []
    for which in (0, 1):
        t = np.empty((len(a), 3), F32)
        for k in range(3):
            if which == 0:
                t[:, k] = ((d0[:, k] * d1[:, 4]).astype(F32)
                           - (d0[:, 4] * d1[:, k]).astype(F32))
            else:
                t[:, k] = ((d0[:, 3] * d1[:, k]).astype(F32)
                           - (d0[:, k] * d1[:, 3]).astype(F32))
        f = rsqrt((t * t).astype(F32).sum(1, dtype=np.float32), mode)
        f = np.where(sign, -f, f).astype(F32)
        tang.append((t * f[:, None]).astype(F32))

    nv = len(tri.xyz)
    vn = np.zeros((nv, 3), np.float64)
    vt = [np.zeros((nv, 3), np.float64), np.zeros((nv, 3), np.float64)]
    for corner in (i0, i1, i2):
        for k in range(3):
            vn[:, k] += np.bincount(corner, weights=n[:, k].astype(np.float64),
                                    minlength=nv)
            for w in (0, 1):
                vt[w][:, k] += np.bincount(corner,
                                           weights=tang[w][:, k].astype(np.float64),
                                           minlength=nv)
    vn = vn.astype(F32)
    vt = [v.astype(F32) for v in vt]

    # The dupVerts merge, quirk intact: the duplicate takes duplicate+master,
    # then the master takes the duplicate's total.  With three or more verts
    # sharing one position only the last pair's sum survives on the master.
    dv = tri.dup_verts
    for i in range(len(dv)):
        vn[dv[i, 0]] = (vn[dv[i, 0]] + vn[dv[i, 1]]).astype(F32)
    for i in range(len(dv)):
        vn[dv[i, 1]] = vn[dv[i, 0]]

    vn = (vn * rsqrt((vn * vn).astype(F32).sum(1, dtype=np.float32), mode)[:, None]).astype(F32)
    outt = []
    for t in vt:
        d = (t * vn).astype(F32).sum(1, dtype=np.float32)
        t = (t - (d[:, None] * vn).astype(F32)).astype(F32)
        t = (t * rsqrt((t * t).astype(F32).sum(1, dtype=np.float32), mode)[:, None]).astype(F32)
        outt.append(t)
    tri.normal = vn
    tri.tangents = outt


def cleanup_triangles(tri, mode=RSQRT_EXACT):
    """R_CleanupTriangles(tri, createNormals=true, identifySilEdges=true,
    useUnsmoothedTangents=false).  R_IdentifySilEdges and
    R_TestDegenerateTextureSpace change no geometry, so they are not here."""
    create_sil_indexes(tri)
    removed = remove_degenerate_triangles(tri)
    duplicate_mirrored_vertexes(tri)
    create_dup_verts(tri)
    derive_face_planes(tri, mode)
    derive_tangents(tri, mode)
    return removed


def low_mesh_normals(tri, mode=RSQRT_EXACT):
    """RenderBumpTriangles' own smoothed normals.

    "create smoothed normals for the surface, which might be different than
    the normals at the vertexes ... We need properly smoothed normals to make
    sure that the traces always go off normal to the true surface."  The
    silIndexes are rebuilt first, which merges the mirrored verts back
    together, so this sum is complete where the dupVerts merge above is not.
    """
    derive_face_planes(tri, mode)
    create_sil_indexes(tri)
    pn, _ = tri.face_planes
    nv = len(tri.xyz)
    acc = np.zeros((nv, 3), np.float64)
    for k in range(3):
        for c in range(3):
            acc[:, c] += np.bincount(tri.sil[k::3],
                                     weights=pn[:, c].astype(np.float64),
                                     minlength=nv)
    out = np.zeros((nv, 3), F32)
    out[tri.indexes] = normalize_exact(acc.astype(F32)[tri.sil])
    return out


# =============================================================================
#  renderbump.cpp
# =============================================================================

HASH_AXIS_BINS = 100
RAY_STEPS = 100


class TriHash(object):
    __slots__ = ('bounds', 'bin_size', 'bin_start', 'bin_face')


def create_tri_hash(xyz, indexes):
    """CreateTriHash.

    Two epsilon quirks are reproduced verbatim.  `iBounds[0][j] -= 0.001` on
    an int truncates back down, so it decrements the lower bin by one for any
    positive value and leaves zero alone; and the matching `+= 0.001` was
    written against iBounds[0] rather than iBounds[1], so it is a no-op.
    """
    h = TriHash()
    mn = xyz.min(0).astype(F32)
    mx = xyz.max(0).astype(F32)
    h.bounds = np.stack([mn, mx])
    h.bin_size = ((mx - mn) / F32(HASH_AXIS_BINS)).astype(F32)
    if (h.bin_size <= 0).any():
        raise ValueError('CreateTriHash: bad bounds (%s) to (%s)' % (mn, mx))

    tv = xyz[indexes.reshape(-1, 3)]
    lo = c_trunc((tv.min(1).astype(F32) - mn) / h.bin_size)
    lo = np.where(lo > 0, lo - 1, lo)
    lo = np.clip(lo, 0, HASH_AXIS_BINS - 1)
    hi = np.clip(c_trunc((tv.max(1).astype(F32) - mn) / h.bin_size),
                 0, HASH_AXIS_BINS - 1)

    span = hi - lo + 1
    counts = span.prod(1)
    total = int(counts.sum())
    face = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    run = np.arange(total, dtype=np.int64) - np.repeat(
        np.concatenate(([0], np.cumsum(counts)[:-1])), counts)
    syz = np.repeat(span[:, 1] * span[:, 2], counts)
    szr = np.repeat(span[:, 2], counts)
    bi = lo[face, 0] + run // syz
    bj = lo[face, 1] + (run % syz) // szr
    bk = lo[face, 2] + run % szr
    binid = (bi * HASH_AXIS_BINS + bj) * HASH_AXIS_BINS + bk

    order = np.argsort(binid, kind='stable')
    h.bin_face = face[order].astype(np.int32)
    cnt = np.bincount(binid[order], minlength=HASH_AXIS_BINS ** 3)
    h.bin_start = np.zeros(HASH_AXIS_BINS ** 3 + 1, np.int64)
    np.cumsum(cnt, out=h.bin_start[1:])
    return h, total


def _tri_area(a, b, c):
    """idWinding::TriangleArea."""
    cr = np.cross((b - a).astype(F32), (c - a).astype(F32)).astype(F32)
    return (F32(0.5) * np.sqrt((cr.astype(np.float64) ** 2).sum(-1)).astype(F32)).astype(F32)


def sample_high_mesh(rb, points, dirs, chunk=40000, pair_chunk=4000000):
    """SampleHighMesh + TraceToMeshFace, for a whole batch of rays.

    The original walks the ray in 100 fixed steps from -traceDist to
    +traceDist, tests every triangle in each bin it lands in (skipping a bin
    it has already tested on this ray), and keeps any hit at least as far
    along the ray as the best so far.  Because bestDist only ever grows and
    every hit above it is accepted, the surviving hit is simply the farthest
    one found -- which makes the whole thing order-independent and safe to
    vectorise.  Bins repeated on consecutive steps are skipped here too; a
    straight line only ever revisits a bin consecutively, so the candidate
    set is identical.
    """
    n = len(points)
    normal = normalize_exact(dirs)              # "we allow non-normalized directions"
    bounds0 = rb['hash'].bounds[0]
    bin_size = rb['hash'].bin_size
    trace = F32(rb['traceDist'])
    bin_start = rb['hash'].bin_start
    bin_face = rb['hash'].bin_face
    hxyz = rb['xyz']
    hidx = rb['indexes'].reshape(-1, 3)
    hnrm = rb['normals']
    hcol = rb['colors']
    pn, pd = rb['planes']

    got = np.zeros(n, bool)
    out_n = np.zeros((n, 3), F32)
    out_c = np.zeros((n, 4), F32)

    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        P = points[lo:hi]
        N = normal[lo:hi]
        m = hi - lo
        rel = (P - bounds0).astype(F32)

        bins = np.empty((m, RAY_STEPS), np.int64)
        for s in range(RAY_STEPS):
            scale = F32(float(-1.0 + 2.0 * s / RAY_STEPS) * float(trace))
            p = (rel + (N * scale).astype(F32)).astype(F32)
            blk = np.floor(p / bin_size).astype(np.int64)
            ok = ((blk >= 0) & (blk < HASH_AXIS_BINS)).all(1)
            bins[:, s] = np.where(
                ok, (blk[:, 0] * HASH_AXIS_BINS + blk[:, 1]) * HASH_AXIS_BINS + blk[:, 2], -1)

        keep = np.ones_like(bins, bool)
        keep[:, 1:] = bins[:, 1:] != bins[:, :-1]
        keep &= bins >= 0
        ray_of = np.repeat(np.arange(m, dtype=np.int64)[:, None], RAY_STEPS, 1)[keep]
        bid = bins[keep]
        del bins, keep

        cnts = bin_start[bid + 1] - bin_start[bid]
        nz = cnts > 0
        ray_of, bid, cnts = ray_of[nz], bid[nz], cnts[nz]
        if not len(bid):
            continue
        total = int(cnts.sum())
        rep = np.repeat(ray_of, cnts)
        base = np.repeat(bin_start[bid], cnts)
        run = np.arange(total, dtype=np.int64) - np.repeat(
            np.concatenate(([0], np.cumsum(cnts)[:-1])), cnts)
        faces = bin_face[base + run].astype(np.int64)
        del ray_of, bid, cnts, base, run

        best = np.full(m, -float(trace), np.float64)
        nrm_c = np.zeros((m, 3), F32)
        col_c = np.zeros((m, 4), F32)

        for a in range(0, total, pair_chunk):
            bnd = min(total, a + pair_chunk)
            r = rep[a:bnd]
            f = faces[a:bnd]
            pt = P[r]
            nr = N[r]
            fn = pn[f]
            # "only test against planes facing the same direction as our normal"
            d = (fn * nr).astype(F32).sum(1, dtype=np.float32)
            ok = d > F32(0.0001)
            if not ok.any():
                continue
            r, f, pt, nr, fn, d = r[ok], f[ok], pt[ok], nr[ok], fn[ok], d[ok]
            dist = ((fn * pt).astype(F32).sum(1, dtype=np.float32) + pd[f]).astype(F32)
            dist = (dist / (-d)).astype(F32)
            ok = (dist <= trace) & (dist >= -trace)
            if not ok.any():
                continue
            r, f, pt, nr, dist = r[ok], f[ok], pt[ok], nr[ok], dist[ok]
            tri3 = hidx[f]
            v0, v1, v2 = hxyz[tri3[:, 0]], hxyz[tri3[:, 1]], hxyz[tri3[:, 2]]
            # "if normal is inside all edge planes, this face is hit"
            dir0 = (v0 - pt).astype(F32)
            dir1 = (v1 - pt).astype(F32)
            dir2 = (v2 - pt).astype(F32)
            ok = (np.cross(dir0, dir1).astype(F32) * nr).astype(F32).sum(1, dtype=np.float32) <= 0
            ok &= (np.cross(dir1, dir2).astype(F32) * nr).astype(F32).sum(1, dtype=np.float32) <= 0
            ok &= (np.cross(dir2, dir0).astype(F32) * nr).astype(F32).sum(1, dtype=np.float32) <= 0
            if not ok.any():
                continue
            r, tri3, pt, nr, dist = r[ok], tri3[ok], pt[ok], nr[ok], dist[ok]
            v0, v1, v2 = v0[ok], v1[ok], v2[ok]
            tv = (pt + (dist[:, None] * nr).astype(F32)).astype(F32)
            ba = _tri_area(v0, v1, v2)
            b0 = (_tri_area(tv, v1, v2) / ba).astype(F32)
            b1 = (_tri_area(v0, tv, v2) / ba).astype(F32)
            b2 = (_tri_area(v0, v1, tv) / ba).astype(F32)
            ok = (b0 + b1 + b2) <= F32(1.1)
            if not ok.any():
                continue
            r, tri3, dist = r[ok], tri3[ok], dist[ok]
            b0, b1, b2 = b0[ok], b1[ok], b2[ok]
            sn = (hnrm[tri3[:, 0]] * b0[:, None]).astype(F32)
            sn = (sn + (hnrm[tri3[:, 1]] * b1[:, None]).astype(F32)).astype(F32)
            sn = (sn + (hnrm[tri3[:, 2]] * b2[:, None]).astype(F32)).astype(F32)
            sn = normalize_exact(sn)
            sc = (hcol[tri3[:, 0]] * b0[:, None]).astype(F32)
            sc = (sc + (hcol[tri3[:, 1]] * b1[:, None]).astype(F32)).astype(F32)
            sc = (sc + (hcol[tri3[:, 2]] * b2[:, None]).astype(F32)).astype(F32)

            order = np.lexsort((dist.astype(np.float64), r))
            r, dist, sn, sc = r[order], dist[order], sn[order], sc[order]
            last = np.ones(len(r), bool)
            last[:-1] = r[:-1] != r[1:]
            r, dist, sn, sc = r[last], dist[last], sn[last], sc[last]
            better = dist.astype(np.float64) >= best[r]
            r, sn, sc = r[better], sn[better], sc[better]
            best[r] = dist[better].astype(np.float64)
            nrm_c[r] = sn
            col_c[r] = sc

        hit = best > -float(trace)
        got[lo:hi] = hit
        out_n[lo:hi] = nrm_c
        out_c[lo:hi] = col_c
    return got, out_n, out_c


def tri_texture_area(a, b, c):
    """TriTextureArea -- signed, "This may be negatove"."""
    d1x = (b[..., 0] - a[..., 0]).astype(F32)
    d1y = (b[..., 1] - a[..., 1]).astype(F32)
    d2x = (c[..., 0] - a[..., 0]).astype(F32)
    d2y = (c[..., 1] - a[..., 1]).astype(F32)
    cz = ((d1x * d2y).astype(F32) - (d1y * d2x).astype(F32)).astype(F32)
    area = (F32(0.5) * np.abs(cz)).astype(F32)
    return np.where(cz < 0, -area, area).astype(F32)


EDGE_OVERLAP = 4.0


def rasterize_triangle(tri, face, width, height):
    """RasterizeTriangle, up to but not including the trace.

    "we intentionally rasterize somewhat outside the triangles, so the bilerp
    support texels (which may be anti-aliased down) are not just duplications
    of what is on the interior" -- hence the four-texel overlap and the
    interior/edge distinction the caller resolves.

    Returns the covered texels, whether each is interior or edge, and the
    barycentric weights; the caller interpolates position and tangent frame
    later, for only the samples it actually traces.
    """
    ii = tri.indexes.reshape(-1, 3)[face]
    verts = np.empty((3, 2), F32)
    verts[:, 0] = (tri.st[ii, 0] * F32(width) - F32(0.5)).astype(F32)
    verts[:, 1] = (tri.st[ii, 1] * F32(height) - F32(0.5)).astype(F32)
    lo = verts.min(0)
    hi = verts.max(0)
    x0 = int(math.floor(float(lo[0]) - EDGE_OVERLAP))
    x1 = int(math.ceil(float(hi[0]) + EDGE_OVERLAP))
    y0 = int(math.floor(float(lo[1]) - EDGE_OVERLAP))
    y1 = int(math.ceil(float(hi[1]) + EDGE_OVERLAP))
    if x1 <= x0 or y1 <= y0:
        return None

    eo = F32(EDGE_OVERLAP)
    edge = np.empty((3, 3), F32)
    for e in range(3):
        v1 = verts[e]
        v2 = verts[(e + 1) % 3]
        ex = (v2[1] - v1[1]).astype(F32)
        ey = (v1[0] - v2[0]).astype(F32)
        ln = F32(math.sqrt(float(ex) * float(ex) + float(ey) * float(ey)))
        with np.errstate(divide='ignore', invalid='ignore'):
            ex = (ex / ln).astype(F32)
            ey = (ey / ln).astype(F32)
        edge[e] = (ex, ey,
                   (-((v1[0] * ex).astype(F32) + (v1[1] * ey).astype(F32))).astype(F32))

    J, I = np.meshgrid(np.arange(x0, x1, dtype=np.int64),
                       np.arange(y0, y1, dtype=np.int64))
    I = I.ravel()
    J = J.ravel()
    fi = I.astype(F32)
    fj = J.astype(F32)
    with np.errstate(invalid='ignore'):
        d = np.stack([((fi * edge[e, 1]).astype(F32) + (fj * edge[e, 0]).astype(F32)
                       + edge[e, 2]).astype(F32) for e in range(3)])
        # "the edge polarities might be either way"
        inside = ((d[0] >= -eo) & (d[1] >= -eo) & (d[2] >= -eo)) | \
                 ((d[0] <= eo) & (d[1] <= eo) & (d[2] <= eo))
    if not inside.any():
        return None
    I, J, d = I[inside], J[inside], d[:, inside]
    interior = ((d[0] >= 0) & (d[1] >= 0) & (d[2] >= 0)) | \
               ((d[0] <= 0) & (d[1] <= 0) & (d[2] <= 0))

    tvv = np.stack([J.astype(F32), I.astype(F32)], 1)
    base = tri_texture_area(verts[0], verts[1], verts[2])
    with np.errstate(divide='ignore', invalid='ignore'):
        ba = np.stack([
            (tri_texture_area(tvv, verts[1][None, :], verts[2][None, :]) / base).astype(F32),
            (tri_texture_area(verts[0][None, :], tvv, verts[2][None, :]) / base).astype(F32),
            (tri_texture_area(verts[0][None, :], verts[1][None, :], tvv) / base).astype(F32)], 1)
    tot = ba.sum(1, dtype=np.float32)
    ok = (tot >= F32(0.99)) & (tot <= F32(1.01))   # "should never happen"
    if not ok.any():
        return None
    I, J, interior, ba = I[ok], J[ok], interior[ok], ba[ok]
    texel = ((I & (height - 1)) * width + (J & (width - 1))).astype(np.int64)
    return texel, interior, ba


def interpolate_samples(tri, lmn, faces, bary):
    """The rest of RasterizeTriangle's inner loop: the interpolated xyz,
    trace normal, shading normal and tangents at each sample."""
    ii = tri.indexes.reshape(-1, 3)[faces]
    n = len(faces)
    z = np.zeros((n, 3), F32)
    point, tnrm, nrm, tg0, tg1 = z.copy(), z.copy(), z.copy(), z.copy(), z.copy()
    for k in range(3):
        w = bary[:, k][:, None]
        v = ii[:, k]
        point = (point + (w * tri.xyz[v]).astype(F32)).astype(F32)
        # "traceNormal will differ from normal if the surface uses unsmoothedTangents"
        tnrm = (tnrm + (w * lmn[v]).astype(F32)).astype(F32)
        nrm = (nrm + (w * tri.normal[v]).astype(F32)).astype(F32)
        tg0 = (tg0 + (w * tri.tangents[0][v]).astype(F32)).astype(F32)
        tg1 = (tg1 + (w * tri.tangents[1][v]).astype(F32)).astype(F32)
    return point, tnrm, nrm, tg0, tg1


def normal_to_byte(n):
    """r = 128 + 127 * n[i]; then (int)r.

    Truncation, not rounding -- and the expression itself runs wider than
    float on the original's x87 stack, so it is evaluated in double here.
    """
    v = 128.0 + 127.0 * n.astype(np.float64)
    return np.clip(np.trunc(v), 0, 255).astype(np.uint8)


def tangent_transform(t0, t1, n, sampled):
    """mat = {tangents[0], tangents[1], normal}; mat.InverseSelf();
    localNormal = mat * sampledNormal; localNormal.Normalize();

    idMat3::operator*(idVec3) is result[i] = sum_j mat[j][i] * vec[j], so the
    pair is really (M^T)^-1 s: express the sampled normal in the tangent
    basis.  InverseSelf stores its cofactors back into float but keeps the
    determinant and reciprocal in double.
    """
    m = np.empty((len(t0), 3, 3), F32)
    m[:, 0] = t0
    m[:, 1] = t1
    m[:, 2] = n
    d = m.astype(np.float64)

    inv = np.empty((len(t0), 3, 3), F32)
    inv[:, 0, 0] = (d[:, 1, 1] * d[:, 2, 2] - d[:, 1, 2] * d[:, 2, 1]).astype(F32)
    inv[:, 1, 0] = (d[:, 1, 2] * d[:, 2, 0] - d[:, 1, 0] * d[:, 2, 2]).astype(F32)
    inv[:, 2, 0] = (d[:, 1, 0] * d[:, 2, 1] - d[:, 1, 1] * d[:, 2, 0]).astype(F32)
    det = (d[:, 0, 0] * inv[:, 0, 0].astype(np.float64)
           + d[:, 0, 1] * inv[:, 1, 0].astype(np.float64)
           + d[:, 0, 2] * inv[:, 2, 0].astype(np.float64))
    ok = np.abs(det) >= 1e-14                     # MATRIX_INVERSE_EPSILON
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_det = 1.0 / det

    inv[:, 0, 1] = (d[:, 0, 2] * d[:, 2, 1] - d[:, 0, 1] * d[:, 2, 2]).astype(F32)
    inv[:, 0, 2] = (d[:, 0, 1] * d[:, 1, 2] - d[:, 0, 2] * d[:, 1, 1]).astype(F32)
    inv[:, 1, 1] = (d[:, 0, 0] * d[:, 2, 2] - d[:, 0, 2] * d[:, 2, 0]).astype(F32)
    inv[:, 1, 2] = (d[:, 0, 2] * d[:, 1, 0] - d[:, 0, 0] * d[:, 1, 2]).astype(F32)
    inv[:, 2, 1] = (d[:, 0, 1] * d[:, 2, 0] - d[:, 0, 0] * d[:, 2, 1]).astype(F32)
    inv[:, 2, 2] = (d[:, 0, 0] * d[:, 1, 1] - d[:, 0, 1] * d[:, 1, 0]).astype(F32)

    with np.errstate(invalid='ignore'):
        res = (inv.astype(np.float64) * inv_det[:, None, None]).astype(F32)
    res[~ok] = m[~ok]              # InverseSelf() returned false, matrix untouched

    r = res.astype(np.float64)
    sv = sampled.astype(np.float64)
    out = np.empty((len(t0), 3), F32)
    for i in range(3):
        out[:, i] = (r[:, 0, i] * sv[:, 0] + r[:, 1, i] * sv[:, 1]
                     + r[:, 2, i] * sv[:, 2]).astype(F32)
    return normalize_exact(out)


def outline_normal_map(data, width, height, empty=(128, 128, 128)):
    """OutlineNormalMap -- "Puts a single pixel border around all non-empty
    pixels.  Does NOT copy the alpha channel, so it can be used as an alpha
    test map."  Addressing wraps, and the source is a snapshot."""
    img = data.reshape(height, width, 4)
    orig = img.copy()
    er, eg, eb = empty
    is_empty = (orig[:, :, 0] == er) & (orig[:, :, 1] == eg) & (orig[:, :, 2] == eb)
    acc = np.zeros((height, width, 3), np.float64)
    rgb = orig[:, :, :3].astype(np.float64)
    for k in (-1, 0, 1):
        for l in (-1, 0, 1):
            s = np.roll(np.roll(rgb, -l, axis=0), -k, axis=1)
            se = np.roll(np.roll(is_empty, -l, axis=0), -k, axis=1)
            acc += np.where(se[:, :, None], 0.0, s - 128.0)
    ln = np.sqrt((acc ** 2).sum(2))
    ok = is_empty & (ln.astype(F32) >= F32(0.5))    # "no valid samples"
    if not ok.any():
        return data
    with np.errstate(divide='ignore', invalid='ignore'):
        n = (acc / ln[:, :, None]).astype(F32)
    vals = normal_to_byte(n)
    for c in range(3):
        img[:, :, c] = np.where(ok, vals[:, :, c], img[:, :, c])
    return data


def outline_color_map(data, width, height, empty=(128, 128, 128)):
    """OutlineColorMap -- the same bleed, but a plain average."""
    img = data.reshape(height, width, 4)
    orig = img.copy()
    er, eg, eb = empty
    is_empty = (orig[:, :, 0] == er) & (orig[:, :, 1] == eg) & (orig[:, :, 2] == eb)
    acc = np.zeros((height, width, 3), np.float64)
    cnt = np.zeros((height, width), np.int64)
    rgb = orig[:, :, :3].astype(np.float64)
    for k in (-1, 0, 1):
        for l in (-1, 0, 1):
            s = np.roll(np.roll(rgb, -l, axis=0), -k, axis=1)
            se = np.roll(np.roll(is_empty, -l, axis=0), -k, axis=1)
            acc += np.where(se[:, :, None], 0.0, s)
            cnt += (~se).astype(np.int64)
    ok = is_empty & (cnt > 0)
    if not ok.any():
        return data
    n = (acc * (1.0 / np.maximum(cnt, 1)[:, :, None])).astype(F32)
    vals = np.clip(np.trunc(n.astype(np.float64)), 0, 255).astype(np.uint8)
    for c in range(3):
        img[:, :, c] = np.where(ok, vals[:, :, c], img[:, :, c])
    return data


def mip_map(data, width, height):
    """R_MipMap(..., preserveBorder = false) on RGBA8."""
    img = data.reshape(height, width, 4).astype(np.int64)
    out = ((img[0::2, 0::2] + img[0::2, 1::2]
            + img[1::2, 0::2] + img[1::2, 1::2]) >> 2).astype(np.uint8)
    return out.reshape(-1), max(1, width >> 1), max(1, height >> 1)


# =============================================================================
#  The whole command
# =============================================================================

def render_bump(low_src, high_src, width=512, height=512, trace_frac=0.05,
                anti_alias=1, outline=8, rsqrt_mode=RSQRT_EXACT,
                want_global=False, want_color=False, report=None):
    """RenderBump_f, for a single low/high pair.

    Returns dict(local=, glob=, color=, width=, height=), each image a flat
    RGBA8 array in top-to-bottom raster order -- the order the engine lays
    localPic out in, and the order R_WriteTGA writes (it sets the TGA
    top-to-bottom flip bit rather than storing bottom-up).
    """
    def say(msg):
        if report:
            report(msg)

    W = width << anti_alias
    H = height << anti_alias
    if W <= 0 or H <= 0 or (W & (W - 1)) or (H & (H - 1)):
        raise ValueError('renderbump size must be a power of two after '
                         'anti-aliasing (the rasteriser wraps its texel '
                         'coordinates with & (width - 1))')

    t_start = time.time()
    # ---- InitRenderBump: the high poly, fastLoad ----
    say('loading high poly (%s)' % high_src.name)
    if high_src.kind == 'ASE':
        high = build_high_mesh_ase(high_src)
    else:
        hi_norm = lw_corner_normals(high_src)
        high = build_high_mesh(high_src, hi_norm)
    ht = Tri()
    ht.xyz = high['xyz']
    ht.indexes = high['indexes']
    derive_face_planes(ht, rsqrt_mode)
    hashg, nlinks = create_tri_hash(high['xyz'], high['indexes'])
    say('%i triangles made %i links' % (len(high['indexes']) // 3, nlinks))

    mn = high['xyz'].min(0)
    mx = high['xyz'].max(0)
    trace_dist = F32(0.0)
    for i in range(3):
        d = (F32(trace_frac) * F32(mx[i] - mn[i])).astype(F32)
        if d > trace_dist:
            trace_dist = d
    say('trace fraction %4.2f = %6.2f model units' % (trace_frac, trace_dist))
    if trace_dist <= 0:
        raise ValueError('high poly model has no extent')

    rb = dict(xyz=high['xyz'], indexes=high['indexes'], normals=high['normals'],
              colors=high['colors'], planes=ht.face_planes, hash=hashg,
              traceDist=trace_dist)

    # ---- the low poly, full load ----
    say('loading low poly (%s)' % low_src.name)
    if low_src.kind == 'ASE':
        surfaces = build_low_surfaces_ase(low_src, progress=say)
    else:
        surfaces = build_low_surfaces(low_src, progress=say)

    local = np.tile(np.array([128, 128, 128, 0], np.uint8), W * H)
    glob = local.copy()
    color = local.copy()

    # Rasterise every surface first.  edgeDistances lives on the renderBump_t,
    # not on the surface, so a texel an earlier surface filled as an edge texel
    # can still be taken over by a later surface's interior texel -- which is
    # why the resolve below spans all surfaces rather than running per surface.
    prepared = []
    s_texel, s_interior, s_face, s_bary, s_surf, s_order = [], [], [], [], [], []
    face_counter = 0
    for si, tri in enumerate(surfaces):
        removed = cleanup_triangles(tri, rsqrt_mode)
        if removed:
            say('removed %i degenerate triangles' % removed)
        lmn = low_mesh_normals(tri, rsqrt_mode)
        prepared.append((tri, lmn))
        nfaces = len(tri.indexes) // 3
        say('surface %i: rasterising %i triangles' % (si, nfaces))
        for face in range(nfaces):
            got = rasterize_triangle(tri, face, W, H)
            if got is None:
                face_counter += 1
                continue
            tex, inter, ba = got
            s_texel.append(tex.astype(np.int64))
            s_interior.append(inter)
            s_face.append(np.full(len(tex), face, np.int32))
            s_bary.append(ba)
            s_surf.append(np.full(len(tex), si, np.int32))
            s_order.append(np.full(len(tex), face_counter, np.int64))
            face_counter += 1

    if s_texel:
        texel = np.concatenate(s_texel)
        interior = np.concatenate(s_interior)
        face_of = np.concatenate(s_face)
        bary = np.concatenate(s_bary)
        surf_of = np.concatenate(s_surf)
        order_id = np.concatenate(s_order)
        del s_texel, s_interior, s_face, s_bary, s_surf, s_order

        # The sequential edgeDistances state machine reduces to a priority: a
        # texel takes the first triangle, in the engine's own surface-then-face
        # order, that both traces successfully and covers it in the interior;
        # failing that, the first that traces successfully at all.
        rank = np.lexsort((order_id, ~interior, texel))
        texel, interior = texel[rank], interior[rank]
        face_of, bary, surf_of = face_of[rank], bary[rank], surf_of[rank]
        del order_id, rank
        newgrp = np.ones(len(texel), bool)
        newgrp[1:] = texel[1:] != texel[:-1]
        grp_start = np.nonzero(newgrp)[0]
        seq = (np.arange(len(texel))
               - np.repeat(grp_start, np.diff(np.append(grp_start, len(texel)))))
        del newgrp, grp_start

        settled = np.zeros(W * H, bool)
        gv = glob.reshape(-1, 4)
        lv = local.reshape(-1, 4)
        cv = color.reshape(-1, 4)
        rnd = 0
        while True:
            cand = np.nonzero((seq == rnd) & ~settled[texel])[0]
            if not len(cand):
                break
            say('  tracing %i texels (pass %i)' % (len(cand), rnd + 1))
            rnd += 1
            point = np.empty((len(cand), 3), F32)
            tnrm = np.empty((len(cand), 3), F32)
            nrm = np.empty((len(cand), 3), F32)
            tg0 = np.empty((len(cand), 3), F32)
            tg1 = np.empty((len(cand), 3), F32)
            for si, (tri, lmn) in enumerate(prepared):
                m = surf_of[cand] == si
                if not m.any():
                    continue
                sub = cand[m]
                vals = interpolate_samples(tri, lmn, face_of[sub], bary[sub])
                point[m], tnrm[m], nrm[m], tg0[m], tg1[m] = vals

            hit, sn, sc = sample_high_mesh(rb, point, tnrm)
            good = cand[hit]
            if not len(good):
                continue
            sn, sc = sn[hit], sc[hit]
            tx = texel[good]
            settled[tx] = True

            gv[tx, :3] = normal_to_byte(sn)
            gv[tx, 3] = 255
            lv[tx, :3] = normal_to_byte(
                tangent_transform(tg0[hit], tg1[hit], nrm[hit], sn))
            lv[tx, 3] = 255
            cv[tx] = np.clip(np.trunc(sc.astype(np.float64)), 0, 255).astype(np.uint8)

    # ---- WriteRenderBump ----
    passes = outline << anti_alias
    if passes:
        say('outlining %i pixels' % passes)
    for _ in range(passes):
        outline_normal_map(local, W, H)
        if want_global:
            outline_normal_map(glob, W, H)
        if want_color:
            outline_color_map(color, W, H)
    for _ in range(anti_alias):
        local, W2, H2 = mip_map(local, W, H)
        if want_global:
            glob, _, _ = mip_map(glob, W, H)
        if want_color:
            color, _, _ = mip_map(color, W, H)
        W, H = W2, H2

    say('%5.2f seconds for renderBump' % (time.time() - t_start))
    return dict(local=local, glob=glob, color=color, width=W, height=H)


# =============================================================================
#  Blender image output
# =============================================================================

def store_image(name, rgba, width, height, replace=True):
    """Put a flat top-to-bottom RGBA8 buffer into an internal Blender image.

    Blender's pixel buffer starts at the BOTTOM row and the engine's starts
    at the top (R_WriteTGA sets the TGA top-to-bottom flip bit), so the rows
    are flipped here.  Colorspace is Non-Color: a normal map is data, and it
    also makes the float<->byte round trip exact.
    """
    img = bpy.data.images.get(name)
    if img is not None and (not replace or img.size[0] != width or img.size[1] != height):
        img = None
    if img is None:
        img = bpy.data.images.new(name, width, height, alpha=True, float_buffer=False)
    try:
        img.colorspace_settings.name = 'Non-Color'
    except Exception:
        pass
    img.alpha_mode = 'CHANNEL_PACKED'
    buf = rgba.reshape(height, width, 4)[::-1].reshape(-1).astype(np.float32) / np.float32(255.0)
    img.pixels.foreach_set(buf)
    img.pack()
    img.update()
    return img


# =============================================================================
#  Operator / UI
# =============================================================================

_SOURCE_ITEMS = [
    ('OBJECT', 'Scene Object', 'Use a mesh object already in the scene'),
    ('FILE', 'Model File',
     'Read the .lwo or .ase straight off disk, exactly as the engine does'),
]


class IDTECH4_RB_Props(PropertyGroup):
    low_source: EnumProperty(
        name='Low Poly From', items=_SOURCE_ITEMS, default='OBJECT',
        description='Where to read the low-poly model')
    low_object: PointerProperty(
        name='Low Poly', type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'MESH',
        description='Low-poly mesh; its UV layout is what gets baked into')
    low_file: StringProperty(
        name='Low Poly File', subtype='FILE_PATH',
        description='Low-poly .lwo or .ase')

    high_source: EnumProperty(
        name='High Poly From', items=_SOURCE_ITEMS, default='OBJECT',
        description='Where to read the high-poly model')
    high_object: PointerProperty(
        name='High Poly', type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'MESH')
    high_file: StringProperty(
        name='High Poly File', subtype='FILE_PATH',
        description='High-poly .lwo or .ase')

    width: IntProperty(name='Width', default=512, min=1, max=8192,
                       description='renderbump -size <width> <height>')
    height: IntProperty(name='Height', default=512, min=1, max=8192)
    trace_frac: FloatProperty(
        name='Trace', default=0.05, min=0.001, max=1.0, precision=4,
        description='renderbump -trace: how far a ray may travel, as a '
                    'fraction of the high poly\'s longest bounding axis')
    anti_alias: IntProperty(
        name='Anti-Alias', default=1, min=0, max=3,
        description='renderbump -aa: render at 2^n the requested size and '
                    'mip it back down')
    outline: IntProperty(
        name='Outline', default=8, min=0, max=64,
        description='renderbump -outline: pixels of bleed added around every '
                    'filled region, for bilinear filtering and mip-mapping')
    high_smoothing: FloatProperty(
        name='High Smoothing Angle', default=math.pi / 2, min=0.0,
        max=math.pi, subtype='ANGLE',
        description='The high-poly surface\'s LightWave SMAN smoothing '
                    'angle. Only used when the high poly comes from a scene '
                    'object; a .lwo carries its own, and an .ase carries '
                    'finished normals instead. 90 is LightWave\'s default '
                    'and what the shipped models use')
    use_world: BoolProperty(
        name='Use World Transform', default=True,
        description='Bake in each object\'s world matrix, so a low and high '
                    'poly placed differently in the scene still line up')
    rsqrt_mode: EnumProperty(
        name='SIMD', default='EXACT',
        items=[('EXACT', 'SSE (exact)',
                'Match a stock 32-bit Doom 3 build, whose tangent and plane '
                'builders normalise with the SSE rsqrtps instruction'),
               ('QUAKE', 'Generic SIMD (RSqrt)',
                'Match a build with the SIMD assembly compiled out, which '
                'falls back to idMath::RSqrt')])
    want_global: BoolProperty(
        name='Global Map', default=False,
        description='renderbump globalMap: also emit the object-space normal map')
    want_color: BoolProperty(
        name='Color Map', default=False,
        description='renderbump colorMap: also emit the sampled vertex colors')
    image_name: StringProperty(
        name='Image Name', default='',
        description='Name for the generated image; blank uses the low poly\'s '
                    'name plus _local')


def _get_source(props, which):
    src_mode = getattr(props, which + '_source')
    if src_mode == 'FILE':
        path = bpy.path.abspath(getattr(props, which + '_file'))
        if not path or not os.path.isfile(path):
            raise ValueError('%s poly: no such file' % which)
        return source_from_file(path)
    obj = getattr(props, which + '_object')
    if obj is None or obj.type != 'MESH':
        raise ValueError('%s poly: pick a mesh object' % which)
    return source_from_object(obj, use_world=props.use_world,
                              smooth_angle=props.high_smoothing)


class IDTECH4_OT_renderbump(Operator):
    bl_idname = 'idtech4.renderbump'
    bl_label = 'RenderBump'
    bl_description = ('Generate a tangent-space normal map exactly the way '
                      'idTech4\'s renderbump command does')
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.scene.idtech4_renderbump
        wm = context.window_manager
        lines = []

        def say(msg):
            lines.append(msg)
            print('[renderbump] ' + msg)

        try:
            low = _get_source(props, 'low')
            high = _get_source(props, 'high')
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        for tag, src in (('low', low), ('high', high)):
            if src.kind == 'ASE':
                for msg in ase_source_notes(src, tag):
                    say(msg)
            if src.skipped_polys:
                say('WARNING: %s poly %s has %i non-triangle polygons; the '
                    'engine skips those outright ("Make sure you triplet it '
                    'down"), so they contribute nothing'
                    % (tag, src.name, src.skipped_polys))
        if props.high_source == 'OBJECT':
            say('note: the high poly is a scene object. If it came in through '
                'the .lwo importer, that importer welds points that round to '
                'the same position and drops the polygons that collapses -- '
                'the engine does neither on the high poly (it loads it with '
                'fastLoad, where the vertex remap is the identity). Point it '
                'at the .lwo file instead if the two disagree.')

        wm.progress_begin(0, 100)
        try:
            res = render_bump(low, high,
                              width=props.width, height=props.height,
                              trace_frac=props.trace_frac,
                              anti_alias=props.anti_alias,
                              outline=props.outline,
                              rsqrt_mode=props.rsqrt_mode,
                              want_global=props.want_global,
                              want_color=props.want_color,
                              report=say)
        except Exception as exc:
            wm.progress_end()
            self.report({'ERROR'}, 'renderbump failed: %s' % exc)
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
        wm.progress_end()

        base = props.image_name.strip() or (low.name.rsplit('.', 1)[0] + '_local')
        store_image(base, res['local'], res['width'], res['height'])
        made = [base]
        if props.want_global:
            n = base.rsplit('_local', 1)[0] + '_global'
            store_image(n, res['glob'], res['width'], res['height'])
            made.append(n)
        if props.want_color:
            n = base.rsplit('_local', 1)[0] + '_color'
            store_image(n, res['color'], res['width'], res['height'])
            made.append(n)

        self.report({'INFO'}, 'renderbump wrote %s (%ix%i)'
                    % (', '.join(made), res['width'], res['height']))
        return {'FINISHED'}


class IDTECH4_PT_renderbump(Panel):
    bl_label = 'RenderBump'
    bl_idname = 'IDTECH4_PT_renderbump'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'idTech4 Renderbump'

    def draw(self, context):
        layout = self.layout
        props = context.scene.idtech4_renderbump

        box = layout.box()
        box.label(text='Low Poly (the UV layout)')
        box.prop(props, 'low_source', text='')
        if props.low_source == 'OBJECT':
            box.prop(props, 'low_object', text='')
        else:
            box.prop(props, 'low_file', text='')

        box = layout.box()
        box.label(text='High Poly (the detail)')
        box.prop(props, 'high_source', text='')
        if props.high_source == 'OBJECT':
            box.prop(props, 'high_object', text='')
            box.prop(props, 'high_smoothing')
        else:
            box.prop(props, 'high_file', text='')

        col = layout.column(align=True)
        row = col.row(align=True)
        row.prop(props, 'width')
        row.prop(props, 'height')
        col.prop(props, 'trace_frac')
        col.prop(props, 'anti_alias')
        col.prop(props, 'outline')

        col = layout.column(align=True)
        col.prop(props, 'use_world')
        col.prop(props, 'want_global')
        col.prop(props, 'want_color')
        col.prop(props, 'rsqrt_mode')
        col.prop(props, 'image_name')

        layout.operator(IDTECH4_OT_renderbump.bl_idname, icon='RENDER_STILL')


classes = (IDTECH4_RB_Props, IDTECH4_OT_renderbump, IDTECH4_PT_renderbump)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.idtech4_renderbump = PointerProperty(type=IDTECH4_RB_Props)


def unregister():
    del bpy.types.Scene.idtech4_renderbump
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == '__main__':
    register()
