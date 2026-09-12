# =====================================================================
#  idTech4 .map Importer / Exporter for Blender 4.5
#  Imports: idTech4 brushDef3, Quake 3 / Valve 220 (3-point) formats
#  Exports: worldspawn brush geometry as brushDef3 (Version 2 or 3)
#  Also places .ase / .lwo static meshes and MD5 (skeletal) models
#  referenced by "model" keys on .map entities (honoring "origin"/
#  "rotation"/"angles"/"angle" placement) — and, for characters,
#  the head and attachments their entityDef declares via "def_head" /
#  "def_attach*", each bound to a joint of the body's skeleton — when
#  the relevant companion addon — idTech4_ase_lwo_io.py for .ase/.lwo,
#  idTech4_MD5_Tools.py for MD5 — is installed and enabled. Each is
#  disabled automatically in the import dialog when its addon isn't
#  available.
#  Version:  1.0.0
# =====================================================================

bl_info = {
    "name":        "idTech4 .map Importer",
    "author":      "Samson & Claude Sonnet",
    "version":     (1, 0, 0),
    "blender":     (4, 5, 0),
    "location":    "File > Import > idTech4 .map, File > Export > idTech4 .map, View3D > Sidebar > idTech4 Map",
    "description": "Import idTech4 / Quake3 / Valve 220 .map files (brushes + patches + static + MD5 models, with characters' \"def_head\"/\"def_attach\" heads and attachments bound to their body's joints), and export worldspawn brush geometry back to a .map. Static-model (.ase/.lwo) placement requires the companion idTech4_ase_lwo_io addon; MD5 model/animation placement requires idTech4_MD5_Tools, both also installed and enabled",
    "category":    "Import-Export",
}

import bpy
import bmesh
import json
import math
import os
import re
import struct
import sys
import time
import traceback
import addon_utils
from mathutils import Vector, Matrix
from bpy.props import StringProperty, BoolProperty, FloatProperty, EnumProperty, IntProperty
from bpy_extras.io_utils import ImportHelper, ExportHelper
from bpy.types import Operator


# ─────────────────────────────────────────────────────────────────────
#  COMPANION ADDON DETECTION
#  idTech4_ase_lwo_io.py — .ase / .lwo
#  idTech4_MD5_Tools.py — MD5
#  idTech4_material_import - .mtr material creation
# ─────────────────────────────────────────────────────────────────────
# Static-model (.ase/.lwo), MD5 (skeletal) model, and material support live 
# in separate, independently installable addons so any of the three can be
# used without the others. This file only ever calls a small handful of
# plain functions on each — never their registered operators/classes —
# so all that's required is that the module has been imported and its
# addon checkbox is enabled, not that anything about its own UI is
# currently active.
_MODEL_IMPORT_ADDON_NAME    = "idTech4_ase_lwo_io"
_MD5_IMPORT_ADDON_NAME      = "idTech4_MD5_Tools"
_MATERIAL_IMPORT_ADDON_NAME = "idTech4_material_import"


def _find_installed_addon_module(short_name):
    """Locate an addon module among everything Blender's addon system
    knows about, matching by module basename rather than the full
    module path — a legacy (user-scripts) install is just e.g.
    "idTech4_ase_lwo_io", while Blender's Extensions system prefixes
    it (e.g. "bl_ext.user_default.idTech4_ase_lwo_io"). Returns the
    module, or None if it isn't installed at all."""
    for mod in addon_utils.modules():
        mod_name = getattr(mod, '__name__', '')
        if mod_name == short_name or mod_name.endswith('.' + short_name):
            return mod
    return None


def _get_enabled_addon_module(short_name):
    """Return the named addon's module if it's both installed AND
    currently enabled, else None. Re-checked on every call (cheap — a
    scan of already-imported addon modules) rather than cached, so
    toggling the addon on/off in Preferences takes effect immediately
    without needing a Blender restart.

    addon_utils.modules() is only used here to discover the addon's
    real, possibly Extensions-prefixed name and to confirm it's
    enabled. The actual, fully-executed
    module — the one with every real function/class defined — always
    lives in sys.modules under that same name once Blender has
    imported it, which enabling an addon always does; that's what gets
    returned here.
    """
    mod = _find_installed_addon_module(short_name)
    if mod is None:
        return None
    full_name = mod.__name__
    is_enabled = addon_utils.check(full_name)[1]
    if not is_enabled:
        return None
    return sys.modules.get(full_name, mod)


def get_model_import_addon():
    """Return the idTech4_ase_lwo_io module (.ase / .lwo support) if
    it's installed and enabled, else None."""
    return _get_enabled_addon_module(_MODEL_IMPORT_ADDON_NAME)


def get_md5_import_addon():
    """Return the idTech4_MD5_Tools module (MD5 mesh/anim support) if
    it's installed and enabled, else None."""
    return _get_enabled_addon_module(_MD5_IMPORT_ADDON_NAME)


def get_material_import_addon():
    """Return the idTech4_material_import module (.mtr → Blender material
    support) if it's installed and enabled, else None."""
    return _get_enabled_addon_module(_MATERIAL_IMPORT_ADDON_NAME)


# ─────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# Unified scale handling 
# ---------------------------------------------------------------------------
# idTech4 / Doom 3 world units are inches. Blender's default unit is meters.
#   idTech4 -> Blender :  multiply by MD5_SCALE_IN_TO_M  (inches -> meters)
#   Blender -> idTech4 :  multiply by MD5_SCALE_M_TO_IN  (metres -> inches)
MD5_SCALE_IN_TO_M = 0.0254                    # 1 inch     = 0.0254 m
MD5_SCALE_M_TO_IN = 1.0 / MD5_SCALE_IN_TO_M   # 1 metre    = 39.3701 in (approx.)
SCALE        = 0.0254   # idTech4 game units (inches) → metres
# NOTE: ON_PLANE_EPS must stay well below the thinnest brush a mapper would
# realistically create (decals, trim, whiteboards, etc. are routinely
# built as very thin slabs.
ON_PLANE_EPS = 0.01     # game units: how close a point must be to a plane
DEDUP_EPS    = 0.05     # game units: closer than this = same vertex
PLANE_EPS    = 1e-4     # cross-product / determinant near-zero guard

ROTATION_DEGREE_PRESETS = {
    'NONE':   0,                                    # 0°  – no change
    'X_TO_Y': 90,  # +90° Z
    'Y_TO_X': -90,  # -90° Z
    'R180':   180,  # 180° Z
}

IMPORT_ROTATION_ITEMS = [
    ('NONE',   '0°  : No rotation',      'No rotation applied'),
    ('X_TO_Y', '90°   : X to Y',          'Rotate 90° around Z (X-forward → Y-forward, idTech4 → Blender )'),
    ('Y_TO_X', '-90° : Y to X',          'Rotate -90° around Z (Y-forward → X-forward, Blender → idTech4 )'),
    ('R180',   '180° : Flip',            'Rotate 180° around Z'),
]

EXPORT_ROTATION_ITEMS = [
    ('NONE',   '0°  : No rotation',      'No rotation applied'),
    ('Y_TO_X', '-90° : Y to X',          'Rotate -90° around Z (Y-forward → X-forward, Blender → idTech4 )'),
    ('X_TO_Y', '90°   : X to Y',          'Rotate 90° around Z (X-forward → Y-forward, idTech4 → Blender )'),
    ('R180',   '180° : Flip',            'Rotate 180° around Z'),
]

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


class MD5_ImportScaleRotMixin:
    """Shared Scale properties/UI for every MD5 import operator
    (md5mesh, md5anim, md5camera). Scaling is OFF by default (1:1, no
    change). When enabled, choose either an arbitrary numeric factor or the
    idTech4 -> Blender (inches -> meters) preset."""

    scale_mode: EnumProperty(
        name="",
        description="How the import scale factor below is determined",
        items=[
            ('NONE', "None","No Scaling"),
            ('PRESET', "idTech4 to Blender (in \u2192 m)",
             "Convert idTech4 inches to Blender meters. Multiplies every "
             "imported position by %.6g (1 idTech4 unit = 1 inch = %.4f m)"
             % (MD5_SCALE_IN_TO_M, MD5_SCALE_IN_TO_M)),
            ('FACTOR', "Custom Factor",
             "Multiply every imported position by the arbitrary numeric "
             "Scale Factor specified below"),
        ],
        default='NONE',
    )
    scale_factor: FloatProperty(
        name="Scale Factor",
        description="Custom uniform scale factor multiplied into every "
                    "imported position",
        default=1.0, min=0.0001, max=10000.0,
    )

    rotation: EnumProperty(
        name="",
        description="Rotate all imported data around the Z axis",
        items=IMPORT_ROTATION_ITEMS,
        default='NONE',
    )
    def draw_transforms(self, layout):
        layout.label(text="Transforms:")      
        box = layout.box()
        col = box.column()
        col.label(text="Scale Mode:")
        col.prop(self, "scale_mode")
        if self.scale_mode == 'FACTOR':
            col.prop(self, "scale_factor")
        col.label(text="Rotation:")
        col.prop(self, "rotation")

    def get_scale(self):
        """Return the actual multiplier to apply to imported positions."""
        if self.scale_mode == 'NONE':
            return 1.0
        if self.scale_mode == 'FACTOR':
            return self.scale_factor
        return MD5_SCALE_IN_TO_M
        
    def get_rotation(self):
        return self.rotation


class MD5_ExportScaleRotMixin:
    """Shared Scale properties/UI for every MD5 import operator
    (md5mesh, md5anim, md5camera). Scaling is OFF by default (1:1, no
    change). When enabled, choose either an arbitrary numeric factor or the
    Blender -> idTech4 (meters -> inches) preset."""

    scale_mode: EnumProperty(
        name="",
        description="How the import scale factor below is determined",
        items=[
            ('NONE', "None","No Scaling"),
            ('PRESET', "Blender to idTech4 ( m \u2192 in)",
             "Convert Blender meters to idTech4 inches. Multiplies every "
             "imported position by %.6g (1 Blender unit = 1 m = %.4f idTech4 units(inches)."
             % (MD5_SCALE_M_TO_IN, MD5_SCALE_M_TO_IN)),
            ('FACTOR', "Custom Factor",
             "Multiply every imported position by the arbitrary numeric "
             "Scale Factor specified below"),
        ],
        default='NONE',
    )
    scale_factor: FloatProperty(
        name="Scale Factor",
        description="Custom uniform scale factor multiplied into every "
                    "imported position",
        default=1.0, min=0.0001, max=10000.0,
    )

    rotation: EnumProperty(
        name="",
        description="Rotate all imported data around the Z axis",
        items=EXPORT_ROTATION_ITEMS,
        default='NONE',
    )
    def draw_transforms(self, layout):
        layout.label(text="Transforms:")      
        box = layout.box()
        col = box.column()
        col.label(text="Scale Mode:")
        col.prop(self, "scale_mode")
        if self.scale_mode == 'FACTOR':
            col.prop(self, "scale_factor")
        col.label(text="Rotation:")
        col.prop(self, "rotation")

    def get_scale(self):
        """Return the actual multiplier to apply to imported positions."""
        if self.scale_mode == 'NONE':
            return 1.0
        if self.scale_mode == 'FACTOR':
            return self.scale_factor
        return MD5_SCALE_M_TO_IN
        
    def get_rotation(self):
        return self.rotation

# ─────────────────────────────────────────────────────────────────────
#  GEOMETRY  (all in game-space, un-scaled)
# ─────────────────────────────────────────────────────────────────────

def intersect_three_planes(n0, d0, n1, d1, n2, d2):
    """Unique intersection point of three planes, or None if degenerate."""
    denom = n0.dot(n1.cross(n2))
    if abs(denom) < PLANE_EPS:
        return None
    return (d0 * n1.cross(n2) + d1 * n2.cross(n0) + d2 * n0.cross(n1)) / denom


def point_inside_brush(pt, planes):
    """True when pt is on the INSIDE of all planes.

    Planes stored as (outward_normal n, effective_d) where effective_d = -d_raw.
    The plane equation is n · pt = effective_d.
    Inside (behind outward normal): n · pt < effective_d
    Outside: n · pt - effective_d > ON_PLANE_EPS
    """
    for n, d in planes:
        if n.dot(pt) - d > ON_PLANE_EPS:
            return False
    return True


def brush_vertices(planes):
    """All vertices of the brush (game-space, de-duplicated)."""
    found = []
    c = len(planes)
    for i in range(c - 2):
        for j in range(i + 1, c - 1):
            for k in range(j + 1, c):
                pt = intersect_three_planes(
                    planes[i][0], planes[i][1],
                    planes[j][0], planes[j][1],
                    planes[k][0], planes[k][1],
                )
                if pt is None:
                    continue
                if point_inside_brush(pt, planes):
                    found.append(pt)
    # de-duplicate
    unique = []
    for v in found:
        for u in unique:
            if (v - u).length_squared < DEDUP_EPS * DEDUP_EPS:
                break
        else:
            unique.append(v)
    return unique


def sort_face_verts(verts, normal):
    """CCW sort of coplanar verts viewed from the normal direction."""
    if len(verts) < 3:
        return verts
    center = sum(verts, Vector()) / len(verts)
    ref = (verts[0] - center)
    if ref.length_squared < 1e-8:
        ref = (verts[1] - center)
    ref.normalize()
    perp = normal.cross(ref)
    if perp.length_squared < 1e-8:
        # ref parallel to normal — pick an orthogonal fallback
        ref = Vector((1, 0, 0)) if abs(normal.x) < 0.9 else Vector((0, 1, 0))
        ref = (ref - ref.dot(normal) * normal).normalized()
        perp = normal.cross(ref).normalized()
    else:
        perp.normalize()

    def key(v):
        d = v - center
        return math.atan2(d.dot(perp), d.dot(ref))

    return sorted(verts, key=key)


# ─────────────────────────────────────────────────────────────────────
#  COORDINATE CONVERSION
# ─────────────────────────────────────────────────────────────────────

def to_bl(v, scale):
    """
    Convert idTech4 game coordinates to Blender world coordinates.

    The brushDef3 plane equation is n·pt = -d (the stored d has opposite
    sign to the actual plane distance). This convention, combined with
    simple axis-preserving scale, produces correct world-space positions
    consistent between brush geometry and entity origins.

        Blender X = idTech4.X * scale
        Blender Y = idTech4.Y * scale
        Blender Z = idTech4.Z * scale
    """
    return Vector((v.x * scale, v.y * scale, v.z * scale))


# ─────────────────────────────────────────────────────────────────────
#  UV
# ─────────────────────────────────────────────────────────────────────

# World-space vectors used for face axis derivation.
def _face_axes(normal):
    """
    Return (s_axis, t_axis) for the given outward face normal.

    This is a direct, verified port of idTech4's own
    idMapBrushSide::ComputeAxisBase (idlib/MapFile.cpp) — a spherical-
    coordinate construction, NOT a simple "cross product with a fixed up
    vector". That distinction matters: a naive UP×n / FWD×n cross-product
    approach gives the WRONG sign for ceiling-like faces (n ≈ (0,0,-1))
    compared to this real formula, confirmed by direct comparison against
    the Doom 3 GPL source.
    """
    nx = 0.0 if abs(normal.x) < 1e-6 else normal.x
    ny = 0.0 if abs(normal.y) < 1e-6 else normal.y
    nz = 0.0 if abs(normal.z) < 1e-6 else normal.z

    rot_y = -math.atan2(nz, math.sqrt(ny * ny + nx * nx))
    rot_z = math.atan2(ny, nx)

    s = Vector((-math.sin(rot_z), math.cos(rot_z), 0.0))
    t = Vector((
        -math.sin(rot_y) * math.cos(rot_z),
        -math.sin(rot_y) * math.sin(rot_z),
        -math.cos(rot_y),
    ))
    return s, t


def uv_brushdef3(pos, row0, row1, normal, origin=None):
    """
    idTech4 brushDef3 UV mapping — verified against
    idMapBrushSide::GetTextureVectors (idlib/MapFile.cpp).

    The 2×3 matrix stores (s_comp, t_comp, offset) where s and t are
    dot-products of the WORLD-SPACE vertex position with the face's
    tangent axes (see _face_axes):

        u = row0[0]*(worldpos·S) + row0[1]*(worldpos·T) + row0[2]
        v = 1.0 - (row1[0]*(worldpos·S) + row1[1]*(worldpos·T) + row1[2])

    *pos* is the brush's raw (as-parsed) vertex position; for brushes
    belonging to a non-worldspawn entity, this is local to that entity's
    "origin" — not yet the world-space position (the engine's own
    GetTextureVectors always operates on origin-adjusted geometry, since
    it runs after the entity's brush planes have origin baked in). Pass
    that entity's origin_vec as *origin* so it gets added here too; pass
    None (default) when this brush's position already IS world-space
    (worldspawn, or no origin key).
    """
    s_axis, t_axis = _face_axes(normal)
    world_pos = pos if origin is None else (pos + origin)
    s = world_pos.dot(s_axis)
    t = world_pos.dot(t_axis)
    u = row0[0]*s + row0[1]*t + row0[2]
    v = row1[0]*s + row1[1]*t + row1[2]
    return u, 1.0 - v


# Quake / idTech3 box-projection axis tables
_UP = [
    Vector(( 0,  0, -1)), Vector(( 0,  0,  1)),
    Vector(( 1,  0,  0)), Vector((-1,  0,  0)),
    Vector(( 0,  1,  0)), Vector(( 0, -1,  0)),
]
_RIGHT = [
    Vector((1, 0, 0)), Vector((1, 0, 0)),
    Vector((0, 1, 0)), Vector((0, 1, 0)),
    Vector((1, 0, 0)), Vector((1, 0, 0)),
]


def _quake_axes(normal):
    best_i = max(range(len(_UP)), key=lambda i: abs(normal.dot(_UP[i])))
    s = _RIGHT[best_i].copy()
    t = normal.cross(s).normalized()
    return s, t


def uv_standard(pos, normal, x_off, y_off, rot_deg, x_scale, y_scale):
    """Quake/idTech4 three-point standard UV. (V not flipped — Quake V is already Blender-compatible for world-projected UVs.)"""
    s, t = _quake_axes(normal)
    if abs(rot_deg) > 0.001:
        a = math.radians(rot_deg)
        ca, sa = math.cos(a), math.sin(a)
        s, t = s * ca + t * sa, s * (-sa) + t * ca
    xs = x_scale if abs(x_scale) > 1e-9 else 1.0
    ys = y_scale if abs(y_scale) > 1e-9 else 1.0
    u = pos.dot(s) / xs + x_off
    v = pos.dot(t) / ys + y_off
    return u, v


def uv_valve220(pos, uaxis, ushift, uscale, vaxis, vshift, vscale):
    """Valve 220 explicit-axis UV. (V not flipped — explicit axes already produce Blender-compatible orientation.)"""
    us = uscale if abs(uscale) > 1e-9 else 1.0
    vs = vscale if abs(vscale) > 1e-9 else 1.0
    u = (pos.dot(uaxis) + ushift) / us
    v = (pos.dot(vaxis) + vshift) / vs
    return u, v


# ─────────────────────────────────────────────────────────────────────
#  MAP PARSER
# ─────────────────────────────────────────────────────────────────────

class MapFace:
    """One brush face. Stores a plane (normal+d, outward convention) and UV."""
    __slots__ = (
        # Plane — outward normal, n·p = d for points on the plane
        'normal', 'd',
        # Material
        'material',
        # UV format: 'brushdef3' | 'standard' | 'valve220'
        'uv_fmt',
        # brushDef3 UV matrix rows
        'uv_row0', 'uv_row1',
        # standard UV
        'x_off', 'y_off', 'rot', 'x_scale', 'y_scale',
        # valve220 UV
        'uaxis', 'ushift', 'uscale', 'vaxis', 'vshift', 'vscale',
    )
    def __init__(self):
        self.normal  = None
        self.d       = 0.0
        self.material= 'unknown'
        self.uv_fmt  = 'standard'
        self.uv_row0 = (0.015625, 0, 0)
        self.uv_row1 = (0, 0.015625, 0)
        self.x_off   = self.y_off = self.rot = 0.0
        self.x_scale = self.y_scale = 1.0
        self.uaxis   = self.vaxis = None
        self.ushift  = self.vshift = 0.0
        self.uscale  = self.vscale = 1.0


class MapBrush:
    def __init__(self):
        self.faces = []
        # idMapBrush::Parse reads key/value pairs before the sides ("only
        # used in editor"). None exist in the shipped corpus; captured rather
        # than discarded so a mod that uses them is not silently stripped.
        self.epairs = {}


class MapPatch:
    """A Bezier patch surface (patchDef2 or patchDef3)."""
    __slots__ = ('material', 'rows', 'cols', 'ctrl', 'subdiv_x', 'subdiv_y',
                 'epairs')
    def __init__(self):
        # idMapPatch::Parse reads key/value pairs after the control points,
        # on Version 2 and below (Quake 4's Version 3 has no epair tail).
        self.epairs = {}
        self.material = 'unknown'
        self.rows     = 0     # number of control-point rows
        self.cols     = 0     # number of control-point columns
        self.ctrl     = []    # [row][col] = (x, y, z, u, v)  in game space
        self.subdiv_x = 0     # explicit horizontal subdivisions (patchDef3)
        self.subdiv_y = 0     # explicit vertical subdivisions   (patchDef3)


class MapEntity:
    def __init__(self):
        self.keys    = {}
        self.brushes = []
        self.patches = []


# ── compiled regex pieces ─────────────────────────────────────────────
#
# The token parser below carries its own master alternation (_TOK) and its
# two fused productions. What is left here is shared with it: the number
# pattern every one of them is built from, and the control-point pattern the
# fused patch row re-scans.
#
# The line-oriented face and key/value regexes that used to live here
# (_BD3, _3PTS, _STD, _VALVE, _KV, _PATCH_HDR) are gone with the reader that
# used them - see BEGIN MAP TOKEN PARSER for what replaced them and why.

# Deliberately accepts a trailing-dot literal ("5."), which idLexer does and
# the old pattern did not. Nothing in the corpus writes one.
_NUM = r'[-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?'

# One patch control point: ( x y z u v )
_CTRL_PT = re.compile(
    r'\(\s*' +
    r'(' + _NUM + r')' + r'\s+' +
    r'(' + _NUM + r')' + r'\s+' +
    r'(' + _NUM + r')' + r'\s+' +
    r'(' + _NUM + r')' + r'\s+' +
    r'(' + _NUM + r')' +
    r'\s*\)'
)


def _versioned_material(name, map_version):
    """Apply OLD_MAP_VERSION's implicit "textures/" prefix.

    idMapBrush::Parse and idMapPatch::Parse both do exactly this:

        if ( version < 2.0f ) { material = "textures/" + token; }
        else                  { material = token; }

    (Raven spells the same test `version < 2` on an int.) Nothing in the
    shipped corpus is Version 1, so this never fires there - it exists for
    converted and hand-authored maps, whose materials otherwise resolve one
    directory short with no diagnostic.
    """
    if map_version >= 2.0 or not name:
        return name
    return 'textures/' + name


# ─────────────────────────────────────────────────────────────────────
#  BEGIN MAP TOKEN PARSER
#
#  A recursive-descent parser over a token stream, structured one-to-one
#  with idMapFile.cpp so any future divergence is a diff against the
#  engine rather than a diff against a regex.
#
#  This replaced a line-oriented regex reader. That reader handled every
#  shipped map in the corpus correctly - all 204 of them, 1.69M
#  primitives - but it silently mis-parsed several constructs the engine
#  accepts and that fan missions, prefabs, .reg region files and
#  converters do emit:
#
#      brushDef (3-point + 2x3 texture matrix)   material became "("
#      an entity written on one line             the whole file read as empty
#      material and patch header on one line     patch read as cols=8 rows=0,
#                                                and brace depth desynced for
#                                                the rest of the file
#      a multi-line quoted value                 key lost, entity ended early
#
#  none of which raised, warned, or appeared in the import report.
#
#  PERFORMANCE. A naive token stream is ~3x slower than the line reader
#  in Python: the largest map in the corpus (darkmod/fms/iris, 55 MB)
#  tokenizes to 8.0M tokens, and just tokenizing it costs 5.5s against
#  the old reader's 2.2s for the whole parse. ~92% of those tokens live
#  in exactly two productions - brushDef3 face lines and patch
#  control-point rows - so both have a FUSED fast path that matches the
#  canonical single-line spelling in one regex step and falls back to
#  token-by-token when it does not match. Measured over the whole
#  corpus that costs +26% (70.2s -> 88.4s), with the fused path taking
#  100% of 6.46M faces and 3.04M control rows, and iris.map parses in
#  2.4s. If the fast-path hit rate ever drops materially, the fused
#  regexes have stopped matching real content and that number is void.
# ─────────────────────────────────────────────────────────────────────

# One master alternation, so the scanning loop stays inside the C regex
# engine. Mirrors idLexer as idMapFile::Parse configures it -
# LEXFL_NOSTRINGCONCAT | LEXFL_NOSTRINGESCAPECHARS | LEXFL_ALLOWPATHNAMES:
# no string concatenation, a backslash is a literal character, and path
# characters are legal inside a bare token. Strings do not span newlines
# (the engine errors on that), and the trailing catch-all guarantees the
# scanner always consumes at least one character so no input can loop.
_TOK = re.compile(
    r'(?P<ws>(?:[ \t\r\n\f\v]+|//[^\n]*|/\*.*?\*/)+)'
    r'|(?P<str>"[^"\n]*")'
    r'|(?P<num>' + _NUM + r')'
    r'|(?P<pnc>[(){}\[\],])'
    r'|(?P<name>[^\s(){}\[\],"]+)'
    r'|(?P<other>.)', re.S)

# Fused: one whole brushDef3 face line.
#   ( nx ny nz d ) ( ( su sv sw ) ( tu tv tw ) ) "material" [0 0 0] [// ...]
# The trailing numbers are the legacy Q2 flags, present on Version 2 maps
# (Doom 3, Prey, The Dark Mod) and dropped on Version 3 (Quake 4); both
# spellings are accepted here because the engine ignores them either way.
_FUSED_FACE = re.compile(
    r'[ \t]*\(\s*(' + _NUM + r')\s+(' + _NUM + r')\s+(' + _NUM + r')\s+(' + _NUM + r')\s*\)'
    r'\s*\(\s*\(\s*(' + _NUM + r')\s+(' + _NUM + r')\s+(' + _NUM + r')\s*\)'
    r'\s*\(\s*(' + _NUM + r')\s+(' + _NUM + r')\s+(' + _NUM + r')\s*\)\s*\)'
    r'\s*"([^"\n]*)"[ \t]*(?:' + _NUM + r'[ \t]*){0,3}(?://[^\n]*)?(?=\n|\Z)')

# Fused: one whole patch control-point column, i.e. an outer ( ) holding
# `rows` ( x y z u v ) groups. Columns that a tool wrapped across several
# lines simply miss this and take the token path.
_FUSED_CTRL_ROW = re.compile(
    r'[ \t]*\((?:\s*\(\s*' + _NUM + r'\s+' + _NUM + r'\s+' + _NUM +
    r'\s+' + _NUM + r'\s+' + _NUM + r'\s*\))+\s*\)[ \t]*(?://[^\n]*)?(?=\n|\Z)')


class MapSyntaxError(Exception):
    """A .map file that does not parse, with the line it gave up on."""

    def __init__(self, message, line):
        Exception.__init__(self, "line %d: %s" % (line, message))
        self.line = line


class MapParseDiagnostics(object):
    """What the parser knowingly discarded, so the import report can say so.

    The engine warns about degenerate brush planes ("brush %d has
    degenerate plane equations"); this importer used to drop them without
    a word, leaving a brush with fewer sides than were written and
    possibly no longer closed. Exactly three exist in the whole corpus,
    all in doom3 pdas.map, all written as ( 0 0 0 0 ).
    """

    __slots__ = ('degenerate_planes', 'unknown_primitives', 'notes')

    def __init__(self):
        self.degenerate_planes = 0
        self.unknown_primitives = []
        self.notes = []

    def any(self):
        return bool(self.degenerate_planes or self.unknown_primitives
                    or self.notes)


class _TokenStream(object):
    """Token cursor over the whole text, with position-addressable fast paths.

    Holds the source string so a fused production can be matched straight
    at the current token's offset and the cursor jumped past it.
    """

    __slots__ = ('s', 'pos', 'tok')

    def __init__(self, text):
        self.s = text
        self.pos = 0
        self.tok = None
        self._advance()

    def _advance(self):
        s, n = self.s, len(self.s)
        pos = self.pos
        while pos < n:
            mo = _TOK.match(s, pos)
            if mo is None:              # cannot happen: 'other' matches any char
                break
            if mo.lastgroup == 'ws':
                pos = mo.end()
                continue
            self.tok = (mo.lastgroup, mo.group(), mo.start())
            self.pos = mo.end()
            return
        self.tok = None
        self.pos = n

    def peek(self):
        return self.tok

    def next(self):
        t = self.tok
        self._advance()
        return t

    def line_of(self, t):
        return self.s.count('\n', 0, t[2]) + 1 if t else self.s.count('\n') + 1

    def error(self, message, tok=None):
        t = self.tok if tok is None else tok
        return MapSyntaxError(message, self.line_of(t))

    def expect(self, val):
        t = self.tok
        if t is None:
            raise self.error("expected %r, got end of file" % val)
        if t[1] != val:
            raise self.error("expected %r, got %r" % (val, t[1]), t)
        self._advance()
        return t

    def check(self, val):
        """Consume and return True if the next token is exactly *val*."""
        if self.tok is not None and self.tok[1] == val:
            self._advance()
            return True
        return False

    def number(self):
        t = self.tok
        if t is None or t[0] != 'num':
            raise self.error("expected a number, got %r"
                             % (t[1] if t else 'end of file'), t)
        self._advance()
        return float(t[1])

    def matrix1d(self, count):
        """idLexer::Parse1DMatrix — ( a b c ... )"""
        self.expect('(')
        vals = [self.number() for _ in range(count)]
        self.expect(')')
        return vals

    def text_of(self, t):
        """A token's value with quotes stripped, whatever kind it is."""
        return t[1][1:-1] if t[0] == 'str' else t[1]

    def same_line(self, a, b):
        return b is not None and '\n' not in self.s[a[2]:b[2]]

    def try_fused(self, rx):
        """Match *rx* at the current token's start; on success jump past it."""
        t = self.tok
        if t is None:
            return None
        mo = rx.match(self.s, t[2])
        if mo is None:
            return None
        self.pos = mo.end()
        self._advance()
        return mo


def _face_from_plane(nx, ny, nz, d_raw, row0, row1, material):
    """A brushDef3 side: normal+distance plane plus the 2x3 texture matrix.

    Returns None for a degenerate plane, which the caller counts. The plane
    is normalised first because the engine's own FindFloatPlane does, and
    idMapBrushSide stores the plane exactly as written otherwise.
    """
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if not length >= PLANE_EPS:      # NaN-safe: `not >=` also rejects nan
        return None
    face = MapFace()
    # brushDef3 writes the plane as n·p + d = 0, i.e. n·p = -d, with n the
    # OUTWARD normal. Store effective_d = -d so the inside test is the
    # ordinary n·p < d, matching the outward convention build_brush_mesh
    # expects.
    face.normal = Vector((nx / length, ny / length, nz / length))
    face.d = -(d_raw / length)
    face.material = material
    face.uv_fmt = 'brushdef3'
    face.uv_row0 = row0
    face.uv_row1 = row1
    return face


def _face_from_points(p0, p1, p2):
    """The plane half of a three-point side. Returns None if degenerate."""
    n = (p1 - p0).cross(p2 - p0)
    length = n.length
    if not length >= PLANE_EPS:
        return None
    face = MapFace()
    face.normal = n / length
    face.d = face.normal.dot(p0)
    return face


def parse_map_file(filepath, diagnostics=None):
    """
    Parse a .map file and return list[MapEntity].

    Implements idMapFile::Parse / idMapEntity::Parse / idMapBrush::Parse /
    idMapPatch::Parse, covering every primitive spelling the engine's
    dispatch accepts:

      - brushDef3 / brushDef2   normal+d plane + 2x3 UV matrix
      - brushDef                three points + 2x3 UV matrix
      - a keyword-less brush    idMapBrush::ParseQ3, three points +
                                shift/rotate/scale, and Valve 220's explicit
                                UV axes (not an idTech4 format, but this
                                importer accepts Quake-family maps too)
      - patchDef2 / patchDef3   Bezier patches

    Keyword dispatch is case-insensitive and prefix-based on "brush"/"patch",
    exactly as idMapEntity::Parse's Icmpn(token, "brush", 5) is - so
    brushDef2, brushdef3 and BRUSHDEF3 all work, and an unrecognised keyword
    falls through to ParseQ3 rather than being dropped.

    The leading "Version <n>" line is honoured: below 2 (OLD_MAP_VERSION)
    every brush-side and patch material gets an implicit "textures/" prefix,
    per idMapBrush::Parse and idMapPatch::Parse. Measured across the corpus,
    Doom 3 / Prey / The Dark Mod are Version 2 and Quake 4 is Version 3;
    nothing shipped is Version 1, so that branch exists for converted and
    hand-authored maps, whose materials otherwise resolve one directory
    short with no diagnostic.

    Pass a MapParseDiagnostics to *diagnostics* to receive counts of what was
    knowingly discarded.
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
        text = fh.read()
    return parse_map_text(text, diagnostics=diagnostics)


def parse_map_text(text, diagnostics=None):
    """parse_map_file's body, on a string. Split out so tests and callers
    holding map text (a .reg region, an editor buffer) need no temp file."""
    # NOTE: no `text += '\n'` here, deliberately. The fused productions
    # anchor on `(?=\n|\Z)` rather than `(?=\n)` precisely so the source
    # string is never copied: the largest map in the corpus is 55 MB, and
    # appending one character to it doubles the peak footprint for the whole
    # parse. That mattered - under 32-bit CPython the copy was enough to push
    # a darkmod corpus run into transient, position-varying failures that
    # vanished on retry, while 64-bit (which is what Blender ships) was fine
    # either way.
    diag = diagnostics if diagnostics is not None else MapParseDiagnostics()
    st = _TokenStream(text)

    version = 1.0                       # OLD_MAP_VERSION, per idMapFile::Parse
    t = st.peek()
    if t is not None and t[1].lower() == 'version':
        st.next()
        nxt = st.peek()
        if nxt is not None and nxt[0] == 'num':
            version = float(st.next()[1])

    entities = []
    while st.peek() is not None:
        entities.append(_parse_entity(st, version, diag))
    return entities


def _parse_entity(st, version, diag):
    ent = MapEntity()
    st.expect('{')
    while True:
        t = st.peek()
        if t is None:
            raise st.error("end of file inside an entity")
        if t[1] == '}':
            st.next()
            return ent
        if t[1] == '{':
            st.next()
            _parse_primitive(st, ent, version, diag)
            continue
        # A key/value pair. idMapEntity::Parse takes ANY token as the key and
        # the next token ON THE SAME LINE as the value, then calls
        # StripTrailingWhitespace() on both - trailing only, so a leading
        # space survives, which the corpus depends on (The Dark Mod ships 62
        # keys spelled " is_mantleable" and the engine leaves them alone).
        # Real content always quotes both; accepting bare tokens matches the
        # engine rather than being deliberately lenient.
        key_tok = st.next()
        val_tok = st.peek()
        if val_tok is not None and st.same_line(key_tok, val_tok) \
                and val_tok[1] not in ('{', '}'):
            st.next()
            value = st.text_of(val_tok)
        else:
            value = ''
        ent.keys[st.text_of(key_tok).rstrip()] = value.rstrip()


def _parse_primitive(st, ent, version, diag):
    """The '{' is already consumed. idMapEntity::Parse's dispatch."""
    kw = st.peek()
    if kw is None:
        raise st.error("end of file after '{'")
    low = kw[1].lower()
    if low[:5] == 'brush':
        st.next()
        brush = _parse_brush(st, version, diag,
                             new_format=low in ('brushdef2', 'brushdef3'))
        st.expect('}')                  # the primitive's own closing brace
        if brush.faces:
            ent.brushes.append(brush)
    elif low[:5] == 'patch':
        st.next()
        patch = _parse_patch(st, version, diag, pd3=(low == 'patchdef3'))
        if patch is not None:
            ent.patches.append(patch)
    else:
        # No keyword: a Quake 3 / Quake 2 era brush. The engine unreads the
        # token and calls ParseQ3; so do we.
        #
        # NOTE the brace asymmetry. brushDef*/patchDef* are wrapped in a
        # SECOND brace - idMapBrush::Parse and idMapPatch::Parse each open
        # with ExpectTokenString("{") - so those close twice. A ParseQ3 brush
        # has no inner brace: its sides sit directly inside the primitive's
        # own { }, and ParseQ3 consumes the single closing one itself.
        brush = _parse_q3_brush(st, version, diag)
        if brush.faces:
            ent.brushes.append(brush)


def _parse_brush(st, version, diag, new_format):
    """idMapBrush::Parse. Consumes the inner '{' ... '}'; the caller consumes
    the primitive's outer '}'."""
    brush = MapBrush()
    st.expect('{')
    while True:
        t = st.peek()
        if t is None:
            raise st.error("end of file inside a brush")
        if t[1] == '}':
            st.next()
            return brush
        if t[1] != '(':
            # "here we may have to jump over brush epairs ( only used in
            # editor )" - a quoted key then its value. None exist anywhere in
            # the shipped corpus, but the grammar has them and the old reader
            # dropped them along with any hope of reporting them.
            key_tok = st.next()
            val_tok = st.peek()
            if val_tok is not None and st.same_line(key_tok, val_tok):
                st.next()
                brush.epairs[st.text_of(key_tok)] = st.text_of(val_tok)
            continue

        if new_format:
            mo = st.try_fused(_FUSED_FACE)
            if mo is not None:
                g = mo.groups()
                face = _face_from_plane(
                    float(g[0]), float(g[1]), float(g[2]), float(g[3]),
                    (float(g[4]), float(g[5]), float(g[6])),
                    (float(g[7]), float(g[8]), float(g[9])),
                    _versioned_material(g[10], version))
                if face is None:
                    diag.degenerate_planes += 1
                else:
                    brush.faces.append(face)
                continue
            plane = st.matrix1d(4)
            points = None
        else:
            plane = None
            points = [Vector(st.matrix1d(3)) for _ in range(3)]

        # The 2x3 texture matrix, shared by brushDef and brushDef3.
        st.expect('(')
        row0 = tuple(st.matrix1d(3))
        row1 = tuple(st.matrix1d(3))
        st.expect(')')

        mat_tok = st.next()
        if mat_tok is None:
            raise st.error("end of file reading a brush side material")
        material = _versioned_material(st.text_of(mat_tok), version)

        if plane is not None:
            face = _face_from_plane(plane[0], plane[1], plane[2], plane[3],
                                    row0, row1, material)
        else:
            face = _face_from_points(points[0], points[1], points[2])
            if face is not None:
                # brushDef carries the same 2x3 matrix as brushDef3; the old
                # reader could not see it at all and fell back to Quake
                # shift/scale defaults, with the material set to the literal
                # string "(".
                face.material = material
                face.uv_fmt = 'brushdef3'
                face.uv_row0 = row0
                face.uv_row1 = row1
        if face is None:
            diag.degenerate_planes += 1
        else:
            brush.faces.append(face)

        # "Q2 allowed override of default flags and values, but we don't any
        # more" - up to three tokens, on Version 2 and below. Quake 4's
        # Version 3 omits them. Consume whatever is left on the line either
        # way; the engine ignores the values.
        _skip_rest_of_line(st, mat_tok, limit=3)


def _parse_q3_brush(st, version, diag):
    """idMapBrush::ParseQ3, plus Valve 220's explicit texture axes.

    Valve 220 is not an idTech4 format and appears nowhere in MapFile.cpp or
    in the corpus; it is supported because this importer also accepts
    Quake-family maps, and is tried first because its '[' is unambiguous.
    """
    brush = MapBrush()
    while True:
        t = st.peek()
        if t is None:
            raise st.error("end of file inside a brush")
        if t[1] == '}':
            st.next()
            return brush
        points = [Vector(st.matrix1d(3)) for _ in range(3)]
        mat_tok = st.next()
        if mat_tok is None:
            raise st.error("end of file reading a brush side material")
        face = _face_from_points(points[0], points[1], points[2])
        material = _versioned_material(st.text_of(mat_tok), version)

        if st.peek() is not None and st.peek()[1] == '[':
            # Valve 220: [ ux uy uz ushift ] rot [ vx vy vz vshift ] us vs
            st.next()
            uaxis = Vector((st.number(), st.number(), st.number()))
            ushift = st.number()
            st.expect(']')
            st.number()                             # rotation, unused
            st.expect('[')
            vaxis = Vector((st.number(), st.number(), st.number()))
            vshift = st.number()
            st.expect(']')
            uscale = st.number()
            vscale = st.number()
            if face is not None:
                face.uv_fmt = 'valve220'
                face.uaxis, face.ushift = uaxis, ushift
                face.vaxis, face.vshift = vaxis, vshift
                face.uscale = uscale or 1.0
                face.vscale = vscale or 1.0
        else:
            # shift shift rotate scale scale
            nums = []
            while len(nums) < 5 and st.peek() is not None \
                    and st.peek()[0] == 'num' and st.same_line(mat_tok, st.peek()):
                nums.append(st.number())
            if face is not None and len(nums) >= 5:
                face.uv_fmt = 'standard'
                face.x_off, face.y_off, face.rot = nums[0], nums[1], nums[2]
                face.x_scale = nums[3] or 1.0
                face.y_scale = nums[4] or 1.0

        if face is None:
            diag.degenerate_planes += 1
        else:
            face.material = material
            brush.faces.append(face)
        _skip_rest_of_line(st, mat_tok)


def _skip_rest_of_line(st, anchor_tok, limit=None):
    """Drop the trailing content flags on a face line.

    Bounded by *limit* where the engine reads a fixed number, so a malformed
    line cannot silently eat the next face.
    """
    dropped = 0
    while st.peek() is not None and st.same_line(anchor_tok, st.peek()):
        nxt = st.peek()
        if nxt[1] in ('(', '{', '}'):
            break
        if limit is not None and dropped >= limit:
            break
        st.next()
        dropped += 1


def _parse_patch(st, version, diag, pd3):
    """idMapPatch::Parse. Consumes the inner '{' ... '}' AND the primitive's
    outer '}' - the engine's patch path swallows both."""
    st.expect('{')
    mat_tok = st.next()
    if mat_tok is None:
        raise st.error("end of file reading a patch material")
    patch = MapPatch()
    patch.material = _versioned_material(st.text_of(mat_tok), version)

    # patchDef3 has 7 info floats (w h subdivH subdivV + 3 unused),
    # patchDef2 has 5 (w h + 3 unused).
    info = st.matrix1d(7 if pd3 else 5)
    cols = int(info[0])
    rows = int(info[1])
    if cols < 0 or rows < 0:
        raise st.error("patch has a negative size (%d x %d)" % (cols, rows))
    patch.cols = cols
    patch.rows = rows
    if pd3:
        patch.subdiv_x = int(info[2])
        patch.subdiv_y = int(info[3])

    # idTech4 writes patch control data COLUMN by column, not row by row -
    # idMapPatch::Parse even comments "these were written out in the wrong
    # order, IMHO". Each parenthesised group is one COLUMN of `rows` points,
    # and there are `cols` such groups.
    patch.ctrl = [[None] * cols for _ in range(rows)]
    st.expect('(')
    for j in range(cols):
        mo = st.try_fused(_FUSED_CTRL_ROW)
        if mo is not None:
            pts = _CTRL_PT.findall(mo.group())
            if len(pts) != rows:
                raise st.error("patch column %d has %d control points, "
                               "expected %d" % (j, len(pts), rows))
            for i, p in enumerate(pts):
                patch.ctrl[i][j] = (float(p[0]), float(p[1]), float(p[2]),
                                    float(p[3]), float(p[4]))
            continue
        st.expect('(')
        for i in range(rows):
            v = st.matrix1d(5)
            patch.ctrl[i][j] = (v[0], v[1], v[2], v[3], v[4])
        st.expect(')')
    st.expect(')')

    # Trailing epairs, Version 2 and below only: Quake 4's Version 3 patch
    # closes straight after the control points. None exist in the corpus.
    while True:
        t = st.peek()
        if t is None:
            raise st.error("end of file inside a patch")
        if t[1] == '}':
            st.next()
            break
        key_tok = st.next()
        val_tok = st.peek()
        if val_tok is not None and st.same_line(key_tok, val_tok) \
                and val_tok[1] != '}':
            st.next()
            patch.epairs[st.text_of(key_tok)] = st.text_of(val_tok)
    st.expect('}')
    return patch

# ─────────────────────────────────────────────────────────────────────
#  END MAP TOKEN PARSER
# ─────────────────────────────────────────────────────────────────────


def build_brush_mesh(brush, scale, origin_for_uv=None):
    """
    Returns (bpy.types.Mesh, [material_name, ...]) or (None, []).

    All faces in *brush* already have outward normals and a *d* value
    following the convention  n·p = d  for points on the plane.

    Brush vertices are stored at RAW (as-parsed) coordinates — for
    brushes on a non-worldspawn entity these are local to that entity's
    "origin", not yet world-space (see import_map). Parenting to entity
    empties is handled in import_map by setting matrix_parent_inverse so
    the parent acts as a pure grouping anchor with no positional
    side-effect on the brush geometry.

    *origin_for_uv*: pass the entity's origin_vec here when this brush's
    raw coordinates are local to it (i.e. whenever the brush's positional
    matrix_parent_inverse is being set to Identity so origin reaches the
    mesh) — brushDef3 UV must be computed from the WORLD-SPACE position
    (see uv_brushdef3), and since the mesh's own vertex data stays raw
    (origin is applied only via the Object's transform, never baked into
    vertex coordinates), the UV math needs the offset added explicitly.
    Leave as None when the brush's raw coordinates already ARE world-space
    (worldspawn, or no origin key).
    """
    # Collect valid planes
    planes = [(f.normal, f.d) for f in brush.faces if f.normal is not None]
    if len(planes) < 4:
        return None, []

    # Enumerate all vertices in absolute game space
    all_verts = brush_vertices(planes)
    if len(all_verts) < 4:
        return None, []

    bm     = bmesh.new()
    uv_lay = bm.loops.layers.uv.new("UVMap")
    mat_idx = {}

    # Vertex pool keyed on game coords -> BMVert
    pool = {}

    def get_bv(gv):
        key = (round(gv.x, 2), round(gv.y, 2), round(gv.z, 2))
        if key not in pool:
            pool[key] = bm.verts.new(to_bl(gv, scale))
        return pool[key]

    for face in brush.faces:
        if face.normal is None:
            continue
        normal = face.normal
        d      = face.d

        # Vertices on this face's plane: n·pt = d  (where d = -d_raw)
        on_face = [v for v in all_verts
                   if abs(normal.dot(v) - d) <= ON_PLANE_EPS]
        if len(on_face) < 3:
            continue

        # sort_face_verts sorts CCW viewed from the normal direction.
        # normal is the outward normal, so pass it directly.
        on_face = sort_face_verts(on_face, normal)

        # Build BMVert list, removing duplicates while preserving order.
        # Keep a parallel game-vert list so UV coords stay in sync with loops.
        seen = []
        bverts = []
        gverts = []   # game-space verts in same order as bverts / face loops
        for gv in on_face:
            key = (round(gv.x, 2), round(gv.y, 2), round(gv.z, 2))
            if key not in seen:
                seen.append(key)
                bverts.append(get_bv(gv))
                gverts.append(gv)
        if len(bverts) < 3:
            continue

        try:
            bm_face = bm.faces.new(bverts)
        except ValueError:
            continue  # duplicate face

        # Material slot
        mn = face.material
        if mn not in mat_idx:
            mat_idx[mn] = len(mat_idx)
        bm_face.material_index = mat_idx[mn]

        # UVs — loop order matches bverts/gverts order exactly.
        # Compute UVs using the solved vertex coordinates directly.
        # uv_brushdef3 handles the Z-axis sign convention internally.
        raw_uvs = []
        for gv in gverts:
            if face.uv_fmt == 'brushdef3':
                u, v = uv_brushdef3(gv, face.uv_row0, face.uv_row1, normal, origin=origin_for_uv)
            elif face.uv_fmt == 'valve220' and face.uaxis and face.vaxis:
                u, v = uv_valve220(gv,
                                   face.uaxis, face.ushift, face.uscale,
                                   face.vaxis, face.vshift, face.vscale)
            else:
                u, v = uv_standard(gv, normal,
                                   face.x_off, face.y_off, face.rot,
                                   face.x_scale, face.y_scale)
            raw_uvs.append((u, v))

        # Store UVs as-is (world-projected, may tile beyond 0-1).
        for loop, (u, v) in zip(bm_face.loops, raw_uvs):
            loop[uv_lay].uv = (u, v)

    if not bm.faces:
        bm.free()
        return None, []

    bm.normal_update()
    mesh = bpy.data.meshes.new("brush")
    bm.to_mesh(mesh)
    bm.free()
    mesh.validate(clean_customdata=False)
    mesh.update()
    return mesh, list(mat_idx.keys())


# ─────────────────────────────────────────────────────────────────────
#  BEZIER PATCH TESSELLATION
# ─────────────────────────────────────────────────────────────────────

def _bezier_curve(p0, p1, p2, steps):
    """
    Evaluate a quadratic Bezier curve with control points p0, p1, p2
    at (steps+1) equally-spaced parameter values [0..1].
    p0, p1, p2 are tuples/lists of the same length (3 for xyz, 5 for xyzuv).
    Returns a list of (steps+1) interpolated tuples.
    """
    pts = []
    for i in range(steps + 1):
        t  = i / steps
        t2 = t * t
        b0 = (1 - t) * (1 - t)
        b1 = 2 * (1 - t) * t
        b2 = t2
        pts.append(tuple(b0 * p0[k] + b1 * p1[k] + b2 * p2[k]
                         for k in range(len(p0))))
    return pts


def _linear_segment(p0, p2, steps):
    """Evaluate (steps+1) evenly-spaced points on the straight line from
    p0 to p2 (skipping p1 entirely). Used for control-point segments that
    carry no real curvature — see _segment_is_flat."""
    pts = []
    for i in range(steps + 1):
        t = i / steps
        pts.append(tuple(p0[k] + (p2[k] - p0[k]) * t for k in range(len(p0))))
    return pts


_FLAT_SEGMENT_TOLERANCE = 0.1  # game units


def _segment_is_flat(p0, p1, p2, tolerance=_FLAT_SEGMENT_TOLERANCE):
    """
    Check whether a quadratic segment's middle control point carries any
    real curvature, using only its XYZ position (not UV) — matching how
    idTech4's own idSurface_Patch::RemoveLinearColumnsRows decides which
    columns/rows to collapse.

    Many real .map patches (especially simple flat decals) have their
    "control" point set to the exact same position as one of its
    neighboring corners rather than the true midpoint, since the surface
    is meant to be flat, not curved. Bezier-sampling such a segment at
    uniform PARAMETER steps still produces very UNEVEN spacing in actual
    world/UV space (samples bunch up near the coincident corner), even
    though the surface itself is a straight line. Detecting this and
    falling back to plain linear interpolation avoids that artifact while
    leaving genuinely curved segments (arches, pipes, ...) untouched.

    Returns True if p1 lies within `tolerance` world units of the
    straight line from p0 to p2.
    """
    ax, ay, az = p0[0], p0[1], p0[2]
    bx, by, bz = p2[0], p2[1], p2[2]
    px, py, pz = p1[0], p1[1], p1[2]
    dx, dy, dz = bx - ax, by - ay, bz - az
    seg_len_sq = dx * dx + dy * dy + dz * dz
    if seg_len_sq < 1e-9:
        # p0 and p2 coincide; flat only if p1 is there too.
        ex, ey, ez = px - ax, py - ay, pz - az
        return (ex * ex + ey * ey + ez * ez) < tolerance * tolerance
    t = ((px - ax) * dx + (py - ay) * dy + (pz - az) * dz) / seg_len_sq
    projx, projy, projz = ax + t * dx, ay + t * dy, az + t * dz
    ex, ey, ez = px - projx, py - projy, pz - projz
    return (ex * ex + ey * ey + ez * ez) < tolerance * tolerance


def _sample_segment(p0, p1, p2, steps):
    """Bezier-sample a quadratic segment, unless it's geometrically flat
    (see _segment_is_flat), in which case sample it linearly instead to
    avoid uneven spacing artifacts."""
    if _segment_is_flat(p0, p1, p2):
        return _linear_segment(p0, p2, steps)
    return _bezier_curve(p0, p1, p2, steps)


def _patch_is_entirely_flat(ctrl, cols, rows):
    """
    True if every quadratic segment in this patch's control grid — both
    directions — carries no real curvature (see _segment_is_flat). Real
    idTech4 only subdivides a patchDef2 (auto-subdivision) surface where
    it actually curves; a fully flat patch like a simple decal quad gets
    reduced to its base control mesh via idSurface_Patch::
    RemoveLinearColumnsRows before the engine ever draws it. Detecting
    that case here lets us skip subdividing it too, instead of producing
    a dense but redundant coplanar grid.
    """
    num_segs_u = (cols - 1) // 2
    num_segs_v = (rows - 1) // 2
    for r in range(rows):
        for seg in range(num_segs_u):
            ci = seg * 2
            if not _segment_is_flat(ctrl[r][ci], ctrl[r][ci + 1], ctrl[r][ci + 2]):
                return False
    for c in range(cols):
        for seg in range(num_segs_v):
            ri = seg * 2
            if not _segment_is_flat(ctrl[ri][c], ctrl[ri + 1][c], ctrl[ri + 2][c]):
                return False
    return True


def _tessellate_patch(ctrl, cols, rows, subdiv_x, subdiv_y, steps_default=8):
    """
    Tessellate a Bezier patch control grid into a regular mesh grid.

    ctrl     : list of `rows` lists, each with `cols` (x,y,z,u,v) tuples
    cols/rows: control-point grid dimensions (both must be odd ≥ 3)
    subdiv_x : explicit horizontal subdivisions per segment (0 = use default)
    subdiv_y : explicit vertical subdivisions per segment   (0 = use default)

    Returns (verts_grid, uvs_grid) where each is a 2-D list [row][col]
    of Vector(x,y,z) / (u,v) respectively, sized:
        out_rows = (rows-1)//2 * steps_v + 1
        out_cols = (cols-1)//2 * steps_u + 1
    """
    num_segs_u = (cols - 1) // 2   # number of cubic-bezier segments along U
    num_segs_v = (rows - 1) // 2   # number of cubic-bezier segments along V
    steps_u = subdiv_x if subdiv_x > 0 else steps_default
    steps_v = subdiv_y if subdiv_y > 0 else steps_default

    # patchDef2 (subdiv_x/y == 0, i.e. no explicit subdivision requested)
    # uses idTech4's error-based *auto* subdivision in-engine, which adds
    # detail only where the surface actually curves — a fully flat patch
    # (e.g. a simple decal quad) ends up as just its base control mesh,
    # not a dense grid. Match that here: skip subdividing when the whole
    # patch is flat. patchDef3's *explicit* subdivision count is a
    # deliberate mapper request and is always honored as-is, even on a
    # flat patch, matching idSurface_Patch::SubdivideExplicit.
    if subdiv_x == 0 and subdiv_y == 0 and _patch_is_entirely_flat(ctrl, cols, rows):
        steps_u = 1
        steps_v = 1

    # ── Step 1: tessellate each row of control points along U ─────────
    # For each control row we produce (num_segs_u * steps_u + 1) sample points.
    row_curves = []
    for r in range(rows):
        row_pts = []
        for seg in range(num_segs_u):
            ci = seg * 2
            p0 = ctrl[r][ci]
            p1 = ctrl[r][ci + 1]
            p2 = ctrl[r][ci + 2]
            seg_pts = _sample_segment(p0, p1, p2, steps_u)
            if seg == 0:
                row_pts.extend(seg_pts)
            else:
                row_pts.extend(seg_pts[1:])  # skip duplicate endpoint
        row_curves.append(row_pts)

    # ── Step 2: tessellate along V using the row-curve columns ────────
    # row_curves has `rows` entries, each with `out_cols` points.
    out_cols  = num_segs_u * steps_u + 1
    out_rows  = num_segs_v * steps_v + 1

    verts_grid = []
    uvs_grid   = []

    # ── Step 2: tessellate along V using the row-curve columns ────────
    # For each column index in the tessellated U-direction, sweep the
    # control column through V using num_segs_v quadratic Bezier segments.
    # Build the result directly as final_verts[row][col] to avoid a
    # transpose operation that was incorrect for non-square patches.
    final_verts = [[None] * out_cols for _ in range(out_rows)]
    final_uvs   = [[None] * out_cols for _ in range(out_rows)]

    for col in range(out_cols):
        col_pts_v = []
        for seg in range(num_segs_v):
            ri = seg * 2
            p0 = row_curves[ri    ][col]
            p1 = row_curves[ri + 1][col]
            p2 = row_curves[ri + 2][col]
            seg_pts = _sample_segment(p0, p1, p2, steps_v)
            if seg == 0:
                col_pts_v.extend(seg_pts)
            else:
                col_pts_v.extend(seg_pts[1:])

        # col_pts_v is indexed [row] for this column
        for r, pt in enumerate(col_pts_v):
            final_verts[r][col] = pt
            # idTech4's raw "st" is never flipped internally by the engine
            # (confirmed against idSurface_Patch::SampleSinglePatchPoint /
            # LerpVert, which blend it with the same bezier basis as
            # position and pass it straight through) — but that's an
            # internal-consistency fact, not a statement about Blender's
            # convention. idTech4/Quake-lineage engines use a top-left
            # origin for texture coordinates, while Blender uses
            # bottom-left, so V must still be flipped when converting.
            final_uvs[r][col]   = (pt[3], 1.0 - pt[4])

    return final_verts, final_uvs


def build_patch_mesh(patch, scale):
    """
    Tessellate a MapPatch into a Blender mesh.
    Returns (mesh, [material_name]) or (None, []).
    """
    if patch.cols < 3 or patch.rows < 3:
        return None, []
    if len(patch.ctrl) != patch.rows:
        return None, []
    # Guard against truncated/malformed control data (e.g. the file ended
    # before every column was read) — any unfilled cell would otherwise
    # crash the tessellator.
    for row in patch.ctrl:
        if len(row) != patch.cols or any(pt is None for pt in row):
            return None, []

    verts_grid, uvs_grid = _tessellate_patch(
        patch.ctrl, patch.cols, patch.rows,
        patch.subdiv_x, patch.subdiv_y
    )

    out_rows = len(verts_grid)
    out_cols = len(verts_grid[0]) if out_rows else 0
    if out_rows < 2 or out_cols < 2:
        return None, []

    bm = bmesh.new()
    uv_lay = bm.loops.layers.uv.new("UVMap")

    # Build vertex grid
    bv = []
    for r in range(out_rows):
        row_bv = []
        for c in range(out_cols):
            pt = verts_grid[r][c]
            row_bv.append(bm.verts.new(to_bl(Vector((pt[0], pt[1], pt[2])), scale)))
        bv.append(row_bv)

    bm.verts.ensure_lookup_table()

    # Build quad faces
    for r in range(out_rows - 1):
        for c in range(out_cols - 1):
            v0 = bv[r    ][c    ]
            v1 = bv[r    ][c + 1]
            v2 = bv[r + 1][c + 1]
            v3 = bv[r + 1][c    ]
            try:
                face = bm.faces.new((v0, v1, v2, v3))
                # UVs: stored in the patch control data, already world-projected
                uvs = [
                    uvs_grid[r    ][c    ],
                    uvs_grid[r    ][c + 1],
                    uvs_grid[r + 1][c + 1],
                    uvs_grid[r + 1][c    ],
                ]
                for loop, uv in zip(face.loops, uvs):
                    loop[uv_lay].uv = uv
            except ValueError:
                continue  # degenerate face

    if not bm.faces:
        bm.free()
        return None, []

    bm.normal_update()
    mesh = bpy.data.meshes.new("patch")
    bm.to_mesh(mesh)
    bm.free()
    mesh.validate(clean_customdata=False)
    mesh.update()
    return mesh, [patch.material]


# ─────────────────────────────────────────────────────────────────────
#  STATIC MESH IMPORT  (.ase / .lwo) — via companion addon
# ─────────────────────────────────────────────────────────────────────
#
# .ase/.lwo parsing and mesh-building used to live inline here; it now
# lives in the separate idTech4_ase_lwo_io.py addon (see
# get_model_import_addon() near the top of this file), so this file
# never parses either format itself. The only thing called across that
# boundary is <that addon>.load_model_meshes(filepath, scale,
# lwo_first_layer_only=, name_hint=) -> [(name, bpy.types.Mesh,
# [material_name, ...]), ...], used below wherever a .map entity's
# resolved model file is loaded — only ever reached after confirming
# get_model_import_addon() returned a module.

def _user_roots(user_root):
    """Normalise a user root argument to a tuple, highest priority first.

    Every resolver below takes either a single root (the old signature,
    still used by the derive-from-model walk) or several — Mod Base then
    Base — so the two cases don't need separate call sites.
    """
    if not user_root:
        return ()
    if isinstance(user_root, str):
        return (user_root,)
    return tuple(r for r in user_root if r)


def model_search_paths(map_filepath, model_path, user_root=''):
    """Every path resolve_model_path would try for model_path, in order.

    Split out from resolve_model_path so a "model file not found" report
    can say exactly where it looked. That matters far more with a Mod
    Base configured: "not found" is otherwise indistinguishable between
    a mod that doesn't ship the model (fine, the base should have) and a
    Base Directory pointing at the wrong tree entirely.
    """
    model_path = model_path.replace('\\', '/').strip().strip('"')
    if not model_path:
        return []

    candidates = []
    for root in _user_roots(user_root):
        candidates.append(os.path.join(root, model_path))

    map_dir = os.path.dirname(map_filepath)
    cur = map_dir
    for _ in range(8):
        candidates.append(os.path.join(cur, model_path))
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    return [os.path.normpath(c) for c in candidates]


def resolve_model_path(map_filepath, model_path, user_root=''):
    """
    Resolve a "model" key value (e.g. "models/mapobjects/foo.lwo") from a
    .map file to an actual file on disk. Tries, in order:
      1. As an absolute path, if it already is one.
      2. Joined with each user-supplied game/mod root directory, in the
         order given — Mod Base before Base, so a mod's own copy of a
         model wins and anything it doesn't ship falls through to the
         base game's.
      3. Joined with the .map file's own directory.
      4. Joined with each ancestor directory of the .map file, up to a
         handful of levels (covers the common .../base/maps/foo.map layout,
         where the model path is relative to .../base/).
    Returns the resolved path, or None if no candidate exists on disk.
    """
    model_path = model_path.replace('\\', '/').strip().strip('"')
    if not model_path:
        return None

    if os.path.isabs(model_path) and os.path.isfile(model_path):
        return model_path

    for c in model_search_paths(map_filepath, model_path, user_root):
        if os.path.isfile(c):
            return c
    return None


def resolve_decl_roots(map_filepath, subdir, user_root=''):
    """
    Every existing <root>/<subdir> directory, highest priority first —
    the union of:
      1. <subdir> under each user-supplied game/mod root, in the order
         given (Mod Base before Base).
      2. <subdir> under the .map file's own directory.
      3. <subdir> under each ancestor directory of the .map file, up to
         the same handful of levels used for model resolution (covers
         the common .../base/def/ layout when the .map file lives at
         .../base/maps/foo.map).

    EVERY hit is returned, not just the first. That is the whole point
    with a Mod Base configured: a mod ships a handful of .def files and
    inherits the rest, so stopping at <mod>/def would hide the base
    game's entire def tree behind three files. Callers scan the list in
    order and let the first declaration of a given name win, which is
    what the engine's own fs_game search order does.

    Duplicates are dropped (a normpath-keyed seen-set), because the
    ancestor walk reaches the base directory again whenever the .map
    file lives inside the tree a user root already points at.
    """
    candidates = []
    for root in _user_roots(user_root):
        candidates.append(os.path.join(root, subdir))

    map_dir = os.path.dirname(map_filepath)
    cur = map_dir
    for _ in range(8):
        candidates.append(os.path.join(cur, subdir))
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    found, seen = [], set()
    for c in candidates:
        c = os.path.normpath(c)
        key = os.path.normcase(c)
        if key in seen:
            continue
        seen.add(key)
        if os.path.isdir(c):
            found.append(c)
    return found


def resolve_def_roots(map_filepath, user_root=''):
    """Every "def" directory to scan, highest priority first — see
    resolve_decl_roots."""
    return resolve_decl_roots(map_filepath, 'def', user_root)


def resolve_skins_roots(map_filepath, user_root=''):
    """Every "skins" directory to scan, highest priority first — see
    resolve_decl_roots."""
    return resolve_decl_roots(map_filepath, 'skins', user_root)


def resolve_def_root(map_filepath, user_root=''):
    """The single highest-priority "def" directory, or None.

    Kept for the callers that genuinely want one directory rather than a
    search list; anything that SCANS def files wants resolve_def_roots,
    or a mod's three .def files hide the base game's nine hundred.
    """
    roots = resolve_def_roots(map_filepath, user_root)
    return roots[0] if roots else None


def resolve_skins_root(map_filepath, user_root=''):
    """The single highest-priority "skins" directory, or None — see
    resolve_def_root for why the plural form is usually the one wanted."""
    roots = resolve_skins_roots(map_filepath, user_root)
    return roots[0] if roots else None


def resolve_materials_roots(map_filepath, user_root=''):
    """Every "materials" directory (.mtr files) to scan, highest priority
    first — see resolve_decl_roots. The game/mod root that owns each one
    (needed separately, to resolve "textures/..." image paths referenced
    from inside .mtr files) is just that path's parent directory:
    "materials" is always a direct child of it here, by the same
    construction the def and skins scans rely on."""
    return resolve_decl_roots(map_filepath, 'materials', user_root)


def resolve_materials_root(map_filepath, user_root=''):
    """The single highest-priority "materials" directory, or None.

    This is the one the derive-from-model walk wants: it asks "which tree
    is this file in?" and answers with one root, whose parent becomes the
    derived Base Directory. A search list would have no single parent to
    derive from.
    """
    roots = resolve_materials_roots(map_filepath, user_root)
    return roots[0] if roots else None


# ─────────────────────────────────────────────────────────────────────
#  SHARED idTech4 SOURCES (Base Directory / Materials Source). Duplicated
#  identically across every idTech4 Blender addon (map import, .ase/.lwo
#  import, MD5 tools, materials) so Base Directory/Materials Source work
#  no matter which subset of these addons happens to be installed — they
#  used to live only in the "idTech4 Materials" addon's own
#  AddonPreferences, so nothing could read OR save them unless that one
#  specific addon was enabled. Backed here by a small JSON file under
#  Blender's per-user config directory instead, shared by plain file
#  path rather than by any addon's registration state. The panel and its
#  two operators are registered at most once regardless of how many of
#  these addons are enabled at once — see _register_shared_ui.
# ─────────────────────────────────────────────────────────────────────

_SHARED_CONFIG_SUBDIR = "idTech4"
_SHARED_CONFIG_FILE   = "shared_sources.json"


def _shared_config_path():
    d = bpy.utils.user_resource('CONFIG', path=_SHARED_CONFIG_SUBDIR, create=True)
    return os.path.join(d, _SHARED_CONFIG_FILE)


def get_shared_paths():
    """Return (base_directory, mod_base_directory, materials_mtr_source)
    from the shared idTech4 config file, or ('', '', '') if it doesn't
    exist yet or is unreadable.

    mod_base_directory is optional and usually blank. When it is set it
    takes priority over base_directory for every asset lookup, with
    base_directory still searched behind it - see shared_search_roots.
    """
    try:
        with open(_shared_config_path(), 'r', encoding='utf-8') as f:
            data = json.load(f)
        return (data.get('base_directory', ''),
                data.get('mod_base_directory', ''),
                data.get('materials_mtr_source', ''))
    except (OSError, ValueError):
        return '', '', ''


def set_shared_paths(base_directory=None, mod_base_directory=None,
                     materials_mtr_source=None):
    """Update the shared idTech4 config file, creating it if needed.
    Any argument left as None leaves that field unchanged.

    Every idTech4 addon carries its own copy of this function and each
    one rewrites the whole file, so a copy predating mod_base_directory
    DROPS that key on its next write, silently clearing a configured Mod
    Base. Nothing on this side can defend against that - the four addons
    have to be updated together.
    """
    base, mod_base, mtr_source = get_shared_paths()
    if base_directory is not None:
        base = base_directory
    if mod_base_directory is not None:
        mod_base = mod_base_directory
    if materials_mtr_source is not None:
        mtr_source = materials_mtr_source
    with open(_shared_config_path(), 'w', encoding='utf-8') as f:
        json.dump({'base_directory': base,
                   'mod_base_directory': mod_base,
                   'materials_mtr_source': mtr_source}, f, indent=2)


def shared_search_roots(base_directory=None, mod_base_directory=None):
    """Every asset root to search, highest priority first.

    Mod Base, when set, is searched BEFORE Base, and Base is still
    searched after it - the engine's own fs_game behaviour, where a mod
    supplies part of the tree and inherits the rest. The fallback is per
    ITEM, not per tree: a mod shipping three .def files must not hide the
    base's nine hundred, so a caller scanning a whole directory unions
    every root instead of stopping at the first one that exists.

    Either argument left as None is read from the shared config, so a
    caller with no override of its own can pass nothing. A caller that
    HAS an override (a gate one-off, or a base derived from the imported
    file's own location) passes mod_base_directory='' - a stored Mod Base
    pairs with the stored Base, not with an arbitrary derived root.
    """
    if base_directory is None or mod_base_directory is None:
        cfg_base, cfg_mod, _ = get_shared_paths()
        if base_directory is None:
            base_directory = cfg_base
        if mod_base_directory is None:
            mod_base_directory = cfg_mod
    return tuple(r for r in (mod_base_directory, base_directory) if r)


class IDTECH4_OT_SelectBaseDirectory(bpy.types.Operator):
    """Open a directory browser to set the idTech4 Base Directory
    (shared across every idTech4 addon)"""
    bl_idname  = "idtech4.select_base_directory"
    bl_label   = "Select Base Directory"
    bl_options = {'REGISTER', 'UNDO'}

    # Use filepath (not directory) so the browser's Accept button fires
    # execute() immediately when clicked, rather than navigating into
    # the folder — execute() then extracts the directory portion, so
    # either clicking any file inside the target folder or just
    # pressing Accept while sitting in it both work.
    filepath: StringProperty(name="File Path", subtype='FILE_PATH', default="")

    def invoke(self, context, event):
        self.filepath = ""  # start with empty filename field
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        dirpath = os.path.dirname(self.filepath)
        if not dirpath.endswith(os.sep):
            dirpath += os.sep
        set_shared_paths(base_directory=dirpath)
        return {'FINISHED'}


class IDTECH4_OT_SelectModBaseDirectory(bpy.types.Operator):
    """Open a directory browser to set the idTech4 Mod Base Directory
    (shared across every idTech4 addon)"""
    bl_idname  = "idtech4.select_mod_base_directory"
    bl_label   = "Select Mod Base Directory"
    bl_options = {'REGISTER', 'UNDO'}

    # Same filepath-not-directory trick as the Base Directory operator
    # above, for the same reason - see its comment.
    filepath: StringProperty(name="File Path", subtype='FILE_PATH', default="")

    def invoke(self, context, event):
        self.filepath = ""  # start with empty filename field
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        dirpath = os.path.dirname(self.filepath)
        if not dirpath.endswith(os.sep):
            dirpath += os.sep
        set_shared_paths(mod_base_directory=dirpath)
        return {'FINISHED'}


class IDTECH4_OT_SelectSource(bpy.types.Operator):
    """Open a file browser to select a .mtr file or materials directory
    (shared across every idTech4 addon)"""
    bl_idname  = "idtech4.select_source"
    bl_label   = "Select Source (.mtr file or directory)"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: StringProperty(
        name="Source Path",
        description="Path to a .mtr file or a directory containing .mtr files.",
        subtype='FILE_PATH', default="")
    filter_glob: StringProperty(default="*.mtr", options={'HIDDEN'})

    def invoke(self, context, event):
        self.filepath = ""  # start with empty filename field
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        chosen     = self.filepath
        abs_chosen = bpy.path.abspath(chosen)

        if os.path.isdir(abs_chosen):
            chosen = abs_chosen if abs_chosen.endswith(os.sep) else abs_chosen + os.sep
        elif os.path.isfile(abs_chosen):
            if not abs_chosen.lower().endswith('.mtr'):
                self.report(
                    {'ERROR'},
                    f"Source must be a .mtr file or a directory containing .mtr files, "
                    f"not '{os.path.basename(abs_chosen)}'. "
                    f"Please select a .mtr file or a materials directory."
                )
                return {'CANCELLED'}
            chosen = abs_chosen
        else:
            chosen = abs_chosen

        set_shared_paths(materials_mtr_source=chosen)
        return {'FINISHED'}


class IDTECH4_OT_ClearSharedPath(bpy.types.Operator):
    """Clear a shared idTech4 path (Base Directory or Materials Source)"""
    bl_idname  = "idtech4.clear_shared_path"
    bl_label   = "Clear"
    bl_options = {'REGISTER', 'UNDO'}

    target: EnumProperty(
        items=[
            ('BASE',   "Base Directory",     ""),
            ('MOD',    "Mod Base Directory", ""),
            ('SOURCE', "Materials Source",   ""),
        ],
        options={'HIDDEN'},
    )

    def execute(self, context):
        if self.target == 'BASE':
            set_shared_paths(base_directory='')
        elif self.target == 'MOD':
            set_shared_paths(mod_base_directory='')
        else:
            set_shared_paths(materials_mtr_source='')
        return {'FINISHED'}


def _get_shared_base_display(self):
    """Read-only mirror of the shared config's Base Directory, so the
    Sources panel can show it in a greyed-out text field. A
    StringProperty given a `get` but no `set` is read-only at the RNA
    level, which is what makes Blender draw the field locked - the value
    can only be changed through the browse/clear buttons beside it, and
    a stray click can't leave a typed path that nothing would save."""
    base, _mod, _ = get_shared_paths()
    return base or "(not set)"


def _get_shared_mod_base_display(self):
    """Read-only mirror of the shared config's Mod Base Directory - see
    _get_shared_base_display. Blank is the normal state and means "just
    use Base Directory", so it says that rather than "(not set)", which
    would read as something left misconfigured."""
    _base, mod, _ = get_shared_paths()
    return mod or "(none - using Base Directory only)"


def _get_shared_source_display(self):
    """Read-only mirror of the shared config's Materials Source - see
    _get_shared_base_display. Spells out the unset-but-defaulted case
    the same way the importers resolve it, so the field shows the path
    that will actually be read rather than an empty box - including the
    Mod Base half of it, since with a Mod Base set the default is BOTH
    trees' materials folders, mod first."""
    base, mod, source = get_shared_paths()
    if source:
        return source
    roots = shared_search_roots(base, mod)
    if roots:
        return "(defaults to %s)" % ", ".join(
            os.path.join(r, 'materials') for r in roots)
    return "(not set)"


class IDTECH4_PG_SharedPathDisplay(bpy.types.PropertyGroup):
    """The two shared paths as locked display fields. Lives on
    WindowManager rather than Scene: both are only a live view of the
    shared config file, so there is nothing worth saving into a .blend
    and nothing that could come back from one stale."""
    base_directory: StringProperty(
        name="Base Directory",
        description="Shared idTech4 Base Directory. Read-only - use the "
                    "folder button to change it, or the X to clear it",
        get=_get_shared_base_display,
    )
    mod_base_directory: StringProperty(
        name="Mod Base Directory",
        description="Optional shared idTech4 Mod Base Directory, searched "
                    "BEFORE Base Directory - anything the mod does not "
                    "supply falls back to Base Directory. Leave unset to "
                    "use Base Directory alone. Read-only - use the folder "
                    "button to change it, or the X to clear it",
        get=_get_shared_mod_base_display,
    )
    materials_source: StringProperty(
        name="Materials Source",
        description="Shared idTech4 Materials Source: a .mtr file, or a "
                    "folder of them. Search order: this path if set, and "
                    "nothing else; otherwise <Mod Base>/materials if a Mod "
                    "Base is set, then <Base Directory>/materials. "
                    "Read-only - use the folder button to change it, or "
                    "the X to clear it",
        get=_get_shared_source_display,
    )


class IDTECH4_PT_sources(bpy.types.Panel):
    """Base Directory / Materials Source, shared across every idTech4
    addon regardless of which subset is installed."""
    bl_label       = "Sources"
    bl_idname      = "IDTECH4_PT_sources"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = "idTech4"

    def draw(self, context):
        layout = self.layout
        base_dir, mod_dir, source_path = get_shared_paths()

        # Panel.bl_description only tooltips a popover *button* that
        # opens this panel elsewhere — it does nothing for the sidebar's
        # own category tab strip (no public API covers that), so the
        # explanation lives here instead, as plain text. Split into
        # short lines since layout.label() never wraps on its own — one
        # long string would just run off the sidebar's edge when it's
        # narrower than the text (same reasoning as the gate popup's own
        # multi-line notes elsewhere in this file).
        info = layout.box()
        info.label(text="Base / Mod Base / Materials Source —", icon='INFO')
        info.label(text="shared across every idTech4 addon:")
        info.label(text="map import, .ase/.lwo import, MD5")
        info.label(text="tools, and materials.")

        # Same shape as the Materials panel's own Source Paths block:
        # a label above a locked field, with browse and clear beside it.
        # The field is read-only (see IDTECH4_PG_SharedPathDisplay) and
        # its tooltip carries the full path, which is what a narrow
        # sidebar truncates away.
        box = layout.box()
        box.label(text="Source Paths", icon='INFO')

        # getattr, not plain attribute access: an older copy of one of
        # the other idTech4 addons may have been the one to register the
        # shared UI, in which case this PropertyGroup was never attached.
        # Fall back to plain labels rather than drawing a broken panel.
        display = getattr(context.window_manager, 'idtech4_shared_paths', None)

        col = box.column(align=True)
        col.label(text="Base Directory:")
        row = col.row(align=True)
        if display:
            row.prop(display, "base_directory", text="")
        else:
            row.label(text=base_dir or "(not set)")
        row.operator("idtech4.select_base_directory", text="",
                     icon='FILEBROWSER')
        clear = row.row(align=True)
        clear.enabled = bool(base_dir)
        clear.operator("idtech4.clear_shared_path", text="",
                       icon='X').target = 'BASE'

        # Directly under Base Directory and in the same format, because
        # it is the same kind of thing: a game tree root. The only
        # differences are that this one is optional and that it is
        # searched first.
        col = box.column(align=True)
        col.label(text="Mod Base Directory (optional):")
        row = col.row(align=True)
        if display:
            row.prop(display, "mod_base_directory", text="")
        else:
            row.label(text=mod_dir or "(none)")
        row.operator("idtech4.select_mod_base_directory", text="",
                     icon='FILEBROWSER')
        clear = row.row(align=True)
        clear.enabled = bool(mod_dir)
        clear.operator("idtech4.clear_shared_path", text="",
                       icon='X').target = 'MOD'

        col = box.column(align=True)
        col.label(text="Materials Source:")
        row = col.row(align=True)
        if display:
            row.prop(display, "materials_source", text="")
        else:
            row.label(text=source_path or "(not set)")
        row.operator("idtech4.select_source", text="", icon='FILEBROWSER')
        clear = row.row(align=True)
        clear.enabled = bool(source_path)
        clear.operator("idtech4.clear_shared_path", text="",
                       icon='X').target = 'SOURCE'


_SHARED_UI_CLASSES = (
    # The PropertyGroup first - the WindowManager pointer below can only
    # be created once its type is registered.
    IDTECH4_PG_SharedPathDisplay,
    IDTECH4_OT_SelectBaseDirectory,
    IDTECH4_OT_SelectModBaseDirectory,
    IDTECH4_OT_SelectSource,
    IDTECH4_OT_ClearSharedPath,
    IDTECH4_PT_sources,
)


def _register_shared_ui():
    """Register the shared Sources panel/operators at most once, no
    matter how many of the idTech4 addons are enabled at the same time —
    each carries an identical copy of these 3 classes, guarded here by a
    refcount kept in bpy.app.driver_namespace (a plain dict Blender
    keeps alive for the whole session, shared by every addon) so the
    second and later addons to register just add to the count instead of
    calling bpy.utils.register_class a second time, which would raise —
    Blender doesn't allow two classes registered under the same
    bl_idname at once."""
    ns    = bpy.app.driver_namespace
    count = ns.get('_idtech4_shared_ui_refcount', 0)
    if count == 0:
        for cls in _SHARED_UI_CLASSES:
            bpy.utils.register_class(cls)
        bpy.types.WindowManager.idtech4_shared_paths = (
            bpy.props.PointerProperty(type=IDTECH4_PG_SharedPathDisplay))
    ns['_idtech4_shared_ui_refcount'] = count + 1


def _unregister_shared_ui():
    """Undo _register_shared_ui — only actually unregisters once the
    last addon that had registered it is also unregistering, so the
    panel stays available as long as any one of the idTech4 addons is
    still enabled."""
    ns    = bpy.app.driver_namespace
    count = ns.get('_idtech4_shared_ui_refcount', 1) - 1
    if count <= 0:
        # Before the classes, since the pointer's type is one of them.
        # Guarded the same way, and for the same reason: a different
        # addon may have been the one to create it.
        if hasattr(bpy.types.WindowManager, 'idtech4_shared_paths'):
            del bpy.types.WindowManager.idtech4_shared_paths
        for cls in reversed(_SHARED_UI_CLASSES):
            # Unregister whatever is actually registered under this name,
            # not this module's own copy of the class. Every idTech4 addon
            # carries its own identical copy, and only the FIRST one to
            # register calls register_class - so if a different addon is the
            # last one to unregister, its copies were never registered and
            # unregister_class raises "missing bl_rna attribute", leaving the
            # Sources panel registered as a zombie. Going through bpy.types
            # picks the right object whichever addon got there first.
            registered = getattr(bpy.types, cls.__name__, cls)
            try:
                bpy.utils.unregister_class(registered)
            except (RuntimeError, ValueError):
                pass
        ns.pop('_idtech4_shared_ui_refcount', None)
    else:
        ns['_idtech4_shared_ui_refcount'] = count


def _resolve_sources(filepath, derive_from_model, override_base_directory,
                      override_source_path, save_derived_as_default=False):
    """Return (base_directory, mod_base_directory, source_path).
    base_directory, in priority
    order:
      1. derive_from_model=True — walk up from the .map file's own
         directory looking for a "materials" folder (resolve_materials_
         root, no hint), base_directory = that folder's parent. If
         save_derived_as_default is also True (the sources gate's
         "...and save settings" choice) and something was found, this
         ALSO writes the result into the shared idTech4 config file
         (get_shared_paths/set_shared_paths) before returning — the only
         place that can happen, since the actual derived paths aren't
         known until a real filepath exists, which is after the gate's
         own popup has already closed.
      2. override_base_directory, if non-blank — a one-off value from
         the sources gate's "Select now" choice that the user chose NOT
         to save as the default.
      3. Otherwise the shared idTech4 config file's base_directory — the
         normal case, shared across every idTech4 addon regardless of
         which are currently installed/enabled. Blank if none of the
         above resolved anything.

    mod_base_directory is only ever the shared config's optional Mod
    Base, and ONLY in case 3. A stored Mod Base pairs with the stored
    Base; pairing it with a one-off gate override or a base derived from
    the imported file's own location would search a mod tree that has
    nothing to do with either, so both of those return it blank.

    source_path may be a TUPLE rather than a string. An explicitly
    configured Materials Source is used exactly as given — that is what
    the field is for — but a source DERIVED from the roots is one
    "materials" folder per root, mod first, because a mod ships a
    handful of .mtr files and inherits the rest.

    source_path is mandatory-base/optional-source: whichever of the
    above found a base_directory, source_path follows the same
    priority (override_source_path, else the shared config file's
    materials_mtr_source) but — unlike base_directory — is allowed to
    come up blank, in which case it defaults to base_directory's own
    "materials" subfolder. That folder not actually existing isn't
    checked here; collect_mtr_files (idTech4_material_import.py) just
    finds nothing under a missing path, so the materials post-pass's own
    "no .mtr materials found under X" issue message already covers a
    default that doesn't pan out, same as it would for a hand-typed
    wrong path.

    base_directory is the single root this file uses for everything:
    def/skins/model resolution (resolve_def_root/resolve_skins_root/
    resolve_model_path) AND the materials post-pass — previously the
    former used a separate, second "Game/Mod Root" (models_root) field,
    which just meant configuring the same folder twice."""
    if derive_from_model:
        found = resolve_materials_root(filepath, '')
        if found is None:
            base_dir, source_path = '', ''
        else:
            base_dir, source_path = os.path.dirname(found), found
            if save_derived_as_default:
                set_shared_paths(base_directory=base_dir, materials_mtr_source=found)
        mod_dir = ''
    elif override_base_directory or override_source_path:
        base_dir, source_path = override_base_directory, override_source_path
        mod_dir = ''
    else:
        base_dir, mod_dir, source_path = get_shared_paths()

    if not source_path:
        roots = shared_search_roots(base_dir, mod_dir)
        if roots:
            source_path = tuple(os.path.join(r, 'materials') for r in roots)

    return base_dir, mod_dir, source_path


# ---------------------------------------------------------------------------
# Everything from the `import re as _re` below down to
# `def model_shading_overrides` is sliced out of this file and exec()d under
# plain CPython by tests/decl_parity.py and tests/decl_inherit.py: the DECL
# LEXER block, IdDict, the three decl parsers and the name helpers. NONE of
# it may touch bpy, and the two anchor lines have to keep their spelling.
#
# tests/decl_parity.py also pulls the DECL LEXER and DECL CANONICAL blocks
# out on their own, by their banner lines, and runs them with nothing but
# builtins in scope - so each block imports whatever it needs itself.
# ---------------------------------------------------------------------------


# ===========================================================================
# BEGIN DECL LEXER
#
# idLexer (idlib/Lexer.cpp), cut down to what a decl file needs, and
# parameterised by the same flag bits the engine uses so one copy serves
# every format. This addon lexes .def and .skin with DECL_LEXER_FLAGS; the
# constants for the .map and .md5mesh flag sets are here too, unused, so a
# sibling addon can drop this block in and ask for its own combination
# without editing it.
#
# NOTHING BETWEEN THE BEGIN/END BANNERS MAY TOUCH bpy.
#
# That restriction is load-bearing, not stylistic: tests/decl_parity.py
# slices this block straight out of this file by its banner comments and
# exec()s it under plain CPython, so the lexer can be held to the shipping
# corpora without launching Blender. Keep imports here to the standard
# library - `re` is the only one it uses.
#
# WHY THIS EXISTS. The .def parser was a pair of flat regexes matching
# `"key" "value"` pairs out of raw text. DECL_LEXER_FLAGS sets
# LEXFL_ALLOWBACKSLASHSTRINGCONCAT, which joins a string, a backslash and
# the next string into ONE token:
#
#     "editor_var trigger"  "If set, ... If 'attack_path' is set," \
#                           "monster waits until triggered to follow it."
#
# A pair-matching regex sees three strings, not two, so every key/value pair
# after the continuation is shifted by one - wrong key AND wrong value for
# the rest of the body, then propagated to every def that inherits it. It is
# 105 continuations across 14 shipped .def files, and monster_default is one
# of the casualties, so essentially every Doom 3 monster inherited garbage.
#
# WHERE THIS DELIBERATELY DIVERGES FROM idLexer. The engine's ReadName stops
# at any punctuation character, so `-` and `*` end a name and come back as
# their own tokens. Reproducing that faithfully would split 85 model paths in
# The Dark Mod's .skin files (models/darkmod/lights/non-extinguishable/...)
# and ~34,000 bare tokens across the .def corpora (`-dest`, `-prefix`,
# `*Hips`), which is a real engine mis-parse that we would be importing for
# nothing: the skin table those files produce today is correct and is what
# the acceptance test pins. So a bare token here is the same
# [^\s{}()"]+ run the old _MODEL_TOKEN_RE used - path characters admitted,
# plus everything else that is not a delimiter. The string rules, which are
# where the demonstrated corruption lives, are the engine's exactly.
# ===========================================================================

import re as _re          # self-contained: the block is exec'd standalone

# idlib/Lexer.h:50 - the bit values, so a flag set reads like the engine's.
LEXFL_NOFATALERRORS = 1 << 2
LEXFL_NOSTRINGCONCAT = 1 << 3            # whitespace does NOT join two strings
LEXFL_NOSTRINGESCAPECHARS = 1 << 4       # no \n / \" processing inside strings
LEXFL_ALLOWPATHNAMES = 1 << 7
LEXFL_ALLOWMULTICHARLITERALS = 1 << 11
LEXFL_ALLOWBACKSLASHSTRINGCONCAT = 1 << 12   # '\' between strings joins them

# framework/DeclManager.h:93 - .mtr, .def and .skin are all read with these.
DECL_LEXER_FLAGS = (LEXFL_NOSTRINGCONCAT | LEXFL_NOSTRINGESCAPECHARS |
                    LEXFL_ALLOWPATHNAMES | LEXFL_ALLOWMULTICHARLITERALS |
                    LEXFL_ALLOWBACKSLASHSTRINGCONCAT | LEXFL_NOFATALERRORS)

# idlib/MapFile.cpp:722 - no concatenation of any kind.
MAP_LEXER_FLAGS = (LEXFL_NOSTRINGCONCAT | LEXFL_NOSTRINGESCAPECHARS |
                   LEXFL_ALLOWPATHNAMES)

# renderer/Model_md5.cpp:497 - NOSTRINGCONCAT is NOT set, so two strings
# separated by whitespace ARE one token there.
MD5_LEXER_FLAGS = LEXFL_ALLOWPATHNAMES | LEXFL_NOSTRINGESCAPECHARS

DECL_TT_STRING = 'string'
DECL_TT_NAME = 'name'
DECL_TT_PUNCT = 'punct'

# Punctuation kept as its own token. The engine's table is much larger, but
# these five are the only ones any decl grammar we read is structured by, and
# widening it would start splitting the bare names above. See the divergence
# note in the banner.
_DECL_PUNCT = '{}()'

_DECL_ESCAPES = {'n': '\n', 't': '\t', 'r': '\r', 'v': '\v', 'b': '\b',
                 'f': '\f', 'a': '\a', '\\': '\\', "'": "'", '"': '"'}

_DECL_RE_CACHE = {}


def _decl_token_re(escapes):
    """The token alternation, one compiled form per string rule.

    Regex here produces TOKENS - it never recognises grammar. That is the
    line the .def parser crossed: `"key"\\s*"value"` matched structure out of
    raw text and could not see a continuation at all.
    """
    rx = _DECL_RE_CACHE.get(escapes)
    if rx is None:
        string = r'"(?:[^"\\]|\\.)*"?' if escapes else r'"[^"]*"?'
        rx = _re.compile(
            r'(?P<BLOCKCOMMENT>/\*.*?(?:\*/|\Z))'
            r'|(?P<LINECOMMENT>//[^\n]*)'
            r'|(?P<WS>\s+)'
            r'|(?P<STRING>' + string + r')'
            r'|(?P<PUNCT>[' + _re.escape(_DECL_PUNCT) + r'])'
            r'|(?P<NAME>[^\s{}()"]+)',
            _re.DOTALL)
        _DECL_RE_CACHE[escapes] = rx
    return rx


def _decl_unescape(body):
    """idLexer::ReadEscapeCharacter, for the flag sets that ask for it.

    Never runs for a decl file - DECL_LEXER_FLAGS sets NOSTRINGESCAPECHARS,
    which is exactly why "[^"]*" is the engine-correct string pattern there.
    It is here so the block stays honest about what the flag means.
    """
    if '\\' not in body:
        return body
    out = []
    i, n = 0, len(body)
    while i < n:
        c = body[i]
        if c == '\\' and i + 1 < n:
            nxt = body[i + 1]
            if nxt in _DECL_ESCAPES:
                out.append(_DECL_ESCAPES[nxt])
                i += 2
                continue
        out.append(c)
        i += 1
    return ''.join(out)


class DeclToken(object):
    """One token: DECL_TT_STRING / DECL_TT_NAME / DECL_TT_PUNCT, its text
    (quotes already stripped for a string) and the 1-based line it began on,
    which is what a diagnostic can point the user at."""

    __slots__ = ('type', 'val', 'line')

    def __init__(self, type_, val, line):
        self.type = type_
        self.val = val
        self.line = line

    def __repr__(self):
        return '<%s %r@%d>' % (self.type, self.val, self.line)


def decl_lex(text, flags=DECL_LEXER_FLAGS):
    """Tokenise decl source. Returns a list of DeclToken.

    Comments are handled here rather than stripped beforehand, so a `//` or
    `/*` INSIDE a quoted value can no longer truncate it. No shipped decl
    file has one - the pre-strip was safe in practice - but the string rules
    below only mean anything if the string boundaries are found first.
    """
    escapes = not (flags & LEXFL_NOSTRINGESCAPECHARS)
    ws_concat = not (flags & LEXFL_NOSTRINGCONCAT)
    bs_concat = bool(flags & LEXFL_ALLOWBACKSLASHSTRINGCONCAT)

    raw = []
    line = 1
    last = 0
    for m in _decl_token_re(escapes).finditer(text):
        kind = m.lastgroup
        start = m.start()
        line += text.count('\n', last, start)
        last = m.end()
        if kind in ('BLOCKCOMMENT', 'LINECOMMENT', 'WS'):
            line += text.count('\n', start, last)
            continue
        val = m.group()
        if kind == 'STRING':
            body = val[1:-1] if len(val) > 1 and val.endswith('"') else val[1:]
            raw.append((DECL_TT_STRING,
                        _decl_unescape(body) if escapes else body, line))
        elif kind == 'PUNCT':
            raw.append((DECL_TT_PUNCT, val, line))
        else:
            raw.append((DECL_TT_NAME, val, line))
        line += text.count('\n', start, last)

    if not (ws_concat or bs_concat):
        return [DeclToken(t, v, ln) for t, v, ln in raw]

    # idLexer::ReadString's concatenation loop. A lone '\' arrives here as a
    # NAME token because the bare-word class stops at the quote on either
    # side of it, which is what makes the join detectable at all.
    out = []
    i, n = 0, len(raw)
    while i < n:
        kind, val, line = raw[i]
        if kind != DECL_TT_STRING:
            out.append(DeclToken(kind, val, line))
            i += 1
            continue
        buf = val
        j = i + 1
        while j < n:
            if (bs_concat and raw[j][0] == DECL_TT_NAME and raw[j][1] == '\\'
                    and j + 1 < n and raw[j + 1][0] == DECL_TT_STRING):
                buf += raw[j + 1][1]
                j += 2
                continue
            if ws_concat and raw[j][0] == DECL_TT_STRING:
                buf += raw[j][1]
                j += 1
                continue
            break
        out.append(DeclToken(DECL_TT_STRING, buf, line))
        i = j
    return out


def decl_skip_braced(tokens, start):
    """Index just past the '}' matching the '{' at *start*, or len(tokens).

    idLexer::SkipBracedSection. Nested braces are tracked, so a per-anim
    sound block or a nested options group cannot end the decl early.
    """
    depth = 0
    i, n = start, len(tokens)
    while i < n:
        val = tokens[i].val
        if tokens[i].type == DECL_TT_PUNCT:
            if val == '{':
                depth += 1
            elif val == '}':
                depth -= 1
                if depth == 0:
                    return i + 1
        i += 1
    return n


def decl_scan(tokens):
    """idDeclFile::LoadAndParse's top-level scan.

    Yields (type_name, decl_name, body_start, body_end, line) with
    body_start/body_end as token indices bounding the body, exclusive of the
    braces themselves. A decl's own body is skipped wholesale, so a keyword
    that happens to appear INSIDE one is never mistaken for a declaration -
    which is the other thing the flat `\\bentityDef\\s+NAME\\s*\\{` regexes
    could not do.

    One divergence, deliberate: where the engine hits `type name` not
    followed by '{' it warns and re-reads from AFTER the offending token,
    this resyncs by one token instead. That can only find decls the engine
    would also find, plus any the old scanner did - it cannot lose one.
    """
    i, n = 0, len(tokens)
    while i < n:
        if tokens[i].val == '{' and tokens[i].type == DECL_TT_PUNCT:
            i = decl_skip_braced(tokens, i)          # "Missing decl name"
            continue
        if i + 1 >= n:
            return
        name_tok = tokens[i + 1]
        if name_tok.type == DECL_TT_PUNCT and name_tok.val == '{':
            i = decl_skip_braced(tokens, i + 1)      # "Missing decl name"
            continue
        if i + 2 >= n:
            return
        brace = tokens[i + 2]
        if not (brace.type == DECL_TT_PUNCT and brace.val == '{'):
            i += 1
            continue
        end = decl_skip_braced(tokens, i + 2)
        yield (tokens[i].val, name_tok.val, i + 3, end - 1, tokens[i].line)
        i = end


def decl_read_file(path, flags=DECL_LEXER_FLAGS):
    """Lex one decl file off disk, or None if it cannot be read."""
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as fh:
            text = fh.read()
    except Exception:
        return None
    return decl_lex(text, flags)


# END DECL LEXER
# ===========================================================================


# ─────────────────────────────────────────────────────────────────────
#  .DEF FILE PARSING (entityDef declarations)
# ─────────────────────────────────────────────────────────────────────

# idTech4 matches a decl's TYPE KEYWORD case-insensitively:
# idDeclFile::LoadAndParse reads the leading token and hands it to
# idDeclManagerLocal::GetDeclTypeFromName, which compares with
# idStr::Icmp (framework/DeclManager.cpp). Real content relies on it -
# The Dark Mod ships 268 "entitydef" and 2 "EntityDef" declarations
# alongside its 4485 canonical "entityDef" ones, and Doom 3's own base
# has three. Matching only the canonical spelling made every one of
# those invisible: any entity whose classname was declared that way
# silently resolved no keys at all, so it got no model, no skin, and no
# head or attachments. The NAME that follows stays case-SENSITIVE as
# written; every lookup against these dicts already lowercases both
# sides (see _norm_slashes / engine_canonical_decl).
DECL_TYPE_ENTITYDEF = 'entitydef'
DECL_TYPE_MODEL = 'model'
DECL_TYPE_SKIN = 'skin'


def _decl_scan_roots(root_dir):
    """The directories a decl scan should walk, highest priority first.

    Accepts one directory (the old signature) or several — a Mod Base
    makes def/, skins/ and materials/ each a LIST of directories rather
    than one, because a mod supplies part of the tree and inherits the
    rest. Non-existent entries are dropped here so every caller's own
    "nothing to scan" check stays a single `if not roots`.
    """
    if not root_dir:
        return []
    roots = [root_dir] if isinstance(root_dir, str) else list(root_dir)
    return [r for r in roots if r and os.path.isdir(r)]


class IdDict(dict):
    """idDict (idlib/Dict.cpp) — the container an entityDef body IS.

    A plain dict got two of its three rules wrong, and both are load-bearing:

    LAST OCCURRENCE WINS. idDict::Set looks the key up and REPLACES the
    value in place; only a key it has never seen is appended. The parser
    kept the first occurrence instead, so three shipped spawnargs resolved
    to a value the engine never uses — ai_volta_igna's editor_displayfolder,
    tdm_player_thief's climb_max_speed_vert_default and
    monster_hunter_invul's light_color.

    KEY ACCESS IS CASE-INSENSITIVE. idDict::FindKeyIndex hashes with
    GenerateKey( key, false ) — the `false` is "do not case-sensitise" — and
    confirms the hit with idStr::Icmp. So a def writing "Model" is read back
    by every `GetString( "model" )` in the engine. No shipped def in the five
    corpora depends on it, which is exactly why it was invisible.

    INSERTION ORDER IS PRESERVED, and the key keeps its ORIGINAL spelling
    and position when a later Set replaces its value — args is an idList,
    appended to, never re-ordered. resolve_entitydef_keys walks this order to
    reproduce MatchPrefix, where file order decides which parent wins, so
    sorting or re-inserting here would silently re-resolve 94 Quake 4 defs.

    Subclasses dict so it is one everywhere it is passed, iterated or
    compared; the case-folded index alongside is what makes lookup match
    Icmp without touching that order.
    """

    __slots__ = ('_folded',)

    def __init__(self, other=None):
        dict.__init__(self)
        self._folded = {}          # lowercased key -> key as first written
        if other:
            for k, v in (other.items() if hasattr(other, 'items') else other):
                self[k] = v

    # -- idDict::Set / FindKeyIndex -----------------------------------------
    def _real_key(self, key):
        return self._folded.get(key.lower() if isinstance(key, str) else key)

    def __setitem__(self, key, value):
        existing = self._real_key(key)
        if existing is None:
            self._folded[key.lower() if isinstance(key, str) else key] = key
            dict.__setitem__(self, key, value)
        else:
            dict.__setitem__(self, existing, value)

    def __getitem__(self, key):
        existing = self._real_key(key)
        if existing is None:
            raise KeyError(key)
        return dict.__getitem__(self, existing)

    def __contains__(self, key):
        return self._real_key(key) is not None

    def get(self, key, default=None):
        existing = self._real_key(key)
        return default if existing is None else dict.__getitem__(self, existing)

    def setdefault(self, key, default=None):
        existing = self._real_key(key)
        if existing is None:
            self[key] = default
            return default
        return dict.__getitem__(self, existing)

    def pop(self, key, *default):
        existing = self._real_key(key)
        if existing is None:
            if default:
                return default[0]
            raise KeyError(key)
        del self._folded[existing.lower() if isinstance(existing, str)
                         else existing]
        return dict.pop(self, existing)

    def __delitem__(self, key):
        self.pop(key)

    def update(self, other=None, **kw):
        if other:
            for k, v in (other.items() if hasattr(other, 'items') else other):
                self[k] = v
        for k, v in kw.items():
            self[k] = v

    def copy(self):
        return IdDict(self)

    def __reduce__(self):
        return (IdDict, (list(self.items()),))


def parse_def_files(root_dir):
    """
    Scan every .def file under root_dir for "entityDef NAME { ... }"
    declarations. Uses the same comment-stripping rules as the .map
    parser: // single-line and /* ... */ multi-line comments are both
    removed before anything else is scanned.

    Each declaration body is a flat set of "key" "value" pairs (matching
    real idTech4 .def syntax — see the moveable_base example), read off
    the DECL LEXER's token stream rather than matched out of raw text.
    Nested braces within a body (rare for entityDef, but not disallowed
    by the format) are tracked by decl_skip_braced, so the body boundary
    is always found correctly.

    Returns a dict: lowercase entityDef name -> {
        'keys': IdDict of {key: value, ...} in file order — keys spelled
                as declared and read back case-insensitively, with the
                LAST occurrence of a repeated key winning, all three
                exactly as idDict does them,
        'inherit': lowercase parent entityDef name, or None,
    }
    Returns an empty dict if root_dir is blank or doesn't exist.
    """
    entity_defs = {}
    roots = _decl_scan_roots(root_dir)
    if not roots:
        return entity_defs
    if len(roots) > 1:
        # One root at a time, merged with setdefault so the FIRST root to
        # declare a name keeps it. Per-root rather than one merged walk
        # because that leaves each root's own duplicate policy exactly as
        # it was; "inherit" is resolved later, by resolve_entitydef_keys
        # over this merged dict, so a mod entityDef inheriting a base
        # game one still resolves.
        for one in roots:
            for name, decl in parse_def_files(one).items():
                entity_defs.setdefault(name, decl)
        return entity_defs
    root_dir = roots[0]

    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fn in filenames:
            if not fn.lower().endswith('.def'):
                continue
            tokens = decl_read_file(os.path.join(dirpath, fn))
            if tokens is None:
                continue

            for dtype, raw_name, start, end, _line in decl_scan(tokens):
                if dtype.lower() != DECL_TYPE_ENTITYDEF:
                    continue

                # A body is a flat run of "key" "value" STRING tokens. The
                # lexer has already joined any backslash continuation into
                # the single token the engine sees, which is the whole
                # point: pairing them off by position is only correct once
                # the token stream is.
                keys = IdDict()
                i = start
                while i < end:
                    if tokens[i].type != DECL_TT_STRING:
                        i += 1
                        continue
                    if i + 1 >= end or tokens[i + 1].type != DECL_TT_STRING:
                        i += 1
                        continue
                    # Unconditional: idDict::Set replaces, so the LAST
                    # occurrence of a repeated key is the one the engine
                    # ends up holding. See IdDict.
                    keys[tokens[i].val.strip()] = tokens[i + 1].val.strip()
                    i += 2

                # 'inherit' below is the PLAIN spelling only, kept because
                # callers (and tests) have always been able to read it.
                # It is NOT what resolution uses: idDeclEntityDef::Parse
                # prefix-matches "inherit", so "inherit1"/"inherit2"/
                # "inherits" are inheritance sources too and a def can have
                # several parents. resolve_entitydef_keys scans `keys` for
                # the prefix itself — see its docstring. `keys` therefore
                # has to stay in FILE ORDER, which the insertion-ordered
                # dict built above gives for free; do not sort it.
                inherit = keys.get('inherit')
                entity_defs[raw_name.strip().lower()] = {
                    'keys': keys,
                    'inherit': inherit.strip().lower() if inherit else None,
                }
    return entity_defs


# idTech4's "model NAME { ... }" decl giving the actual
# renderable-model data (mesh path, per-anim identifiers) that an
# entityDef's "model" spawnarg usually just names by reference rather
# than embedding directly. Only relevant for MD5 (skeletal) models: a
# .map entity's "model" key never names an .md5mesh file directly the
# way it can for .ase/.lwo. Like entityDef, a model decl can "inherit"
# another one — see parse_model_decls' second pass for how that
# chain gets resolved.
#
# Unlike an entityDef body (a flat set of quoted "key" "value" spawnarg
# pairs), a model decl body is idTech4's own decl-script syntax — bare,
# usually-unquoted keyword/path tokens, e.g.:
#     inherit npc_base
#     mesh    models/md5/monsters/zombie/zombie.md5mesh
#     anim    idle   models/md5/monsters/zombie/idle.md5anim
#     channel torso  ( "*Torso" )
# decl_lex's bare-word class deliberately EXCLUDES { } ( ) — real content
# routinely closes a "channel" bone list with no space before the ')'
# (e.g. "...eyecontrol chair)"), and a bare-word pattern that swallowed
# it (a plain \S+ does) glues it onto the preceding word as one token,
# so the balanced-paren skip below never sees a standalone ')' to
# close on — it then silently consumes every token for the rest of the
# decl body (every later "anim" line included) hunting for one that
# never comes. Splitting these characters out always, even with no
# surrounding whitespace, is what makes that skip actually terminate.


def parse_model_decls(root_dir):
    """
    Scan every .def file under root_dir for "model NAME { ... }"
    declarations (see the model-decl note above for why this is a
    separate pass from parse_def_files' entityDef scan), then resolve
    each one's own "inherit NAME" line (a bare-keyword decl-level
    inherit, distinct from — but resolved exactly the same way as —
    entityDef's quoted "inherit" spawnarg key; see
    resolve_entitydef_keys). Real idTech4 content routinely chains
    these several levels deep — e.g. a per-map character decl
    inheriting a shared "suit" decl, which itself inherits a base
    skeleton decl that supplies most of the actual "anim" lines.

    Returns a dict: lowercase decl name -> {
        'mesh':  the FULLY RESOLVED "mesh" path (str) — the decl's own
                 if it set one, else whatever its inherit chain
                 supplies (nearest ancestor wins), else None if
                 nothing in the chain ever set one (e.g. a .ase/.lwo-
                 only, or sound-only, decl — not every "model" decl is
                 an MD5 model),
        'anims': {lowercase anim identifier: path, ...}, the FULLY
                 MERGED result of every "anim <identifier> <path>"
                 line anywhere in the inherit chain — loaded root-most
                 ancestor first, so a decl's own anim entries always
                 override an inherited one using the same identifier,
                 while every non-overridden inherited identifier is
                 still present.
        'skin':  the FULLY RESOLVED "skin" path (str), same precedence
                 as 'mesh' — the decl's own if it set one, else whatever
                 its inherit chain supplies (nearest ancestor wins),
                 else None. This is idTech4's "default skin": real
                 engine code (idEntity::UpdateModel in Entity.cpp) only
                 ever consults it when the ENTITY itself has no "skin"
                 spawnarg at all (not directly, not inherited via its
                 entityDef) — see the "skin" resolution in
                 import_map_generator, which checks entity/entityDef
                 first and only falls back to this field if both are
                 empty, exactly matching that precedence.
    }
    Returns an empty dict if root_dir is blank or doesn't exist.
    """
    roots = _decl_scan_roots(root_dir)
    if not roots:
        return {}

    # Raw, UN-resolved decls as literally written — 'inherit' resolution
    # happens in a second pass below, once every decl in every file has
    # been read (a decl can inherit one declared in a completely
    # different .def file, so this can't be resolved file-by-file).
    raw_decls = {}   # lowercase name -> {'inherit':, 'mesh':, 'anims': {}}

    for _scan_root in roots:
        # One dict per root, merged with setdefault below: that leaves each
        # root's own duplicate policy (last declaration in the tree wins)
        # exactly as it was, while ACROSS roots the highest-priority one
        # wins outright - a mod redeclaring a model decl replaces the base
        # game's rather than being overwritten by whichever file the walk
        # happened to reach last.
        root_decls = {}
        for dirpath, _dirnames, filenames in os.walk(_scan_root):
            for fn in filenames:
                if not fn.lower().endswith('.def'):
                    continue
                file_tokens = decl_read_file(os.path.join(dirpath, fn))
                if file_tokens is None:
                    continue

                for dtype, raw_name, start, end, _line in decl_scan(file_tokens):
                    if dtype.lower() != DECL_TYPE_MODEL:
                        continue

                    # The body walk below reads plain strings, and a
                    # DeclToken's .val is exactly what the old body
                    # tokeniser produced: quoted strings with their quotes
                    # off, lone { } ( ), and bare words.
                    tokens = [t.val for t in file_tokens[start:end]]

                    inherit_name = None
                    mesh_path = None
                    skin_path = None
                    anims = {}
                    i, ntok = 0, len(tokens)
                    while i < ntok:
                        tok = tokens[i]
                        if tok == '(':
                            # Skip a balanced parenthesised group, e.g. a
                            # "channel NAME ( boneglob ... )" bone list —
                            # nothing inside it is a mesh/anim reference.
                            pdepth = 1
                            i += 1
                            while i < ntok and pdepth > 0:
                                if tokens[i] == '(':
                                    pdepth += 1
                                elif tokens[i] == ')':
                                    pdepth -= 1
                                i += 1
                            continue
                        if tok == '{':
                            # Skip a balanced braced group, e.g. the
                            # per-frame sound-event block real content
                            # routinely attaches straight after an anim's
                            # path ("anim walk path.md5anim { frame 3
                            # sound_body snd_footstep ... }") — depth-
                            # tracked the same way as the decl's own outer
                            # braces above, so nothing inside is mistaken
                            # for a top-level mesh/anim/inherit line.
                            bdepth = 1
                            i += 1
                            while i < ntok and bdepth > 0:
                                if tokens[i] == '{':
                                    bdepth += 1
                                elif tokens[i] == '}':
                                    bdepth -= 1
                                i += 1
                            continue
                        if tok in ('}', ')'):
                            i += 1
                            continue
                        if tok == 'inherit' and i + 1 < ntok:
                            inherit_name = tokens[i + 1]
                            i += 2
                            continue
                        if tok == 'mesh' and i + 1 < ntok:
                            mesh_path = tokens[i + 1]
                            i += 2
                            continue
                        if tok == 'skin' and i + 1 < ntok:
                            # A model decl's own "default skin" line — see
                            # the 'skin' entry in this function's docstring
                            # and the fallback precedence in
                            # import_map_generator. Real content routinely
                            # sets this on a base character decl (e.g.
                            # "model npc_security { skin skins/characters/
                            # npcs/regsec.skin ... }") so every entityDef/
                            # entity that places it via that decl — directly
                            # or through a further "inherit" chain of model
                            # decls — wears it without needing its own
                            # "skin" spawnarg at all.
                            skin_path = tokens[i + 1]
                            i += 2
                            continue
                        if tok == 'anim' and i + 2 < ntok:
                            # Some anim lines give several comma-separated
                            # alternate paths (e.g. "path1.md5anim,
                            # path2.md5anim" — a real random-variant/blend
                            # pattern in idTech4 content); only the first
                            # is usable here, so trailing alternates (and
                            # the comma itself) are dropped.
                            anim_path = tokens[i + 2].split(',', 1)[0].strip()
                            anims[tokens[i + 1].strip().lower()] = anim_path
                            i += 3
                            continue
                        i += 1

                    root_decls[_norm_slashes(raw_name.strip()).lower()] = {
                        'inherit': _norm_slashes(inherit_name.strip()).lower() if inherit_name else None,
                        'mesh': mesh_path,
                        'skin': skin_path,
                        'anims': anims,
                    }
        for _name, _decl in root_decls.items():
            raw_decls.setdefault(_name, _decl)

    # Second pass: resolve each decl's "inherit" chain into the fully
    # merged form callers actually want — root-most ancestor loaded
    # first, so a decl's own "mesh"/"anim" entries always win over
    # whatever they inherited (same precedence as
    # resolve_entitydef_keys). Memoized in `resolved` since the same
    # ancestor (e.g. a shared base skeleton decl) is typically inherited
    # by many other decls.
    resolved = {}

    def resolve(name, seen):
        if name in resolved:
            return resolved[name]
        raw = raw_decls.get(name)
        if raw is None:
            return None
        if name in seen:
            # Cycle in "inherit" (malformed content) — stop climbing,
            # use what this decl set on itself rather than recursing
            # forever.
            return {'mesh': raw['mesh'], 'skin': raw['skin'], 'anims': dict(raw['anims'])}

        mesh  = raw['mesh']
        skin  = raw['skin']
        anims = dict(raw['anims'])
        if raw['inherit']:
            parent = resolve(raw['inherit'], seen | {name})
            if parent is not None:
                if mesh is None:
                    mesh = parent['mesh']
                if skin is None:
                    skin = parent['skin']
                merged_anims = dict(parent['anims'])
                merged_anims.update(anims)   # this decl's own entries win
                anims = merged_anims

        result = {'mesh': mesh, 'skin': skin, 'anims': anims}
        resolved[name] = result
        return result

    for name in raw_decls:
        resolve(name, set())

    return resolved


def _spawnarg_bool(value):
    """idTech4's own idDict::GetBool semantics (verified against the
    Doom 3 BFG GPL source, idlib/Dict.cpp): out = atoi(value) != 0 — NOT
    a literal "1"-string check. Any leading nonzero integer is true
    ("1", "2", "-1", " 1 trailing junk", ...); a missing/empty value, a
    literal "0", or anything with no leading integer at all is false.
    Used for spawnargs like "hide" that idTech4 treats as booleans."""
    if not value:
        return False
    m = re.match(r'\s*([+-]?\d+)', value)
    return bool(m) and int(m.group(1)) != 0


# ─────────────────────────────────────────────────────────────────────
#  .SKIN FILE PARSING (per-instance material remapping)
# ─────────────────────────────────────────────────────────────────────
# idTech4's "skin" spawnarg lets one .map entity swap out specific
# materials on the model it places, WITHOUT affecting any other entity
# placing that same model elsewhere in the map (e.g. re-skinning one
# particular soda machine prop to a different label texture). The swap
# table itself lives in a "skin NAME { old new  old new  ... }" decl
# under a "skins" folder — see resolve_skins_root — using the same bare,
# unquoted decl-script syntax as a "model NAME { ... }" decl, so this
# reads the same DECL LEXER token stream.
#
# The leading "skin" type keyword is required here, as it always has
# been. A .skin file gives idDeclFile a defaultType, so the engine also
# accepts a bare "NAME { ... }" with no type word at all — no shipped
# .skin in the five corpora is written that way, and accepting it would
# mean treating any unrecognised top-level word as a decl type.


# ===========================================================================
# BEGIN DECL CANONICAL
#
# idDeclManagerLocal::MakeNameCanonical (framework/DeclManager.cpp:1557),
# byte-identical in all five engines this toolchain reads: Doom 3, Doom 3
# BFG, Quake 4 / Prey, The Dark Mod and the Q3E ports.
#
# THE FOUR ADDONS INSTALL INDEPENDENTLY. Any one of them may be present
# without the others, so this block is duplicated VERBATIM into each rather
# than imported from one - a cross-addon import would turn "installed
# alongside" into a hard dependency. tests/decl_parity.py pulls the block out
# of all four files by these banners and fails unless the copies are
# byte-identical, so an edit to one is a failing test until it is an edit to
# all four.
#
# NOTHING BETWEEN THE BEGIN/END BANNERS MAY TOUCH bpy OR IMPORT ANYTHING.


def engine_canonical_decl(name):
    """The name a decl is registered and looked up under.

    Runs on BOTH sides in the engine - registration (CreateNewDecl,
    FindTypeWithoutParsing) and every lookup - so canonicalising only one
    side fixes nothing. Three rules, in one pass:

        backslashes become forward slashes
        everything else is lowercased
        the name is truncated at the LAST dot anywhere in the string

    That last rule is the engine's, not a convenience. A decl path with a
    dot in a FOLDER name loses everything after it, which is a real and
    silent way to lose a material; the .ase/.lwo export validator flags it.

    This is NOT the rule for image file paths. Those get
    BackSlashesToSlashes plus "remove .tga anywhere"
    (renderer/Image_init.cpp:1495) and are correct already - do not route
    them through here.
    """
    out = []
    last_dot = -1
    for i, c in enumerate(name or ''):
        if c == '\\':
            out.append('/')
        elif c == '.':
            last_dot = i
            out.append(c)
        else:
            out.append(c.lower())
    if last_dot != -1:
        return ''.join(out[:last_dot])
    return ''.join(out)


# END DECL CANONICAL
# ===========================================================================


def _norm_slashes(s):
    """Normalize path separators for STRING COMPARISON/lookup purposes
    (dict keys, decl names, spawnarg values) — NOT filesystem access,
    which os.path already handles fine either way on Windows. idTech4
    content mixes '\\' and '/' interchangeably (e.g. a .map file's
    "skin" spawnarg authored with one, the matching "skin NAME { }"
    decl name in a .def/.skin file authored with the other — a real
    case that silently failed to match before this existed), so every
    place that keys/compares a path-shaped keyword needs both forms
    treated as identical, not just os.path.join/isfile calls."""
    return s.replace('\\', '/')


def _decl_display_name(name):
    """The DISPLAY twin of engine_canonical_decl: slashes and the last-dot
    rule, but the case left as written.

    engine_canonical_decl is the identity and is what every key and every
    lookup goes through — a "skin" spawnarg like "skins/sodamachinesq.skin"
    matches a decl named "skins/sodamachinesq" because both sides run
    through it. This one is for the single place a name is not being
    compared but WRITTEN: the replacement material a skin pair names, which
    becomes a Blender datablock the user reads. Lowercasing that would
    rename roughly a thousand materials to fix nothing.

    Mirrors idTech4_material_import.py's strip_material_extension, which
    draws the same line for the same reason."""
    name = _norm_slashes(name.strip())
    idx = name.rfind('.')
    return name[:idx] if idx != -1 else name


def parse_skin_files(root_dir):
    """
    Scan every .skin file under root_dir for "skin NAME { ... }"
    declarations. Each body is a flat list of bare
    "<old material> <new material>" pairs — idTech4's per-instance
    material-swap table, referenced by a .map entity's "skin" spawnarg.
    An optional "model <path>" line restricts which base model the skin
    is meant for; it's tokenised (so it doesn't get mistaken for a
    material pair) but not otherwise interpreted, since the entity's own
    placed model already tells us which mesh the swap applies to.

    Returns a dict: canonical skin name -> {
        canonical old-material-name: new-material-name (original case), ...
    }
    Returns an empty dict if root_dir is blank or doesn't exist.
    """
    skins = {}
    roots = _decl_scan_roots(root_dir)
    if not roots:
        return skins
    if len(roots) > 1:
        # First root to declare a skin name keeps it — see parse_def_files.
        for one in roots:
            for name, table in parse_skin_files(one).items():
                skins.setdefault(name, table)
        return skins
    root_dir = roots[0]

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Sorted, because the duplicate rule below is "the first file to
        # declare a name keeps it" and os.walk's order is the filesystem's.
        dirnames.sort()
        for fn in sorted(filenames):
            if not fn.lower().endswith('.skin'):
                continue
            file_tokens = decl_read_file(os.path.join(dirpath, fn))
            if file_tokens is None:
                continue

            # One dict per FILE. Within a file idDeclFile::LoadAndParse
            # re-points the existing decl at the later text (sourceFile ==
            # this, so it falls through the "previously defined" warning),
            # so the last declaration wins; ACROSS files that warning fires
            # and `continue`s, so the first file wins. Both, exactly.
            file_skins = {}

            for dtype, raw_name, start, end, _line in decl_scan(file_tokens):
                if dtype.lower() != DECL_TYPE_SKIN:
                    continue

                # Plain strings, as parse_model_decls does — the body walk
                # below compares token text, not token types.
                tokens = [t.val for t in file_tokens[start:end]]

                pairs = {}
                i, ntok = 0, len(tokens)
                while i < ntok:
                    tok = tokens[i]
                    if tok in ('{', '}', '(', ')'):
                        i += 1
                        continue
                    if tok.lower() == 'model' and i + 1 < ntok:
                        # Restricts applicability to a specific base
                        # model — not a material pair, skip its value.
                        i += 2
                        continue
                    if i + 1 < ntok and tokens[i + 1] not in ('{', '}'):
                        # Key canonical, value verbatim: the key is
                        # matched against a mesh's material name, the
                        # value becomes a datablock the user reads.
                        pairs[engine_canonical_decl(tok)] = (
                            _decl_display_name(tokens[i + 1]))
                        i += 2
                        continue
                    i += 1

                file_skins[engine_canonical_decl(raw_name)] = pairs

            for name, table in file_skins.items():
                skins.setdefault(name, table)

    return skins


def model_shading_overrides(model_addon, filepath, lwo_first_layer_only,
                            source_path, import_materials,
                            model_shading='ENGINE'):
    """(renderBump, unsmoothedTangents) decl sets for one placed model.

    Both keywords make the engine discard the model file's own normals -
    renderBump because R_DeriveTangents is documented to "completely ignore
    any explict normals on surfaces with a renderbump command", and
    unsmoothedTangents because R_DeriveUnsmoothedTangents takes each vertex
    normal from a single dominant triangle instead of averaging. Doom 3 alone
    ships 759 materials with the first and 178 with the second, so a placed
    prop whose material carries either was being shaded from normals the
    engine never uses.

    The lookup has to happen BEFORE the mesh is built: these change the
    normals themselves, not anything applied afterwards. The standalone model
    importer has always done this; this importer went straight to
    load_model_meshes without it.

    Gated on import_materials to match the standalone importer, which only
    consults the .mtr when it was asked to deal with materials at all, and
    on Engine Shading for the same reason it is there: the other three
    modes are not claiming to be the engine, so what the engine would have
    thrown away is none of their business.
    """
    if model_shading != 'ENGINE':
        return (set(), set())
    if not (import_materials and source_path and model_addon is not None):
        return (set(), set())
    names_fn = getattr(model_addon, 'model_material_names', None)
    overrides_fn = getattr(model_addon, 'material_shading_overrides', None)
    if names_fn is None or overrides_fn is None:
        return (set(), set())
    try:
        names = names_fn(filepath, lwo_first_layer_only)
        return overrides_fn(names, source_path)
    except Exception:
        return (set(), set())


def apply_skin_material_swap(mesh_objs, skin_pairs, mat_cache):
    """
    Apply one .skin file's material-swap pairs to ONE entity's placed
    model instance only.

    A model referenced by several entities shares its Mesh datablock's
    material slots (see model_mesh_cache / md5_mesh_cache and
    get_or_create_material) — real Blender instancing, so a straight
    mesh.materials[i] = new_mat here would repaint every OTHER entity
    placing the same model too. Blender's per-slot 'link' field is what
    keeps the swap scoped to just this entity: switching a slot from
    'DATA' (the default, shared via the Mesh) to 'OBJECT' stores that
    slot's material as an override on the Object itself, leaving the
    shared Mesh — and every other Object instancing it — untouched.
    Only creates a new material datablock when one by that name doesn't
    already exist (get_or_create_material / mat_cache already guarantee
    that; see its docstring).
    """
    if not skin_pairs:
        return
    for obj in mesh_objs:
        mesh = obj.data
        for slot_idx, mat in enumerate(mesh.materials):
            if mat is None:
                continue
            new_name = skin_pairs.get(engine_canonical_decl(mat.name))
            if not new_name:
                continue
            slot = obj.material_slots[slot_idx]
            slot.link = 'OBJECT'
            slot.material = get_or_create_material(new_name, mat_cache)


def resolve_entitydef_keys(entity_defs, classname, max_depth=32):
    """
    Return *classname*'s entityDef keys fully merged with every
    entityDef it inherits from — reproducing idDeclEntityDef::Parse
    (framework/DeclEntityDef.cpp) exactly.

    The engine does NOT look up a key called "inherit". It does:

        while ( 1 ) {
            kv = dict.MatchPrefix( "inherit", NULL );   // Icmpn: PREFIX
            if ( !kv ) break;
            defList.Append( FindType( DECL_ENTITYDEF, kv->GetValue() ) );
            dict.Delete( kv->GetKey() );
        }
        for ( i = 0; i < defList.Num(); i++ )
            dict.SetDefaults( &defList[i]->dict );      // only fills ABSENT keys

    Three consequences, each of which real content depends on:

    1. EVERY key whose name begins with "inherit" is an inheritance
       source — "inherit", "inherit1", "inherit2", Prey's "inherits",
       and Quake 4's typo'd "inherity" (which the prefix match honours,
       so an inherit/inherit1/inherit2 allowlist would silently drop
       it). Doom 3 and The Dark Mod use only the plain spelling, which
       is why a single exact lookup looked correct for years; Quake 4
       has 60 entityDefs whose ONLY inherit key is a numbered one — the
       character base defs — and Prey has 43 using "inherits".

    2. A def can therefore have SEVERAL parents, so this is a DAG walk,
       not a chain walk.

    3. FILE ORDER decides conflicts, not the numeric suffix.
       MatchPrefix scans the dict in insertion order and SetDefaults
       only fills keys that are absent, so the inherit key written
       FIRST in the file wins. char_marine_combat writes
       "inherit2" "ai_tactical" BEFORE "inherit1" "char_marine_base",
       so ai_tactical wins — and 94 of Quake 4's 1804 entityDefs
       resolve differently under file order than under suffix order.
       parse_def_files builds 'keys' with an insertion-ordered dict, so
       preserving file order here just means not sorting.

    The engine also DELETES the inherit keys from the resolved dict, so
    they are stripped from the result too.

    Returns an empty dict if classname has no matching entityDef at
    all. Cycles in the inherit graph (malformed data) are guarded by
    max_depth and an on-path visited set rather than looping forever.
    """
    def resolve(name, depth, on_path):
        edef = entity_defs.get(name)
        if edef is None or depth >= max_depth:
            return IdDict()
        keys = edef['keys']
        # IdDict, not dict: the merged result is what every caller reads
        # spawnargs out of, and idDict's case-insensitive access has to
        # survive the merge or it only ever applied to the raw parse.
        merged = IdDict(keys)
        # Insertion order == file order; see point 3 above.
        parents = [k for k in keys if k[:7].lower() == 'inherit']
        for pk in parents:
            parent_name = (keys[pk] or '').strip().lower()
            if not parent_name or parent_name in on_path:
                continue
            # SetDefaults semantics: an ancestor never overwrites a key
            # the child (or an earlier-listed parent) already set.
            for k, v in resolve(parent_name,
                                depth + 1,
                                on_path | {parent_name}).items():
                merged.setdefault(k, v)
        for pk in parents:
            merged.pop(pk, None)          # dict.Delete( kv->GetKey() )
        return merged

    root = (classname or '').strip().lower()
    if not root:
        return IdDict()
    return resolve(root, 0, {root})


def resolve_entitydef_key(entity_defs, classname, key, max_depth=32):
    """
    Look up a single *key* for *classname*'s entityDef, following the
    "inherit" chain — see resolve_entitydef_keys for the actual
    inherit-chain merge this resolves against. Returns None if
    classname has no matching entityDef at all, or if the key is never
    set anywhere in its inheritance chain.
    """
    return resolve_entitydef_keys(entity_defs, classname, max_depth).get(key)



def angles_to_matrix(pitch, yaw, roll):
    """
    idAngles::ToMat3() (idlib/math/Angles.cpp), converted to Blender's
    column-vector convention.

    idTech4 transforms points as row vectors (v * M), Blender as column
    vectors (M @ v), so the idMat3 whose ROWS are the rotated basis
    vectors becomes its own transpose here — which is exactly what
    every other rotation in this file already does (see
    entity_rotation_matrix's "rotation" branch). Keeping the conversion
    in one place matters because the same idAngles convention is read
    from several different spawnargs: an entity's own "angles", an
    attachment entityDef's "angles"/"angles_<pos>", and The Dark Mod's
    "attach_pos_angles_*" — see resolve_attachment_specs.

    pitch/yaw/roll are in degrees, in idTech4's own key order.
    """
    pitch, yaw, roll = math.radians(pitch), math.radians(yaw), math.radians(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw),   math.cos(yaw)
    sr, cr = math.sin(roll),  math.cos(roll)
    row0 = Vector((cp * cy,                 cp * sy,               -sp))
    row1 = Vector((sr * sp * cy - cr * sy,   sr * sp * sy + cr * cy, sr * cp))
    row2 = Vector((cr * sp * cy + sr * sy,   cr * sp * sy - sr * cy, cr * cp))
    return Matrix((row0, row1, row2)).transposed().to_4x4()


def _spawnarg_vec3(value, default=(0.0, 0.0, 0.0)):
    """idDict::GetVector semantics, tolerant of the formatting real
    content uses: three numbers separated by whitespace and/or commas,
    optionally wrapped in parens/brackets by an editor. Anything that
    doesn't yield at least three numbers falls back to *default* — the
    same "missing key means zero" behaviour idTech4 itself has, and
    also what a key that isn't a vector at all (e.g. The Dark Mod's
    per-head offset spawnarg lookup in resolve_attachment_specs, which
    is keyed by a NAME that may or may not also be a vector spawnarg)
    must degrade to."""
    if not value:
        return tuple(default)
    n = [float(x) for x in re.findall(_NUM, value)]
    if len(n) < 3:
        return tuple(default)
    return (n[0], n[1], n[2])


def entity_rotation_matrix(entity):
    """
    Build a rotation-only 4x4 matrix from an idTech4 entity's placement
    keys. Checked in priority order (matches DoomEdit/idTech4 conventions):

      "rotation"  9 numbers, row-major 3x3 matrix. Each row is a world-space
                  basis vector: world = local.x*row0 + local.y*row1 + local.z*row2.
      "angles"    "pitch yaw roll" in degrees (idAngles::ToMat3 convention).
      "angle"     single yaw value in degrees, rotation around Z.

    Numbers are extracted with a regex rather than a plain .split() so
    values wrapped in parentheses/brackets by some editors are still read
    correctly.

    Some idTech4 tools leave a placeholder identity "rotation" key on an
    entity that was actually rotated with the simple yaw handle (which only
    writes "angle"). If "rotation" is numerically identity AND a non-zero
    "angle" is also present, "angle" is trusted instead — otherwise the
    identity "rotation" would silently win and the entity would render
    unrotated.

    Falls back to the identity matrix if none of these keys are present.
    """
    def nums(key):
        raw = entity.keys.get(key, '')
        if not raw:
            return []
        return [float(x) for x in re.findall(_NUM, raw)]

    rot_vals = nums('rotation')

    angle_val = None
    angle_nums = nums('angle')
    if angle_nums:
        angle_val = angle_nums[0]

    if len(rot_vals) >= 9:
        v = rot_vals[:9]
        is_identity = (
            abs(v[0] - 1.0) < 1e-4 and abs(v[1]) < 1e-4 and abs(v[2]) < 1e-4 and
            abs(v[3]) < 1e-4 and abs(v[4] - 1.0) < 1e-4 and abs(v[5]) < 1e-4 and
            abs(v[6]) < 1e-4 and abs(v[7]) < 1e-4 and abs(v[8] - 1.0) < 1e-4
        )
        if not (is_identity and angle_val not in (None, 0.0)):
            row0 = Vector(v[0:3])
            row1 = Vector(v[3:6])
            row2 = Vector(v[6:9])
            return Matrix((row0, row1, row2)).transposed().to_4x4()

    ang_vals = nums('angles')
    if len(ang_vals) >= 3:
        return angles_to_matrix(*ang_vals[:3])

    if angle_val is not None:
        return Matrix.Rotation(math.radians(angle_val), 4, 'Z')

    return Matrix.Identity(4)


# ─────────────────────────────────────────────────────────────────────
#  MATERIAL CACHE
# ─────────────────────────────────────────────────────────────────────

def get_or_create_material(name, cache):
    if name in cache:
        return cache[name]
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        # Flat preview, matching what the material addon gives every
        # material it builds: these are wall/floor/decal surfaces, and a
        # sphere is the wrong shape to judge one on. Most of these
        # placeholders are replaced outright by the material pass later in
        # the import (which sets the same thing) — but the ones it cannot
        # resolve survive as-is, and those are exactly the materials
        # somebody will be looking at in the outliner afterwards.
        mat.preview_render_type = 'FLAT'
    cache[name] = mat
    return mat


# ─────────────────────────────────────────────────────────────────────
#  MD5 MODEL INSTANCING  (via the companion idTech4_MD5_Tools addon)
# ─────────────────────────────────────────────────────────────────────

def _get_or_build_md5_instance(md5_addon, cache, resolved_path, scale, collection,
                               base_name, arm_obj=None, hide_sink=None):
    """
    Returns (arm_obj, [mesh_obj, ...]) for one .map entity's placement
    of an MD5 model — an Armature (so every entity can carry its own
    pose/animation independently of every other entity placing the
    same model), plus one Object per mesh piece in the file.

    *arm_obj*, if given, is used directly instead of building a fresh
    one here — the caller is expected to have already built it via the
    batched armature pre-pass in import_map_generator (see
    md5_prebuilt_armatures there, and _new_armature_object() /
    _populate_armature_bones() / _finish_armature_object() in
    idTech4_MD5_Tools.py): building an Armature one at a time, inline,
    means paying bpy.ops.object.mode_set()'s cost — directly measured
    at ~500ms+ once a scene already has a few thousand objects, vs
    ~1.5ms on a near-empty one — TWICE per call, for every single MD5
    entity. When arm_obj is None (no pre-pass covered this entity —
    e.g. a decl the main loop resolves differently than the pre-pass's
    simplified duplicate check), a fresh Armature is still built here
    exactly as before, so this always produces a correct placement
    either way, just not always the fast one.

    The underlying md5mesh PARSE result and every mesh piece's
    bpy.types.Mesh datablock are cached per resolved file path (in
    *cache*, keyed by resolved_path) and reused — real Blender
    instancing, one Mesh shared by many Objects — by every later entity
    that references the same file, rather than re-parsing the file and
    building duplicate mesh data. Only the Armature (and the thin
    Object/vertex-groups/modifier wrapper around each shared Mesh) is
    ever built fresh per call: vertex groups are recreated in the exact
    same joint-name/order as the cached template so the shared Mesh's
    per-vertex group-index weights still land on the right bones no
    matter which entity's Armature they end up deforming against — the
    same pattern idTech4_MD5_Tools.build_mesh itself uses, just spread
    across more than one Object.

    Raises on the first parse/build failure for a given resolved_path;
    the caller is expected to catch that, report it, and is responsible
    for caching the failure (as None) so a broken reference used by
    many entities is only ever attempted, and logged, once — mirroring
    how the .ase/.lwo model_mesh_cache handles a cached failure.
    """
    # Map-import placements are for visual/spatial reference, not
    # rigging work — the bone shapes just clutter the viewport (and
    # every instance of a repeated NPC adds another full skeleton's
    # worth). Viewport-hide only (not hide_render): reversible with
    # the outliner's eye icon or Alt+H, same as any other manually-
    # hidden object. The standalone MD5 mesh importer (idTech4_MD5_Tools.py's
    # own File > Import > MD5 Mesh) is unaffected — this only
    # runs in the .map import path. Already done by the batched pre-
    # pass for a pre-built arm_obj; only needed here for the fallback
    # (build-it-now) case.
    built_arm_here = arm_obj is None

    cached = cache.get(resolved_path)
    if cached is not None:
        md5, mesh_datas = cached
        if arm_obj is None:
            arm_obj = md5_addon.build_armature(md5, base_name, scale, collection, rot_mat=None)
        if built_arm_here:
            _hide_now_or_later(arm_obj, hide_sink)
        mesh_objs = []
        for me in mesh_datas:
            # Named after this entity + the shared mesh piece, not just
            # the mesh piece alone — same reasoning as the .ase/.lwo
            # instancing above: many entities can share this exact Mesh
            # datablock, and base_name (the entity's own already-unique
            # name) keeps every instance's Object name unique too,
            # without relying on Blender's auto ".001"/".002" uniquify.
            ob = bpy.data.objects.new(f"{base_name}_{me.name}", me)
            collection.objects.link(ob)
            for j in md5['joints']:
                ob.vertex_groups.new(name=j['name'])
            mod        = ob.modifiers.new("Armature", 'ARMATURE')
            mod.object = arm_obj
            mesh_objs.append(ob)
        return arm_obj, mesh_objs

    # First time this file is seen: parse it once, then build every
    # mesh piece for real (vertex groups + weights + Armature modifier
    # all set up by build_mesh itself), bound to THIS entity's Armature.
    # The resulting Mesh datablocks are what every later entity shares.
    md5 = md5_addon.parse_md5mesh(resolved_path)
    if arm_obj is None:
        arm_obj = md5_addon.build_armature(md5, base_name, scale, collection, rot_mat=None)
    if built_arm_here:
        _hide_now_or_later(arm_obj, hide_sink)
    mesh_objs = []
    for i, mesh_data in enumerate(md5['meshes']):
        # smooth=True to match how every other entity-placed model
        # (.ase/.lwo) is always shade-smoothed on placement, below.
        ob = md5_addon.build_mesh(md5, i, mesh_data, base_name, scale,
                                  arm_obj, collection, rot_mat=None,
                                  smooth=True)
        mesh_objs.append(ob)
    cache[resolved_path] = (md5, [ob.data for ob in mesh_objs])
    return arm_obj, mesh_objs


def choose_decl_anim(decl_anims, requested, owner_keys):
    """
    Pick which of a "model NAME { anim ID path ... }" decl's anims to
    apply, following the same priority idTech4 itself uses to find a
    default anim for a placed model.

    decl_anims   the decl's fully inherit-merged {identifier: path}
                 (parse_model_decls' 'anims').
    requested    an explicit "anim" spawnarg value, or '' - from the .map
                 entity for a body model, from the attached entityDef for
                 a head/attachment (see resolve_attachment_specs).
    owner_keys   the inherit-resolved entityDef keys of whatever owns the
                 model, consulted only for the numbered-"animN" fallback
                 below.

    Returns (anim_id, anim_rel, explicit, had_animN, used_af_pose):
        anim_id      display name of what was chosen (or last tried)
        anim_rel     the .md5anim path, or None if nothing resolved
        explicit     a "anim" value was supplied - so a None anim_rel is
                     a real mismatch, not just an absent default
        had_animN    the owner declares the cinematic "animN" pattern
                     described below - likewise makes a None anim_rel a
                     real mismatch
        used_af_pose the last-resort "af_pose" fallback was taken

    An explicit "anim" always wins. With none, the decl's own defaults
    are tried in order: "idle" first, then a numbered "idle1".."idle9"
    variant (some models never define a bare "idle" but do define
    numbered ones; the lowest digit wins), then "initial". None of those
    existing is not a mistake worth flagging on its own - plenty of MD5
    models (static props, non-idle-looping ones) have no default anim.

    Failing all that, real content routinely uses "anim" as a group
    LABEL rather than an identifier, for idTech4's "cinematic" NPC
    pattern: an entityDef declaring "num_cinematics" alongside numbered
    "anim1".."animN" keys (e.g. "reception" -> anim1="reception_a" ..
    anim6="reception_f"), where the game's script logic picks one of the
    variants at runtime - not something a static Blender import can
    meaningfully replicate. The owner's lowest-numbered "animN" that
    DOES match a decl identifier stands in as a representative pose.

    The absolute last resort is "af_pose", idTech4's articulated-figure
    rest pose - near-universally present (inherited from npc_base by
    almost every character decl) since it doubles as the AF ragdoll
    setup pose, and a far better placeholder than leaving the model in
    its raw MD5 bind pose with nothing applied at all.
    """
    anim_id  = (requested or '').strip()
    explicit = bool(anim_id)
    anim_rel = None

    if anim_id:
        anim_rel = decl_anims.get(anim_id.lower())
    else:
        anim_id  = 'idle'
        anim_rel = decl_anims.get('idle')
        if not anim_rel:
            idle_variants = sorted(
                (k for k in decl_anims if re.fullmatch(r'idle[0-9]', k)),
                key=lambda k: int(k[4:]))
            if idle_variants:
                anim_id  = idle_variants[0]
                anim_rel = decl_anims[anim_id]
        if not anim_rel:
            anim_id  = 'initial'
            anim_rel = decl_anims.get('initial')

    had_animN = False
    if not anim_rel:
        numbered = sorted(
            (k for k in owner_keys if re.fullmatch(r'anim\d+', k)),
            key=lambda k: int(k[4:]))
        had_animN = bool(numbered)
        for k in numbered:
            fallback_id  = owner_keys[k].strip()
            fallback_rel = decl_anims.get(fallback_id.lower())
            if fallback_rel:
                anim_id  = '{0}="{1}"'.format(k, fallback_id)
                anim_rel = fallback_rel
                break

    used_af_pose = False
    if not anim_rel:
        fallback_rel = decl_anims.get('af_pose')
        if fallback_rel:
            anim_id      = 'af_pose'
            anim_rel     = fallback_rel
            used_af_pose = True

    return anim_id, anim_rel, explicit, had_animN, used_af_pose


def _get_or_apply_md5_anim(md5_addon, cache, resolved_anim_path, arm_obj, scale,
                           first_frame_only=False):
    """
    Apply the .md5anim at resolved_anim_path onto arm_obj, sharing one
    baked Action across every entity that resolves to the same anim
    file, instead of re-parsing and re-baking a full (often many-
    hundred-frame) Action for every single entity — profiled directly
    against real content as by far the most expensive step in MD5 map
    placement (~110ms to bake a 797-frame anim, vs ~20ms to build an
    entire Armature). This is safe because every armature built for
    the same .md5mesh (see _get_or_build_md5_instance) has an
    IDENTICAL rest pose, so the baked keyframe VALUES for a given anim
    file are the same no matter which entity's armature ends up
    carrying them — only the ASSIGNMENT has to happen per entity, not
    the bake.

    *cache* is keyed by resolved_anim_path -> (parsed anim dict, baked
    Action). Skeleton compatibility is still re-checked against every
    new arm_obj before reusing a cached Action — cheap (a name/
    hierarchy comparison, not a re-bake) — falling back to a fresh
    bake for just this arm_obj if it somehow doesn't match (should
    never happen in practice, since every MD5 model instance placed
    via one .map import shares its skeleton's exact shape with every
    other instance of the same .md5mesh, but cheap enough to guard
    against regardless of how confident that assumption is).

    With *first_frame_only*, no Action is built or assigned at all: the
    armature is posed at frame 0 and left there (md5_addon.
    build_frame_pose). The Action-sharing argument above does not apply -
    a pose is per-Object state, so each entity gets its own regardless -
    but the PARSE is still worth caching, and still is.

    Raises on the first parse/bake failure for a given
    resolved_anim_path; the caller is responsible for caching that
    failure, mirroring how _get_or_build_md5_instance's own cache
    handles a cached failure.
    """
    cached = cache.get(resolved_anim_path)
    if cached is not None:
        anim, action = cached
        ok, _msg = md5_addon.check_skeleton_compatibility(anim, arm_obj)
        if ok:
            if first_frame_only:
                # Nothing to assign - the pose IS the result, and it is
                # per-Object, so every entity sharing this parsed anim gets
                # its own copy of it for free.
                md5_addon.build_frame_pose(anim, arm_obj, 0, scale=scale,
                                           rot_mat=None)
                return
            if arm_obj.animation_data is None:
                arm_obj.animation_data_create()
            arm_obj.animation_data.action = action
            # _md5_get_or_create_slot is a "private" (leading-
            # underscore) helper in idTech4_MD5_Tools.py, reached into
            # directly here rather than reimplemented — it's the exact
            # same slot-assignment logic build_action() itself uses,
            # and it's version-aware (Blender 4.4+ slotted actions vs.
            # older plain actions), which is worth reusing correctly
            # rather than duplicating.
            md5_addon._md5_get_or_create_slot(arm_obj, action)
            return
            # Falls through to a fresh bake below only if compatibility
            # unexpectedly fails for this specific armature.

    anim = md5_addon.parse_md5anim(resolved_anim_path)
    ok, msg = md5_addon.check_skeleton_compatibility(anim, arm_obj)
    if not ok:
        raise ValueError(f"Skeleton mismatch:\n{msg}")
    if first_frame_only:
        # Cache the PARSED anim with no Action beside it, so later entities
        # using the same file skip the re-parse and go straight to posing.
        # (None here is the second half of the tuple, not the cached-failure
        # sentinel the caller stores - that is a bare None in the slot.)
        md5_addon.build_frame_pose(anim, arm_obj, 0, scale=scale, rot_mat=None)
        cache[resolved_anim_path] = (anim, None)
        return
    # Named after the anim file, with no per-entity prefix — unlike a
    # normal import_md5anim() call, this Action is deliberately meant
    # to be one shared, canonical resource, not an entity-specific
    # one. A per-cache-entry index is still appended to keep the name
    # unique across resolved paths: idTech4 content routinely has
    # multiple differently-located anim files sharing a basename (e.g.
    # "stand.md5anim" under several different chars/ subfolders), and
    # build_action() removes any existing same-named Action before
    # creating a new one — without the suffix, baking the second one
    # would delete the first one's still-cached (and possibly still
    # assigned) Action out from under it.
    base_name = os.path.splitext(os.path.basename(resolved_anim_path))[0]
    action_name = f"{base_name}.{len(cache):03d}"
    action = md5_addon.build_action(anim, arm_obj, action_name, scale=scale, rot_mat=None, afu=True)
    cache[resolved_anim_path] = (anim, action)


# ─────────────────────────────────────────────────────────────────────
#  HEADS & ATTACHMENTS  ("def_head" / "def_attach*")
# ─────────────────────────────────────────────────────────────────────
# An idTech4 character is almost never a single model. The body is one
# skeletal (MD5) model; its HEAD is a second, independent model bound to
# a joint of the body's skeleton, and anything the character carries or
# wears (weapon, PDA, pauldron, hat, lantern, chair...) is a third,
# fourth, ... model bound the same way. None of that lives in the .map
# file - the map only places the body - so an importer that reads only
# the .map plus the body's "model" decl produces headless characters
# holding nothing.
#
# The extra models are declared on the character's entityDef, in the
# .def files this module already parses (see parse_def_files /
# parse_model_decls / resolve_entitydef_keys):
#
#   "def_head"     "head_bloodybald"      the head model
#   "head_joint"   "Shoulders"            which body joint it binds to
#                                         ("joint_head" in Quake 4)
#   "def_attach1"  "prop_pistol"          an attached entityDef ...
#   "def_attach2"  "prop_soft_desk_chair2"     ... and another
#
# Ported from the real engine code rather than guessed at:
#   idActor::SetupHead                       (game/Actor.cpp)
#   idAFEntity_WithAttachedHead::SetupHead   (game/AFEntity.cpp)
#   idActor::Spawn's "def_attach" MatchPrefix loop, idActor::Attach
#                                            (game/Actor.cpp)
#   idEntity::GetMasterPosition              (game/Entity.cpp - the bind)
# plus Quake 4's rewritten SetupHead pair (Raven bind the head to the
# joint ORIENTATED instead of composing an origin by hand), and The Dark
# Mod's reworked versions of the same three:
#   idActor::SetupHead, idEntity::ParseAttachmentSpawnargs,
#   idEntity::ParseAttachPositions, idAnimatedEntity::Attach
#
# The engines differ in ways that matter for placement, and all of them
# are in this importer's asset corpus, so all of them are implemented -
# see resolve_attachment_specs for exactly where they diverge and how
# each divergence is detected from the content itself, never from a
# user-selected "game" setting (nothing else in this importer needs one).

_ATTACH_MAX_DEPTH = 4     # body -> head -> hat -> ...


def _attach_lookup_model(name, model_decls, defkeys_fn, decl_first):
    """
    Resolve one "def_head"/"def_attach" VALUE to the model it actually
    places, plus that model's default skin and anim identifier.

    The two engines name two different things here, and the same
    importer has to accept both:

      Doom 3 heads             "def_head" "head_bloodybald" names a
                               "model NAME { mesh ... }" DECL directly
                               (idAFAttachment::SetBody is handed the
                               string as a model name). Prey and the
                               BFG/PS4 port inherit that unchanged.
      Quake 4 heads            "def_head" "char_marinehead_helmet_medic"
                               names an ENTITYDEF, which Quake 4's
                               SetupHead spawns and whose own "model"
                               spawnarg it then hands to SetBody.
      The Dark Mod heads       "def_head" "atdm:ai_head_builderguard"
                               names an ENTITYDEF, whose own "model"
                               spawnarg then names the model decl (TDM's
                               SetupHead calls FindEntityDef() first,
                               then args.GetString("model")).
      every "def_attach"       always an entityDef, in every engine
                               (idActor::Spawn hands the value to
                               SpawnEntityDef as a "classname").

    Which name wins when BOTH exist is what *decl_first* selects, and it
    is not academic: Doom 3 and Quake 4 both declare a model decl and an
    entityDef under one name routinely (Quake 4's heads, e.g.
    "char_marinehead_default" in char_marineheads.def; Doom 3's props,
    e.g. "prop_pistol" in npcs.def). A DOOM 3 head must take the decl, a
    QUAKE 4 head must take the entityDef - which loses that head's
    "skin" if it takes the decl instead - and an attachment must take
    the entityDef in every engine, since taking the decl silently loses
    the "joint"/"origin"/"angles" spawnargs that only ever live on the
    entityDef and the attachment ends up with nowhere to bind. Whichever
    is asked for, the other still serves as a fallback when the
    preferred one does not exist.

    Returns (model, skin, anim, keys):
        model  the model this places - a model-decl name or an .ase/.lwo
               path, exactly as written; None if nothing resolved.
        skin   default "skin" spawnarg, or None.
        anim   default "anim" spawnarg (an anim IDENTIFIER, not a path),
               or None.
        keys   the resolved entityDef keys behind it ({} when the value
               named a model decl directly) - needed by the caller for
               the "joint"/"origin"/"angles" attachment spawnargs and
               for nested "def_attach" on the attached entity itself.
    """
    lookup = _norm_slashes(name).strip().lower()
    if decl_first and lookup in model_decls:
        return name, None, None, {}

    keys = defkeys_fn(name)
    if not keys:
        return (name, None, None, {}) if lookup in model_decls else (None, None, None, {})
    model = (keys.get('model') or '').strip() or None
    skin  = (keys.get('skin')  or '').strip() or None
    anim  = (keys.get('anim')  or '').strip() or None
    return model, skin, anim, keys


def _resolve_copy_joints(body_keys, head_keys, q4_head):
    """
    The "copy_joint" list for one head: joints of the BODY whose
    animation the engine copies into the HEAD's own skeleton every
    frame, so the head bends with the neck instead of riding the bind
    joint rigidly.

    A head model is exported from the same rig as its body and keeps a
    few of the body's joints inside it for exactly this - Doom 3's heads
    keep "neckcontrol"/"headcontrol"/"eyecontrol", Quake 4's keep
    "shoulders"/"neckcontrol"/"legs_channel" - and without the copy they
    just sit at their bind transform. Nothing about that is optional
    dressing: it is the difference between a head that turns and nods
    with the body's animation and one welded to the chest.

    Both engines do the same thing and spell it two different ways
    round, so both shapes are read here:

      Doom 3 (also Prey, the BFG/PS4 port and The Dark Mod)
          idActor::Spawn walks MatchPrefix("copy_joint") over the BODY:
              "copy_joint neckcontrol"        "neckcontrol"
              "copy_joint_world eyecontrol"   "eyecontrol"
          the KEY names the body joint, the VALUE the head joint.
      Quake 4
          idAFAttachment::InitCopyJoints walks the same prefix over the
          HEAD, whose keys are numbered slots:
              "copy_joint1"                   "neckcontrol"
          the VALUE names the head joint, and the body joint is the
          BODY's "copy_joint <value>" (or "copy_joint_world <value>")
          if it declares one, else that same name again.

    An empty value clears an inherited slot in both - Doom 3's character
    defs do that 76 times over with "copy_joint_world eyecontrol" ""
    alone - and is skipped, not copied.

    Returns a list of (body_joint, head_joint, world) tuples, in the
    order the engine builds them; *world* selects the engine's
    JOINTMOD_WORLD_OVERRIDE, which overrides the head joint in world
    space rather than relative to its parent joint.
    """
    pairs = []
    taken = set()

    def add(from_j, to_j, world):
        if not from_j or not to_j or to_j.lower() in taken:
            return
        taken.add(to_j.lower())
        pairs.append((from_j, to_j, world))

    if not q4_head:
        # The key carries the body joint. Sorted only so the list comes
        # out the same way run to run; the engine's dict order is not
        # something a .map importer can or should reproduce.
        for k in sorted(body_keys):
            lk = k.lower()
            to_j = (body_keys.get(k) or '').strip()
            if not to_j:
                continue
            if lk.startswith('copy_joint_world '):
                add(k[len('copy_joint_world '):].strip(), to_j, True)
            elif lk.startswith('copy_joint '):
                add(k[len('copy_joint '):].strip(), to_j, False)
        return pairs

    # Quake 4: the head's numbered slots name head joints, and the body
    # optionally remaps each to a differently-named joint of its own.
    # idDict lookups are case-insensitive, so the remap is matched that
    # way rather than by exact key.
    def body_remap(prefix, name):
        want = (prefix + ' ' + name).lower()
        for k, v in body_keys.items():
            if k.lower() == want:
                return (v or '').strip()
        return None

    for k in sorted((k for k in head_keys
                     if k.lower().startswith('copy_joint') and ' ' not in k),
                    key=lambda k: (len(k), k)):
        to_j = (head_keys.get(k) or '').strip()
        if not to_j:
            continue
        from_j = body_remap('copy_joint_world', to_j)
        if from_j is not None:
            # The key exists, so this one is a world override - but an
            # empty value leaves the engine looking up joint "", which
            # is the INVALID_JOINT it warns about and skips.
            add(from_j, to_j, True)
            continue
        from_j = body_remap('copy_joint', to_j)
        add(to_j if from_j is None else from_j, to_j, False)
    return pairs


def resolve_attachment_specs(merged_keys, classname, model_decls, defkeys_fn):
    """
    Build the flat, engine-faithful placement list for one character's
    head and attachments, from *merged_keys* - the entity's own .map
    spawnargs layered over its classname's fully inherit-resolved
    entityDef keys (the entity's own keys win, exactly as idGameLocal
    layers them at spawn).

    Returns a list of spec dicts, PARENTS BEFORE CHILDREN so a caller
    can place them in order and always find the parent's armature
    already built (The Dark Mod routinely attaches a hat to a head,
    which is itself attached to the body - see _ATTACH_MAX_DEPTH):

      'kind'     'head' | 'attach'
      'key'      the spawnarg it came from ('def_head', 'def_attach2')
      'value'    that spawnarg's value, for messages
      'parent'   index into this list of the model it binds to, or None
                 meaning "binds to the body"
      'joint'    joint (bone) name ON THE PARENT's skeleton
      'model'    model-decl name or .ase/.lwo path, or None
      'skin'     default skin, or None
      'anim'     default anim identifier, or None
      'keys'     resolved entityDef keys of the attached entity ({} if
                 it named a model decl directly)
      'origin'   (x, y, z) offset in idTech4 units, in 'origin_frame'
      'angles'   (pitch, yaw, roll) degrees, applied BEFORE the joint's
                 own orientation (idTech4's "rotate * axis")
      'sub_origin' / 'sub_angles'
                 The Dark Mod's per-entity refinement of a SHARED
                 attachment position ("origin_<posname>" /
                 "angles_<posname>" on the attached entityDef); zero
                 everywhere else
      'origin_frame'
                 'entity' - 'origin' is in the parent MODEL's own frame
                            (Doom 3: "originOffset * renderEntity.axis")
                 'joint'  - 'origin' is in the JOINT's frame (The Dark
                            Mod: "originOffset * axis", axis being the
                            joint's world axis)
      'orient'   True  - take the joint's orientation (every
                         attachment, and a QUAKE 4 head:
                         BindToJoint(..., orientated=true) with an
                         identity local axis)
                 False - keep the body's orientation and ignore the
                         joint's (a DOOM 3 / Prey / BFG / Dark Mod head:
                         SetupHead sets the head entity's axis to
                         renderEntity.axis, not to the joint axis)
      'copy_joints'
                 (body_joint, head_joint, world) triples the engine
                 copies the body's animation through every frame, so a
                 head bends with the neck - see _resolve_copy_joints.
                 Empty for everything that is not a head.

    Nothing here touches bpy - it is pure spawnarg resolution, so the
    .map importer's batched armature pre-pass and its main placement
    loop can both call it and be guaranteed to agree on what exists.
    """
    specs = []

    def add_level(keys, parent_idx, depth, seen):
        if depth > _ATTACH_MAX_DEPTH:
            return

        # -- the head ------------------------------------------------
        head_def = (keys.get('def_head') or '').strip()
        if head_def and head_def != '-' and head_def.lower() not in seen:
            # WHICH SPELLING the character uses is also which ENGINE's
            # SetupHead it gets, and the two do not place a head the same
            # way. "head_joint" is Doom 3's (and Prey's, the BFG/PS4
            # port's and The Dark Mod's); "joint_head" is Quake 4's -
            # Raven renamed the spawnarg along with rewriting the
            # function. No def in the corpus spells both, and no other
            # signal separates the two engines' characters, so the key
            # that supplied the joint selects the placement rules below.
            joint = (keys.get('head_joint') or '').strip()
            q4_head = False
            if not joint:
                joint = (keys.get('joint_head') or '').strip()
                q4_head = bool(joint)

            # Quake 4 SPAWNS the head entityDef and takes ITS "model"
            # (idActor::SetupHead: headEnt->SetBody( this,
            # headEnt->spawnArgs.GetString( "model" ), joint )), where
            # Doom 3 hands the "def_head" string straight to SetBody as a
            # model name. Both games routinely declare a model decl AND
            # an entityDef under the one name, so this is not academic:
            # Quake 4's "char_marinehead_helmet_medic" is both, and only
            # the entityDef carries the "skin" that makes it a MEDIC's
            # helmet rather than a plain one.
            model, skin, anim, hkeys = _attach_lookup_model(
                head_def, model_decls, defkeys_fn, decl_first=not q4_head)

            if q4_head:
                # Quake 4, idActor::SetupHead (and the identical tail of
                # idAFEntity_WithAttachedHead::SetupHead):
                #     headEnt->BindToJoint( this, joint, true );
                #     headEnt->GetPhysics()->SetOrigin( vec3_origin + headOffset );
                #     headEnt->GetPhysics()->SetAxis( mat3_identity );
                # An ORIENTATED bind with an identity local axis: the
                # head takes the joint's whole transform, rotation
                # included. Its models are authored for exactly that -
                # char_marineheads.def exports every head with
                # "-rename chest origin -clearorigin", which expresses
                # the head in the body's CHEST JOINT's own frame, and
                # ships copies of the body's shoulders/neckcontrol/
                # legs_channel/head_channel joints inside the head model
                # for InitCopyJoints to drive. Placing one the Doom 3 way
                # (joint position, body axis) lays that chain out
                # horizontally and the head ends up out by the hands.
                # "headOffset" is zero for anything a .map spawns - only
                # idPlayer passes one, from the playerModel decl - and
                # Quake 4 reads no "offsetModel" here at all.
                off = (0.0, 0.0, 0.0)
            else:
                # Doom 3, idActor::SetupHead:
                #     origin = renderEntity.origin
                #            + ( jointOrigin + modelOffset ) * renderEntity.axis
                #     headEnt->SetAxis( renderEntity.axis )
                # so relative to the BODY MODEL the head sits at
                # jointOrigin + modelOffset, unrotated by the joint.
                # ("offsetModel" also shifts the body's own render origin
                # by that same vector - see
                # idActor::GetPhysicsToVisualTransform - which is why
                # in-game it ends up counted twice from the entity's
                # physics origin. This importer never applies it to the
                # body, so adding it once here reproduces the
                # body-to-head spacing the engine actually renders.) The
                # Dark Mod adds two further terms: a per-head offset
                # looked up by the head def's own NAME, and a blanket
                # "offsetHeadModel".
                off = _spawnarg_vec3(keys.get('offsetModel'))
                for extra in (keys.get(head_def), keys.get('offsetHeadModel')):
                    ex, ey, ez = _spawnarg_vec3(extra)
                    off = (off[0] + ex, off[1] + ey, off[2] + ez)

            specs.append({
                'kind': 'head', 'key': 'def_head', 'value': head_def,
                'parent': parent_idx, 'joint': joint,
                'model': model, 'skin': skin, 'anim': anim, 'keys': hkeys,
                'origin': off, 'angles': (0.0, 0.0, 0.0),
                'sub_origin': (0.0, 0.0, 0.0), 'sub_angles': (0.0, 0.0, 0.0),
                # Quake 4's SetOrigin is a local origin under an
                # orientated bind, so it would be in the JOINT's frame -
                # it is only ever zero here, but say which frame it is
                # zero in.
                'origin_frame': 'joint' if q4_head else 'entity',
                'orient': q4_head,
                'copy_joints': _resolve_copy_joints(keys, hkeys, q4_head),
            })
            if hkeys:
                add_level(hkeys, len(specs) - 1, depth + 1, seen | {head_def.lower()})

        # -- attachments ---------------------------------------------
        # idDict::MatchPrefix("def_attach") - "def_attach", "def_attach1",
        # "def_attach2", ... Sorted so numbered slots come out in their
        # natural order rather than dict order, purely so this importer's
        # messages and object names stay stable run to run.
        attach_keys = sorted((k for k in keys if k.startswith('def_attach')),
                             key=lambda k: (len(k), k))
        for akey in attach_keys:
            aval = (keys.get(akey) or '').strip()
            # The Dark Mod uses "-" to CLEAR an inherited attachment slot
            # (ParseAttachmentSpawnargs skips those explicitly); an empty
            # value clears it the same way in both engines.
            if not aval or aval == '-' or aval.lower() in seen:
                continue

            model, skin, anim, akeys = _attach_lookup_model(
                aval, model_decls, defkeys_fn, decl_first=False)

            suffix       = akey[len('def_attach'):]
            joint        = ''
            origin       = (0.0, 0.0, 0.0)
            angles       = (0.0, 0.0, 0.0)
            sub_origin   = (0.0, 0.0, 0.0)
            sub_angles   = (0.0, 0.0, 0.0)
            origin_frame = 'entity'

            # -- The Dark Mod's named attachment POSITIONS ------------
            # "pos_attach<N>" names a position; the position itself is
            # declared on the CHARACTER (usually inherited from a shared
            # skeleton base def) as a set of "attach_pos_*_<KEY>"
            # spawnargs sharing one arbitrary KEY suffix, of which
            # "attach_pos_name_<KEY>" holds the name being matched. That
            # indirection is what lets a single "hand_r" position,
            # calibrated once on tdm_ai_humanoid, serve every weapon
            # every humanoid AI ever holds. See ParseAttachPositions.
            pos_name = (keys.get('pos_attach' + suffix) or '').strip()
            if pos_name:
                for k2, v2 in keys.items():
                    if not k2.startswith('attach_pos_name_'):
                        continue
                    if (v2 or '').strip() != pos_name:
                        continue
                    pk = k2[len('attach_pos_name_'):]
                    joint      = (keys.get('attach_pos_joint_' + pk) or '').strip()
                    origin     = _spawnarg_vec3(keys.get('attach_pos_origin_' + pk))
                    angles     = _spawnarg_vec3(keys.get('attach_pos_angles_' + pk))
                    # idAnimatedEntity::Attach's per-entity refinement of
                    # a shared position, read off the ATTACHED entity.
                    sub_origin = _spawnarg_vec3(akeys.get('origin_' + pos_name))
                    sub_angles = _spawnarg_vec3(akeys.get('angles_' + pos_name))
                    # TDM rotates a position's offset by the JOINT's axis
                    # where Doom 3 rotates the attached entity's own
                    # "origin" by the CHARACTER's axis - the one real
                    # placement difference between the two systems, and
                    # the reason origin_frame exists. The TDM offsets are
                    # authored for exactly that: tdm_ai_humanoid.def
                    # annotates "attach_pos_origin_handr" with the joint's
                    # own axis order, "down sideways forward".
                    origin_frame = 'joint'
                    break

            # -- the original per-entity system ----------------------
            # Doom 3's only system, and still TDM's fallback: the
            # ATTACHED entityDef carries its own "joint"/"origin"/
            # "angles". TDM additionally allows a body-classname-specific
            # override of the latter two (idAnimatedEntity::Attach's
            # "angles_<classname>" / "origin_<classname>") and accepts
            # "bindToJoint" as a second spelling of "joint".
            if not joint:
                joint = ((akeys.get('joint') or '').strip() or
                         (akeys.get('bindToJoint') or '').strip())
                cls_ang = akeys.get('angles_' + (classname or ''))
                cls_org = akeys.get('origin_' + (classname or ''))
                angles = _spawnarg_vec3(cls_ang if cls_ang is not None else akeys.get('angles'))
                origin = _spawnarg_vec3(cls_org if cls_org is not None else akeys.get('origin'))
                origin_frame = 'entity'

            specs.append({
                'kind': 'attach', 'key': akey, 'value': aval,
                'parent': parent_idx, 'joint': joint,
                'model': model, 'skin': skin, 'anim': anim, 'keys': akeys,
                'origin': origin, 'angles': angles,
                'sub_origin': sub_origin, 'sub_angles': sub_angles,
                'origin_frame': origin_frame, 'orient': True,
                # Only a head is ever a copy_joint target: both engines
                # read the list off the head/body pair alone.
                'copy_joints': (),
            })
            if akeys:
                add_level(akeys, len(specs) - 1, depth + 1, seen | {aval.lower()})

    add_level(merged_keys, None, 1, frozenset())
    return specs


def entity_has_attachment_keys(entity_keys, def_keys):
    """Cheap gate in front of resolve_attachment_specs: is there any
    "def_head"/"def_attach*" spawnarg at all, on the entity itself or
    anywhere in its entityDef inherit chain? The overwhelming majority
    of a map's entities are lights, brush entities and props with none,
    and this keeps them from paying for a merged-dict build apiece."""
    for k in entity_keys:
        if k.startswith('def_head') or k.startswith('def_attach'):
            return True
    for k in def_keys:
        if k.startswith('def_head') or k.startswith('def_attach'):
            return True
    return False


def _find_bone(arm_obj, joint_name):
    """Bone lookup by MD5 joint name. Exact first, then case-
    insensitive: .def files and .md5mesh files get authored by different
    people at different times and routinely disagree on the
    capitalisation of one joint, which the engine doesn't care about
    either - idAnimator::GetJointHandle compares with idStr::Icmp."""
    if not joint_name:
        return None
    bone = arm_obj.data.bones.get(joint_name)
    if bone is not None:
        return bone
    lowered = joint_name.lower()
    for b in arm_obj.data.bones:
        if b.name.lower() == lowered:
            return b
    return None


def attachment_local_matrix(spec, joint_matrix, scale):
    """
    The attached model's transform in its PARENT MODEL's own local space
    - i.e. everything the engine does between the parent's render
    transform and the attachment's, with the parent's own world
    placement factored back out.

    Doom 3, idActor::Attach:
        GetJointWorldTransform( joint, origin, axis )
        ent->SetOrigin( origin + originOffset * renderEntity.axis )
        ent->SetAxis( angleOffset.ToMat3() * axis )
    Doom 3, idActor::SetupHead (and idAFEntity_WithAttachedHead's):
        origin = renderEntity.origin + (jointOrigin + modelOffset) * renderEntity.axis
        headEnt->SetAxis( renderEntity.axis )       <- joint axis NOT used
    Quake 4, idActor::SetupHead (and idAFEntity_WithAttachedHead's):
        headEnt->BindToJoint( this, joint, true )   <- orientated bind
        headEnt->GetPhysics()->SetOrigin( vec3_origin + headOffset )
        headEnt->GetPhysics()->SetAxis( mat3_identity )
            headOffset is zero for everything a .map spawns, so this is
            the joint's own transform, joint axis INCLUDED
    The Dark Mod, idAnimatedEntity::Attach:
        ent->SetOrigin( origin + originOffset * axis + originSubOffset * newAxis )
        ent->SetAxis( angleSubOffset.ToMat3() * newAxis )
            where newAxis = angleOffset.ToMat3() * axis

    Factoring the parent's own transform out of all four leaves the
    joint's transform as the only frame involved, which is what this
    composes. idTech4 multiplies row vectors left to right
    (v * A * B) and Blender multiplies column vectors right to left
    (B @ A @ v), so every product below is the engine's written in
    reverse - see angles_to_matrix.

    *joint_matrix* is the joint's transform in the parent model's own
    space, IN THE POSE THE BIND HAPPENS IN - which is the character's
    "ik_pose", not the .md5mesh bind pose: idActor::Spawn does
    animator.SetFrame(ANIMCHANNEL_ALL, GetAnim(IK_ANIM), 0, 0, 0)
    immediately before attaching anything, under a comment saying the IK
    anim has to be set first or attachments do not "bind correctly". See
    joint_bind_matrices in import_map_generator for where that pose
    comes from, and attachment_bind_local_matrix for the rebasing that
    lets a rest-pose-parented Blender object carry the same relationship.

    None means the parent has no skeleton at all - a static (.ase/.lwo)
    attachment carrying an attachment of its own. The Dark Mod's
    idEntity::Attach covers that case by using the parent's own origin
    and axis in place of a joint's, which is this same composition with
    an identity joint, so it falls out of the code below rather than
    needing a second path. (Doom 3 has no equivalent: it reads
    "def_attach" only in idActor::Spawn, so only an actor ever attaches
    anything.)
    """
    if joint_matrix is None:
        joint_pos = Vector((0.0, 0.0, 0.0))
        joint_rot = Matrix.Identity(4)
    else:
        joint_pos = joint_matrix.translation
        joint_rot = joint_matrix.to_3x3().to_4x4()

    offset = to_bl(Vector(spec['origin']), scale)
    if spec['origin_frame'] == 'joint':
        offset = joint_rot @ offset

    if not spec['orient']:
        # A Doom 3 - lineage head: the engine hands it the body's axis,
        # so in the body's own local space it is simply unrotated. (A
        # Quake 4 head takes the joint's axis like an attachment does,
        # and comes through with orient set.)
        return Matrix.Translation(joint_pos + offset)

    rot = joint_rot @ angles_to_matrix(*spec['angles'])

    sub = Vector(spec['sub_origin'])
    if sub.length_squared > 0.0:
        offset = offset + (rot @ to_bl(sub, scale))

    if spec['sub_angles'] != (0.0, 0.0, 0.0):
        rot = rot @ angles_to_matrix(*spec['sub_angles'])

    return Matrix.Translation(joint_pos + offset) @ rot


def attachment_bind_local_matrix(spec, joint_rest, joint_bind, scale):
    """
    The local matrix to hand attach_parent_to, given the joint's REST
    transform (what Blender parents to) and the transform it had in the
    pose the ENGINE bound in (its "ik_pose").

    The engine stores an attachment's offset relative to the JOINT, once,
    at bind time - idPhysics_Static::SetMaster takes the world transform
    it was just given and expresses it in the joint's frame, and
    idEntity::GetMasterPosition replays it against the joint's current
    transform every frame after. Blender bone parenting does exactly the
    same thing, except that the pose it captures the relationship in is
    whatever the armature is in when the parent is assigned - for an
    import, the .md5mesh bind pose.

    Those two poses are not the same, and the difference is not
    academic: on Doom 3's security NPC the right hand rotates 36.7
    degrees between its bind pose and its "ik_pose", which moves an
    attachment carrying the PDAs' "origin" "5 -6.5 -4.5" by 4.98 units -
    a hand's width, visible as a PDA floating beside the hand rather
    than in it. Joints the arms don't move (the "Shoulders" every Doom 3
    head hangs off, the "chair" a seated NPC sits on) have a zero delta,
    which is why only hand-held props with a non-zero offset look wrong.

    So: compose the engine's transform in the bind pose (where its
    numbers were authored), take it into the joint's frame there, and
    put it back down in the rest pose Blender will parent in. With the
    two poses equal - or no "ik_pose" declared, or a joint missing from
    it - this collapses to composing in the rest pose directly, exactly
    as before.
    """
    if joint_bind is None:
        return attachment_local_matrix(spec, joint_rest, scale)
    local_bind = attachment_local_matrix(spec, joint_bind, scale)
    rest = joint_rest if joint_rest is not None else Matrix.Identity(4)
    return rest @ joint_bind.inverted() @ local_bind


def attach_parent_to(obj, parent_obj, bone, local_matrix):
    """
    Bind *obj* to *parent_obj* so that its world transform is
    parent_obj.matrix_world @ local_matrix.

    With *bone* given, this is bone parenting (see bone_parent_to);
    with bone None the parent has no skeleton and plain object
    parenting is what idEntity::Attach amounts to - the child rides the
    parent's own origin and axis. matrix_parent_inverse is the identity
    there because the parent's object transform IS the frame the local
    matrix was composed in.
    """
    if bone is not None:
        bone_parent_to(obj, parent_obj, bone, local_matrix)
        return
    obj.parent = parent_obj
    obj.parent_type = 'OBJECT'
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.matrix_basis = local_matrix


def bone_parent_to(obj, arm_obj, bone, local_matrix):
    """
    Bind *obj* to *bone* of *arm_obj* so that, with the armature in its
    rest pose, obj's world transform is exactly
    arm_obj.matrix_world @ local_matrix - and so that it then FOLLOWS
    that bone through whatever animation the armature carries. This is
    idEntity::BindToJoint(master, joint, orientated=true), which is how
    the engine attaches every head and every attachment.

    Blender evaluates a bone-parented child as

        world = arm.matrix_world
              @ pose_bone.matrix @ Translation((0, bone.length, 0))
              @ matrix_parent_inverse
              @ matrix_basis

    - the bone's TAIL, not its head, is the parenting origin (a
    backwards-compatibility quirk of Blender's; verified directly
    against a running Blender rather than assumed). Handing
    matrix_parent_inverse the inverse of that whole bracket AS OF THE
    REST POSE cancels both the tail shift and the bone's rest transform,
    leaving matrix_basis free to be exactly the local matrix the engine
    math produced. Deriving it algebraically - rather than assigning
    matrix_world and letting Blender solve for the basis - keeps this
    independent of whether the depsgraph has re-evaluated the parent
    yet, which mid-import it has not.
    """
    rest_parent = bone.matrix_local @ Matrix.Translation((0.0, bone.length, 0.0))
    obj.parent      = arm_obj
    obj.parent_type = 'BONE'
    obj.parent_bone = bone.name
    obj.matrix_parent_inverse = rest_parent.inverted()
    obj.matrix_basis = local_matrix


def _bone_rest_local(bone):
    """A bone's rest transform relative to its PARENT bone's rest - the
    frame a "Local Space" constraint is expressed in. None for a root
    bone, which has no parent frame to be relative to."""
    if bone.parent is None:
        return None
    return bone.parent.matrix_local.inverted() @ bone.matrix_local


def apply_copy_joints(head_arm, body_arm, pairs):
    """
    Reproduce idAFAttachment::CopyJointsFromBody (Quake 4) and
    idActor::UpdateAnimation's copyJoints loop (Doom 3 and its
    descendants) as Blender constraints on the HEAD's pose bones.

    A head model is exported from its body's rig and keeps a few of the
    body's joints inside it - Doom 3's heads keep "neckcontrol",
    "headcontrol" and "eyecontrol", Quake 4's keep "shoulders",
    "neckcontrol" and "legs_channel" - and the engine drives them from
    the body's every frame. Without that the head rides its bind joint
    rigidly and does not bend with the neck. What the engine does, after
    animating both skeletons:

        JOINTMOD_WORLD_OVERRIDE
            body->GetJointWorldTransform( from, pos, axis )
            ... re-expressed in the head entity's own frame ...
            animator.SetJointPos/Axis( to, mod, ... )
        JOINTMOD_LOCAL_OVERRIDE
            bodyAnimator->GetJointLocalTransform( from, pos, axis )
            animator.SetJointPos/Axis( to, mod, pos, axis )

    The WORLD one is a Copy Transforms constraint in world space on both
    ends - exact, and needing nothing to be true of either rig.

    The LOCAL one overwrites the head joint's transform RELATIVE TO ITS
    PARENT JOINT with the body joint's, and Blender has no constraint
    space that says that across two armatures: "Local Space" is a bone's
    delta from ITS OWN REST, "Local With Parent" and a custom space were
    both measured against the engine's composition and are no closer,
    and Copy Transforms carries one custom space for both ends where
    this would need two different ones. So a LOCAL/LOCAL Copy Transforms
    it is, which is the engine exactly WHEN THE TWO RIGS AGREE ABOUT
    WHERE THE JOINT RESTS, and this checks that they do rather than
    assuming it.

    Across the whole corpus - 1205 local-override pairs on characters
    whose body and head meshes both ship - 982 agree about the joint's
    rest ORIENTATION, and those are the ones constrained here:

      * rest orientations agree - the copied rotation is the engine's,
        exactly. Any leftover is a difference in the joint's rest
        POSITION, which is the offset the head model already has today
        with no constraint at all, so this can only improve on it. 686
        pairs have no such difference either and are exact outright.
      * rest orientations disagree (223 pairs: The Dark Mod's new
        skeleton driving a head's "neckcontrol" from a body joint called
        "Neck", Quake 4's monster_bossbuddy wearing a marine head on a
        Strogg body) - a delta from one rest frame applied in a
        differently oriented one turns the head about the wrong axis, so
        the pair is SKIPPED and the head keeps the behaviour it has
        without any of this. Better nothing than a head on crooked.

    A joint either engine names but the model does not carry is skipped
    too, as the engine skips it with a warning; Doom 3's character defs
    name "eyecontrol" on bodies whose heads have none.

    Returns (applied, skipped).
    """
    applied = skipped = 0
    for from_j, to_j, world in pairs:
        to_bone   = _find_bone(head_arm, to_j)
        from_bone = _find_bone(body_arm, from_j)
        if to_bone is None or from_bone is None:
            skipped += 1
            continue
        pbone = head_arm.pose.bones.get(to_bone.name)
        if pbone is None:
            skipped += 1
            continue

        if not world:
            to_rest   = _bone_rest_local(to_bone)
            from_rest = _bone_rest_local(from_bone)
            if to_rest is None or from_rest is None:
                skipped += 1
                continue
            if max(abs(a - b)
                   for ra, rb in zip(to_rest.to_3x3(), from_rest.to_3x3())
                   for a, b in zip(ra, rb)) > 1e-3:
                skipped += 1
                continue

        con = pbone.constraints.new('COPY_TRANSFORMS')
        con.name      = f"idTech4 copy_joint {from_bone.name}"
        con.target    = body_arm
        con.subtarget = from_bone.name
        space = 'WORLD' if world else 'LOCAL'
        con.target_space = space
        con.owner_space  = space
        applied += 1
    return applied, skipped


# ─────────────────────────────────────────────────────────────────────
#  ENTITY PROPERTY STORAGE
# ─────────────────────────────────────────────────────────────────────

def store_entity_keys(obj, keys):
    for k, v in keys.items():
        obj[f"map_{k}"] = v
    obj.id_properties_ensure()


# ─────────────────────────────────────────────────────────────────────
#  MAIN IMPORT
# ─────────────────────────────────────────────────────────────────────

def combine_brush_meshes(mesh_mat_pairs, combined_name):
    """
    Combine multiple already-built brush Meshes (each with its own
    materials list, as returned by build_brush_mesh) into ONE new Mesh,
    preserving every vertex from every source mesh with NO welding/merging
    across brush boundaries — each source brush's geometry is appended
    as-is via bmesh.from_mesh, which does not weld coincident vertices on
    its own (that only happens if something explicitly calls
    bmesh.ops.remove_doubles, which this deliberately never does).

    mesh_mat_pairs: list of (mesh, mat_names) tuples.
    Returns (combined_mesh, combined_mat_names).
    """
    combined_mat_names = []
    mat_index_map = {}   # material name -> index in combined_mat_names
    bm = bmesh.new()

    for mesh, mat_names in mesh_mat_pairs:
        local_to_combined = []
        for name in mat_names:
            if name not in mat_index_map:
                mat_index_map[name] = len(combined_mat_names)
                combined_mat_names.append(name)
            local_to_combined.append(mat_index_map[name])

        bm.faces.ensure_lookup_table()
        start = len(bm.faces)
        bm.from_mesh(mesh)          # appends geometry; does not merge/weld
        bm.faces.ensure_lookup_table()
        for f in bm.faces[start:]:
            if 0 <= f.material_index < len(local_to_combined):
                f.material_index = local_to_combined[f.material_index]
            else:
                f.material_index = 0

    combined_mesh = bpy.data.meshes.new(combined_name)
    bm.to_mesh(combined_mesh)
    bm.free()
    combined_mesh.validate(clean_customdata=False)
    combined_mesh.update()
    return combined_mesh, combined_mat_names


def _combine_geo_objects(built_items, combined_name, obj_name, class_col, empty,
                          parent_inverse, mat_cache, all_objects):
    """
    Combine a list of already-built (idx, mesh, mat_names) brush/patch
    tuples (as produced in import_map_generator's per-entity loop) into
    ONE new mesh Object, linked/parented the same way the per-item
    Objects it replaces would have been. Frees the now-orphaned source
    meshes afterward. Returns the new Object.
    """
    combined_mesh, combined_mat_names = combine_brush_meshes(
        [(mesh, mat_names) for (_, mesh, mat_names) in built_items],
        combined_name,
    )
    for mn in combined_mat_names:
        combined_mesh.materials.append(get_or_create_material(mn, mat_cache))

    obj = bpy.data.objects.new(obj_name, combined_mesh)
    class_col.objects.link(obj)
    obj.parent = empty
    obj.matrix_parent_inverse = parent_inverse
    all_objects.append(obj)

    for (_, mesh, _) in built_items:
        bpy.data.meshes.remove(mesh)
    return obj


def _hide_now_or_later(obj, hide_sink):
    """Viewport-hide *obj* — or, if *hide_sink* is a list, put it there
    for the caller to hide once the import is over.

    Object.hide_set() is a VIEW LAYER operation and raises outright on an
    object the view layer cannot see ("Object 'x' cannot be hidden
    because it is not in View Layer 'ViewLayer'"). The .map import parks
    everything it builds outside the view layer while it works — see
    _find_layer_collection and its use in import_map_generator — so
    during that window the hide has to be deferred rather than done.
    """
    if hide_sink is not None:
        hide_sink.append(obj)
        return
    obj.hide_set(True)


# LayerCollections an in-flight import has parked out of their view
# layer (see import_map_generator's "park the import" note). An import
# that runs to completion un-parks its own and leaves this empty; the
# list exists for the two ways it can NOT reach that point — the user
# pressing ESC, and an exception out of the middle of the run — because
# either one would otherwise leave the collection excluded and the
# import looking like it produced nothing at all.
_PARKED_LAYER_COLLECTIONS = []


def _unpark_layer_collections():
    """Put every parked collection back in its view layer. Idempotent,
    a no-op when nothing is parked, and tolerant of a LayerCollection
    that has since gone away with its scene or file."""
    while _PARKED_LAYER_COLLECTIONS:
        layer_col = _PARKED_LAYER_COLLECTIONS.pop()
        try:
            layer_col.exclude = False
        except (ReferenceError, AttributeError, RuntimeError):
            pass


def _find_layer_collection(layer_collection, collection):
    """The LayerCollection wrapping *collection* somewhere under
    *layer_collection*, or None if this view layer has no view of it.

    Matched by IDENTITY, never by name: bpy.data.collections.new()
    uniquifies a clashing name, so importing the same .map twice leaves
    the second import's root called "mymap.001" while
    view_layer.layer_collection.children["mymap"] is still the FIRST
    one — and excluding that would blank the previous import instead of
    the one being built.
    """
    if layer_collection.collection is collection:
        return layer_collection
    for child in layer_collection.children:
        found = _find_layer_collection(child, collection)
        if found is not None:
            return found
    return None


def collapse_outliner_levels(context, levels=1):
    """
    Collapse the Outliner's collection tree by *levels* levels, so
    newly-imported collections show up closed rather than expanded.

    There is deliberately NO list of "collections created by this
    import" passed in here, because there is nothing to do with such a
    list even if one were built: Blender does not expose per-collection
    expand/collapse state as a stable Python property at all —
    bpy.types.Collection and bpy.types.LayerCollection have no
    "show_expanded" (or equivalent) anywhere in their public RNA (the
    full LayerCollection property set is: collection, children, exclude,
    hide_viewport, holdout, indirect_only, is_visible, name — nothing
    UI/expand-state related). The Outliner's own collapse-arrow UI is
    itself implemented via the outliner.show_one_level operator, and
    that operator has no per-collection targeting either: it just
    collapses whatever is currently the SHALLOWEST EXPANDED LEVEL across
    the entire displayed tree, one level per call, with no argument for
    "only these collections" or "only this depth" — it's blind to
    identity, only aware of depth. So tracking exactly which collections
    this import created wouldn't unlock anything: there's no API call
    that would accept that list and act on it selectively. This is
    therefore a global collapse of the Outliner's current tree state,
    called *levels* times to reach collections nested that many levels
    deep — accurate for what it affects only when nothing else in the
    Outliner happens to already be sitting expanded at the same depth as
    what this import just created.

    IMPORTANT — this runs DEFERRED via bpy.app.timers, not immediately.
    Calling outliner.show_one_level() synchronously from inside another
    operator's execute() (as an earlier version of this function did)
    reliably had NO visible effect at all — a known general class of
    Blender scripting problem: an operator invoked from inside another
    operator's execute() often doesn't take effect, or doesn't take
    effect visibly, because Blender's context/UI hasn't "settled" back
    to its normal state yet at that point (the screen can still be
    transitioning away from the file-select dialog, redraws can be
    suppressed mid-operator, etc.). The standard workaround is to
    register a one-shot timer so the actual work runs on the next
    event-loop tick, after this import operator has fully returned
    control to Blender — which is what happens here. Also now applies
    to EVERY Outliner area found across every open window (not just the
    first one), in case more than one is visible at once (e.g. split
    across workspaces).

    Silently does nothing if no window anywhere has an Outliner area
    open at all by the time the timer fires.
    """
    win_mgr = context.window_manager

    def _do_collapse():
        for window in win_mgr.windows:
            for area in window.screen.areas:
                if area.type != 'OUTLINER':
                    continue
                region = next((r for r in area.regions if r.type == 'WINDOW'), None)
                if region is None:
                    continue
                try:
                    with bpy.context.temp_override(window=window, area=area, region=region):
                        for _ in range(max(1, levels)):
                            bpy.ops.outliner.show_one_level(open=False)
                except Exception:
                    pass
        return None   # one-shot: don't reschedule

    bpy.app.timers.register(_do_collapse, first_interval=0.05)


class MapImportReport(object):
    """One .map import's report, kept as the four piles it falls into.

    Two subjects, two kinds of message each. The MAP pile is about this
    .map and the assets its entities name; the MATERIALS pile is about the
    .mtr tree those surfaces were shaded from — usually identical across
    every map in a game base, and as often "that Materials Source isn't the
    one you meant" as it is anything to do with this map at all. Within
    each, the counts say what happened and the issues say what went wrong,
    and the two only mean anything side by side: "75 models failed/missing"
    is a number until the 75 lines naming them are underneath it.

    Holding the four apart, rather than pre-joining them into one block of
    text, is what lets the report window lay them out in that order —
    MAP counts, MAP issues, MATERIALS counts, MATERIALS issues — while the
    status bar still gets the short counts-only string it wants. Interleaved
    the other way (all counts, then all issues), each section's numbers sat
    a screenful away from the lines explaining them.
    """
    __slots__ = ('headline', 'map_lines', 'map_issues',
                 'material_lines', 'material_issues')

    def __init__(self, headline, map_lines=None, map_issues=None,
                 material_lines=None, material_issues=None):
        # "OK: N entities" / "WARNING: ..." / "ERROR: ...". The operator
        # appends the wall-clock duration to it once the import is over.
        self.headline = headline
        self.map_lines = list(map_lines or [])
        self.map_issues = list(map_issues or [])
        self.material_lines = list(material_lines or [])
        self.material_issues = list(material_issues or [])

    @property
    def sections(self):
        """(heading, summary lines, issue lines), in report order."""
        return (('MAP', self.map_lines, self.map_issues),
                ('MATERIALS', self.material_lines, self.material_issues))

    @property
    def issue_count(self):
        return len(self.map_issues) + len(self.material_issues)

    @property
    def result(self):
        """The counts, without the issue lists.

        This is what the operator hands to self.report() and tests for its
        "OK: " / "WARNING: " / "ERROR: " prefix, so it stays exactly what
        the old result string was: a headline plus the per-section
        summaries. The issues are deliberately not in it — a status-bar
        message is no place for seventy-five lines. An early bail-out
        (no entities, a parse error) has no summaries at all, and then this
        is just the headline.
        """
        out = [self.headline]
        for heading, summary, _issues in self.sections:
            if not summary:
                continue
            out.extend(('', heading))
            out.extend(summary)
        return '\n'.join(out)

    def body(self, title):
        """The whole report, as the Text Editor window shows it."""
        out = [title, '=' * len(title), '', self.headline]
        for heading, summary, issues in self.sections:
            if not summary and not issues:
                continue
            out.extend(('', heading))
            out.extend(summary)
            out.append('')
            if issues:
                out.append('  %d issue(s)' % len(issues))
                out.append('  ' + '-' * 38)
                out.extend('  ' + line for line in issues)
            else:
                out.append('  No issues.')
        return '\n'.join(out) + '\n'


def import_map_generator(context, filepath, scale=SCALE, import_models=True,
                          lwo_first_layer_only=True, parse_defs=True,
                          import_md5_models=False, md5_animation_mode='FIRST_FRAME',
                          import_attachments=True,
                          apply_skins=True, import_materials=True, material_mode='SIMPLE',
                          material_parameters='BAKED', model_shading='ENGINE',
                          worldspawn_geo_grouping='ALL', derive_from_model=False,
                          override_base_directory='', override_source_path='',
                          save_derived_as_default=False,
                          import_md5_animations=None):
    """
    Generator version of the .map import pipeline. Identical logic and
    identical final result to the old synchronous import_map(), except
    it yields (ent_idx, total_entities, status_message) after each
    entity is processed, so a modal operator can advance it
    incrementally — one or a few next() calls per Blender timer tick —
    instead of running the whole import in one uninterrupted call.
    Blender is single-threaded for UI/Python: nothing on screen (status
    bar, popups, even the cursor) redraws until control returns to
    Blender's own event loop, so a long synchronous import gives no
    opportunity for any visible progress feedback at all. Yielding here
    lets a modal operator do a small bounded chunk of work, return
    control to Blender so it can process events and redraw, then resume
    on the next timer tick.

    The final return value is a MapImportReport (see its docstring,
    just above): the "OK: N entities" / "ERROR: ..." / "WARNING: ..."
    headline the old synchronous function used to return on its own, plus
    the per-section summary counts, plus two lists of human-readable issue
    strings — one per failure/omission encountered along the way (which
    entity, which specific brush/patch/model, and why). The ok/skip COUNTS
    alone don't say what actually went wrong or where, and a caller that
    only wants the old one-line-ish string can still ask the report for
    its .result.

    Map issues and material issues stay separate all the way through
    because they answer different questions and are fixed in different
    places — see MapImportReport. It comes back via `return report` at the
    end of this function body, which in a generator surfaces as
    StopIteration.value once the generator is exhausted. See import_map()
    below for a synchronous wrapper that drives this to completion in one
    call and extracts that value, for any caller that doesn't need
    progressive updates.
    """
    filename = os.path.splitext(os.path.basename(filepath))[0]

    try:
        parse_diag = MapParseDiagnostics()
        entities = parse_map_file(filepath, diagnostics=parse_diag)
    except MapSyntaxError as exc:
        # A syntax error carries the line it gave up on, which is worth far
        # more to whoever has to fix the file than a Python traceback.
        return MapImportReport("ERROR: %s could not be parsed - %s"
                               % (os.path.basename(filepath), exc))
    except Exception:
        return MapImportReport("ERROR: " + traceback.format_exc())

    if not entities:
        return MapImportReport("WARNING: No entities found.")

    scene      = context.scene
    view_layer = context.view_layer
    mat_cache  = {}
    # Cache of resolved model filepath -> [(name, mesh, [material_name,...]), ...]
    # so multiple entities referencing the same .ase/.lwo file share one set
    # of Mesh datablocks (Blender's usual object-instancing pattern: many
    # Objects, one Mesh) instead of re-parsing the file and building a full
    # duplicate mesh for every single entity.
    model_mesh_cache = {}
    # Same idea for MD5 models — see _get_or_build_md5_instance, which
    # owns this cache's exact contents (resolved .md5mesh path -> (parsed
    # md5 dict, [Mesh datablock, ...])).
    md5_mesh_cache = {}
    # And again for baked MD5 animations — see _get_or_apply_md5_anim,
    # which owns this cache's exact contents (resolved .md5anim path ->
    # (parsed anim dict, baked Action)). Baking a many-hundred-frame
    # Action is by far the most expensive step in MD5 map placement
    # (profiled directly: ~110ms for a 797-frame anim, vs ~20ms to
    # build an entire Armature) — sharing one baked Action across every
    # entity that resolves to the same anim file, instead of re-baking
    # it per entity, is what actually matters for import speed on a map
    # with many repeated NPCs/props.
    md5_anim_cache = {}

    # search_roots is what this whole import reads under — for def/skins/
    # model resolution below AND (further down) the materials post-pass.
    # Resolved once, up front: derive_from_model/override/shared-config
    # priority is exactly _resolve_sources' job (see its docstring), and
    # an optional Mod Base sits in FRONT of the Base Directory, searched
    # first with the base still searched behind it. Every
    # resolve_def_roots/resolve_skins_roots/resolve_model_path call below
    # gets the whole tuple as their "user_root" hint; empty still means
    # "auto-walk up from the .map file's own location" in every one of
    # them, unchanged.
    base_directory, mod_directory, source_path = _resolve_sources(
        filepath, derive_from_model, override_base_directory,
        override_source_path, save_derived_as_default=save_derived_as_default)
    search_roots = shared_search_roots(base_directory, mod_directory)

    # def_roots is needed by BOTH entityDef classname-fallback lookups
    # (gated by parse_defs) and MD5 model-decl lookups (gated by
    # import_md5_models) — resolved once, up front, whenever either
    # feature might need it, rather than duplicating resolve_def_roots'
    # own directory-search work per feature. Plural: with a Mod Base set
    # this is every def/ directory that exists, mod first, because a mod
    # ships a handful of .def files and inherits the rest.
    def_roots = []
    if parse_defs or import_md5_models:
        def_roots = resolve_def_roots(filepath, search_roots)

    # entityDef declarations, scanned once per import (not per entity) —
    # see resolve_def_root / parse_def_files. Only entities with NO
    # explicit "model" spawnarg of their own fall back to whatever their
    # classname's entityDef declares (directly or via "inherit"), exactly
    # matching how idTech4 itself layers entity spawnargs over entityDef
    # defaults. entity_defs stays an empty dict when parse_defs is off,
    # so the fallback lookup below is always a harmless no-op in that case.
    entity_defs = {}
    if parse_defs and def_roots:
        entity_defs = parse_def_files(def_roots)

    # "skin" decls, scanned once per import — see resolve_skins_root /
    # parse_skin_files / apply_skin_material_swap. Only entities with a
    # "skin" spawnarg of their own ever consult this; skin_defs stays an
    # empty dict when apply_skins is off, so every lookup below is a
    # harmless no-op in that case.
    skin_defs = {}
    if apply_skins:
        skins_roots = resolve_skins_roots(filepath, search_roots)
        skin_defs = parse_skin_files(skins_roots)

    root_col = bpy.data.collections.new(filename)
    scene.collection.children.link(root_col)

    # Cache of classname -> collection so we create each only once
    class_cols = {}

    def get_class_col(classname):
        if classname not in class_cols:
            col = bpy.data.collections.new(classname)
            root_col.children.link(col)
            class_cols[classname] = col
        return class_cols[classname]

    all_objects = []
    # Objects to viewport-hide once the import is over. Nothing can be
    # hidden while it is parked outside the view layer (hide_set() is a
    # view layer operation and raises on an object the layer cannot
    # see), so every hide the import wants is collected here and applied
    # in one pass after the reveal. See _hide_now_or_later.
    pending_hide = []
    brush_ok    = 0
    brush_skip  = 0
    patch_ok    = 0
    patch_skip  = 0
    model_ok    = 0
    model_skip  = 0
    # Format breakdown of the model_ok/model_skip totals above, purely
    # for the final report line — every static (.ase/.lwo) success/skip
    # increments both model_ok/model_skip AND static_model_ok/skip; every
    # MD5 success/skip increments both model_ok/model_skip AND
    # md5_model_ok/skip. The one ambiguous case (an entityDef-resolved
    # "model" value that matches neither a recognized static-model file
    # extension nor any parsed MD5 decl name — we can't tell which format
    # was even intended) only increments the combined model_skip, not
    # either breakdown counter.
    static_model_ok   = 0
    static_model_skip = 0
    md5_model_ok      = 0
    md5_model_skip    = 0
    # Heads and attachments (see resolve_attachment_specs) — counted
    # separately from md5_model_ok/skip because they are placed by a
    # different mechanism (bound to a joint of an already-placed body,
    # not to a .map entity's own origin) and because a map full of
    # characters places far more of them than of bodies, which would
    # otherwise swamp the body totals in the report.
    attach_ok         = 0
    attach_skip       = 0
    # Head joints driven from the body's (idAFAttachment::CopyJointsFromBody):
    # [constrained, skipped]. Skips are not failures — see apply_copy_joints —
    # so they get a clause on the heads/attachments line, not an issue each.
    copy_joint_counts = [0, 0]
    # .prt (particle system decl) references are never a model this
    # importer could load regardless of format support — counted on
    # their own so the report can give them one clear summary line
    # instead of an issue per entity and inflating model_skip/
    # static_model_skip with something that was never a "failed" model.
    prt_skip = 0
    # Wall-clock time actually spent inside each format's whole branch
    # (resolve + parse/build + place, and for MD5 also the anim step) —
    # requested for narrowing down where import time goes when the
    # combined model count alone doesn't say which format is actually
    # expensive. Uses time.monotonic() to match modal()'s own tick-
    # budget timing elsewhere in this file, not wall-clock-of-day.
    static_model_time = 0.0
    md5_model_time    = 0.0
    # Finer split of md5_model_time: how much of it is
    # _get_or_build_md5_instance() (Armature + mesh instancing) versus
    # _get_or_apply_md5_anim() (anim resolve/bake/apply) specifically —
    # requested after md5_model_time alone didn't say which of the two
    # very different costs (an Armature build is a fixed per-entity
    # cost; an anim bake scales with joint count * frame count and
    # only happens once per distinct resolved anim file, cached) was
    # actually dominant.
    md5_build_time = 0.0
    md5_anim_time  = 0.0
    # Split of md5_build_time by whether _get_or_build_md5_instance()
    # actually had to parse+build (cache miss — first time this
    # resolved .md5mesh is seen: full parse_md5mesh + build_armature +
    # build_mesh per mesh piece, including any V12 normal
    # reconstruction) versus just instanced an already-cached Mesh
    # (cache hit — a fresh Armature/vertex-groups/modifier only,
    # meant to be cheap). A real import came back with 68 MD5 models
    # costing 16.9s of armature/mesh time (~248ms/model average) —
    # this tells us whether that's dominated by a handful of expensive
    # first-time builds (many distinct, complex character meshes) or
    # by every instance being unexpectedly costly (which would point
    # at the cache-hit path itself, not mesh complexity).
    md5_build_first_time = 0.0
    md5_build_first_count = 0
    md5_build_cached_time = 0.0
    md5_build_cached_count = 0
    # Detailed, human-readable log of every failure/omission encountered
    # during this import — which entity, which specific brush/patch/
    # model, and why. Distinct from the ok/skip COUNTS above (which only
    # feed the one-line summary): this is what the end-of-import report
    # popup shows so a person can actually track down what went wrong,
    # rather than just knowing a number of things were skipped.
    issues = []
    # Material messages are kept in their own list rather than appended to
    # `issues`: see this function's docstring for why the report separates
    # "what this map could not load" from "what the .mtr tree said".
    material_issues = []

    # What the parser knowingly discarded. dmap warns about degenerate brush
    # planes ("brush %d has degenerate plane equations") and this importer
    # used to drop them in silence, leaving a brush with fewer sides than
    # were written and possibly no longer closed. Three exist in the shipped
    # corpus, all in doom3 pdas.map, all written as ( 0 0 0 0 ).
    if parse_diag.degenerate_planes:
        issues.append(
            "%d brush side%s in this map had a degenerate (zero-length) "
            "plane and could not be used; the brushes they belong to are "
            "missing those sides and may not be closed. The map compiler "
            "warns about these too - they are a fault in the .map file, not "
            "in this import."
            % (parse_diag.degenerate_planes,
               "" if parse_diag.degenerate_planes == 1 else "s"))
    for _note in parse_diag.notes:
        issues.append(_note)

    # Static-model (.ase/.lwo) placement is delegated to the companion
    # idTech4_ase_lwo_io addon — see get_model_import_addon() near the
    # top of this file. If the caller asked for it (import_models=True)
    # but that addon isn't installed/enabled, every entity's "model" key
    # is left unresolved below (model_addon is None disables that whole
    # block per-entity) — surfaced here once, up front, rather than
    # repeating the same explanation on every single affected entity.
    model_addon = get_model_import_addon() if import_models else None
    if import_models and model_addon is None:
        issues.append("Static model import (.ase/.lwo) was requested, but the "
                      "\"idTech4_ase_lwo_io\" addon is not installed or not "
                      "enabled — no \"model\" keys were resolved. Install/enable "
                      "it (File > Import > idTech4 .ase / .lwo should appear in "
                      "the menu once it is), or disable \"Import Static Models\" "
                      "to silence this.")

    # MD5 (skeletal) model placement is delegated the same way, to the
    # companion idTech4_MD5_Tools addon — see get_md5_import_addon().
    # The animation mode only ever matters together with import_md5_models
    # (there is nothing to pose without a model's Armature) — folded in
    # once here so every later check only has to look at one flag.
    #
    # import_md5_animations is the old boolean spelling of this argument,
    # still accepted so a script written against the previous signature
    # keeps working: True meant "bake the whole thing", False meant
    # "nothing at all".
    if import_md5_animations is not None:
        md5_animation_mode = 'FULL' if import_md5_animations else 'NONE'
    if not import_md5_models:
        md5_animation_mode = 'NONE'
    md5_first_frame_only  = md5_animation_mode == 'FIRST_FRAME'
    import_md5_animations = md5_animation_mode != 'NONE'
    md5_addon = get_md5_import_addon() if import_md5_models else None
    if import_md5_models and md5_addon is None:
        issues.append("MD5 model import was requested, but the "
                      "\"idTech4_MD5_Tools\" addon is not installed or not "
                      "enabled — no MD5 \"model\" keys were resolved. "
                      "Install/enable it, or disable \"Import MD5 Models\" "
                      "to silence this.")

    # Material creation is delegated the same way, to the companion
    # idTech4_material_import addon — see get_material_import_addon().
    # Only the availability check happens here, up front (matching the
    # static-model/MD5 pattern above); the actual build runs as a
    # separate pass AFTER every entity below has been placed — see the
    # "Material creation (post-pass)" block near the end of this
    # generator for why.
    if import_materials and get_material_import_addon() is None:
        material_issues.append(
            "Material import was requested, but the "
            "\"idTech4_material_import\" addon is not installed or not "
            "enabled — brushes/patches will get blank placeholder materials "
            "instead. Install/enable it, or disable \"Import Materials\" to "
            "silence this.")

    # model decls ("model NAME { mesh ... anim ... }") are a separate
    # decl type from entityDef, scanned once per import — see
    # parse_model_decls. Only ever needed when MD5 models are actually
    # wanted and available; stays an empty dict otherwise, so the
    # per-entity lookup below is always a harmless no-op in that case.
    model_decls = {}
    if import_md5_models and md5_addon is not None and def_roots:
        model_decls = parse_model_decls(def_roots)

    # resolve_entitydef_keys() walks and merges the WHOLE "inherit"
    # chain from scratch on every call — cheap once, but every entity
    # sharing a classname was redoing that same walk/merge again from
    # nothing, and a single entity could trigger it twice over (once
    # for the "model" key fallback below, again for the "animN"
    # cinematic anim fallback further down). Memoized per classname
    # here so the merge only ever happens once per distinct classname
    # for the whole import, no matter how many entities share it or
    # how many separate fallback lookups each one needs.
    entitydef_merged_cache = {}
    def _entitydef_keys_cached(cls):
        ck = (cls or '').strip().lower()
        if ck not in entitydef_merged_cache:
            entitydef_merged_cache[ck] = resolve_entitydef_keys(entity_defs, cls)
        return entitydef_merged_cache[ck]

    # Debug visibility into the "af_pose" last-resort anim fallback —
    # see below. Included in the final result summary.
    af_pose_fallback_count = 0

    # ── MD5 armature batching pre-pass ──────────────────────────────
    # bpy.ops.object.mode_set() — required to build Armature bones —
    # was directly measured to cost ~1.5ms on a near-empty scene but
    # ~500ms+ once the scene already has a few thousand objects (the
    # scale a large .map import reaches well before it's done), and
    # that cost is paid per CALL, not per bone: building one armature
    # at a time, inline, as each entity is reached (the previous
    # design) meant paying it TWICE per MD5 entity (once entering Edit
    # Mode, once leaving) — a real test came back with 68 MD5 models
    # costing 23s combined, almost entirely this. Blender supports
    # editing multiple Armature objects' bones in a single Edit Mode
    # session when they're all selected together, so every armature
    # this import will need is instead built here, up front — before
    # the main loop below has created any of the brushes/patches/other
    # objects that make later calls expensive — batched into ONE
    # mode_set(EDIT)/mode_set(OBJECT) pair for the whole map (directly
    # verified: 68 armatures batched this way cost ~0.6s combined even
    # with 7000 pre-existing objects already in the scene, versus ~35s
    # built one at a time). md5_prebuilt_armatures maps ent_idx -> the
    # armature the main loop below should use for that entity, instead
    # of building its own; _get_or_build_md5_instance() still builds
    # one inline as a fallback for any entity this pre-pass didn't
    # cover (e.g. a decl the main loop resolves differently than the
    # simplified duplicate check below — kept deliberately narrow and
    # conservative rather than trying to perfectly mirror every edge
    # case the real per-entity resolution below handles).
    md5_prebuilt_armatures = {}
    # ent_idx -> resolve_attachment_specs() result, for every entity that
    # has any. Built in the pre-pass below (which has to resolve them
    # anyway, to know which extra armatures the batch must include) and
    # reused verbatim by the main loop, so the two can never disagree
    # about what a character carries. Stays empty when attachments are
    # off, or when nothing about MD5 import is available.
    attach_specs_by_ent = {}

    def entity_attach_specs(ent_idx, entity, classname):
        """This entity's head/attachment specs, resolved once and
        memoized. Called from the pre-pass (which needs to know which
        extra armatures to batch) and again from the main loop (which
        places them), so the two can never disagree about what a
        character carries — and so an entity the pre-pass's
        deliberately narrow body-model check skipped still gets its
        attachments, just without the batched armature build."""
        if ent_idx in attach_specs_by_ent:
            return attach_specs_by_ent[ent_idx]
        specs = []
        if import_attachments and model_decls:
            def_keys = _entitydef_keys_cached(classname)
            if entity_has_attachment_keys(entity.keys, def_keys):
                merged = dict(def_keys)
                merged.update(entity.keys)   # the entity's own keys win
                specs = resolve_attachment_specs(merged, classname,
                                                 model_decls, _entitydef_keys_cached)
        attach_specs_by_ent[ent_idx] = specs
        return specs

    if import_md5_models and md5_addon is not None and model_decls:
        # A request's key is the ent_idx for a body model, or
        # (ent_idx, spec_index) for one of that entity's heads/
        # attachments — see md5_prebuilt_armatures' use below.
        _pp_requests   = []   # [(key, name, classname, resolved), ...]
        _pp_md5_cache  = {}   # resolved .md5mesh path -> parsed md5 dict, or None on failure

        def _pp_parse(rel):
            """Resolve a decl's "mesh" path and parse it, memoized and
            failure-tolerant — returns the resolved path, or None if
            it can't be found or won't parse. The main loop re-resolves
            and re-parses independently; this only decides whether an
            armature is worth adding to the batched build."""
            if not rel:
                return None
            resolved = resolve_model_path(filepath, rel, search_roots)
            if resolved is None:
                return None
            if resolved not in _pp_md5_cache:
                try:
                    _pp_md5_cache[resolved] = md5_addon.parse_md5mesh(resolved)
                except Exception:
                    _pp_md5_cache[resolved] = None
            return resolved if _pp_md5_cache[resolved] is not None else None

        for _pp_idx, _pp_entity in enumerate(entities):
            _pp_classname = _pp_entity.keys.get('classname', 'unknown')
            _pp_name_key  = _pp_entity.keys.get('name', '').strip()
            _pp_ent_name  = _pp_name_key if _pp_name_key else f"entity_{_pp_idx:04d}_{_pp_classname}"
            _pp_model_key = _pp_entity.keys.get('model', '').strip()
            if not _pp_model_key and entity_defs:
                _pp_def_model = _entitydef_keys_cached(_pp_classname).get('model')
                if _pp_def_model:
                    _pp_model_key = _pp_def_model.strip()
            _pp_model_key = _norm_slashes(_pp_model_key)   # see _norm_slashes
            if not _pp_model_key or _pp_model_key.lower() == _pp_ent_name.lower():
                continue
            # Mirrors looks_like_file_ref below — an MD5 decl name is
            # never file-extension-shaped, so this excludes .ase/.lwo/
            # unsupported-format references the same way the real
            # per-entity elif chain does.
            if re.match(r'^\.[a-z0-9]{1,10}$', os.path.splitext(_pp_model_key)[1].lower()):
                continue
            _pp_decl = model_decls.get(_pp_model_key.lower())
            if _pp_decl is None:
                continue
            _pp_mesh_rel = _pp_decl.get('mesh')
            if not _pp_mesh_rel:
                continue
            _pp_resolved = resolve_model_path(filepath, _pp_mesh_rel, search_roots)
            if _pp_resolved is None:
                continue
            if _pp_resolved not in _pp_md5_cache:
                try:
                    _pp_md5_cache[_pp_resolved] = md5_addon.parse_md5mesh(_pp_resolved)
                except Exception:
                    _pp_md5_cache[_pp_resolved] = None
            if _pp_md5_cache[_pp_resolved] is None:
                continue
            _pp_requests.append((_pp_idx, _pp_ent_name, _pp_classname, _pp_resolved))

            # Heads/attachments hang off a body model, so the batch
            # only ever covers an entity that just got queued for one.
            # Specs come parents-first, so a nested one's name stem is
            # always already known by the time it is reached - the same
            # stem place_attachment_specs() derives, so an armature is
            # named identically whether it came from this batch or was
            # built inline as the fallback.
            _pp_stems = {}
            for _pp_si, _pp_spec in enumerate(
                    entity_attach_specs(_pp_idx, _pp_entity, _pp_classname)):
                _pp_stem = "%s_%s" % (_pp_stems.get(_pp_spec['parent'], _pp_ent_name),
                                      _pp_spec['key'])
                _pp_stems[_pp_si] = _pp_stem
                _pp_sub_model = _norm_slashes(_pp_spec['model'] or '')
                _pp_sub_decl  = model_decls.get(_pp_sub_model.lower())
                if _pp_sub_decl is None:
                    continue    # static (.ase/.lwo) or unresolved: no armature
                _pp_sub_res = _pp_parse(_pp_sub_decl.get('mesh'))
                if _pp_sub_res is None:
                    continue
                _pp_requests.append(((_pp_idx, _pp_si), _pp_stem,
                                     _pp_classname, _pp_sub_res))

        if _pp_requests:
            # NOTE: deliberately does NOT seed md5_mesh_cache with what
            # was just parsed above — that cache's contract is (parsed
            # md5, already-built Mesh datablocks) as one pair (see
            # _get_or_build_md5_instance), and no meshes have been
            # built yet at this point (build_mesh needs a real, bone-
            # populated armature to reference for normal
            # reconstruction, and doesn't need Edit Mode at all, so
            # there's no benefit to doing it in this batched pre-pass).
            # The main loop below will parse_md5mesh() again the first
            # time it reaches one of these resolved paths — redundant,
            # but parsing alone was already confirmed cheap (not the
            # cost this pre-pass exists to avoid); only the Armature
            # build (the bpy.ops.object.mode_set() pair) actually
            # needed batching, and that's what md5_prebuilt_armatures
            # hands the main loop below instead.
            _pp_arm_objs = []   # [(ent_idx, arm_obj, md5), ...]
            for _pp_key, _pp_ent_name, _pp_classname, _pp_resolved in _pp_requests:
                _pp_arm_obj = md5_addon._new_armature_object(_pp_ent_name, get_class_col(_pp_classname))
                _pp_arm_obj.select_set(True)
                md5_prebuilt_armatures[_pp_key] = _pp_arm_obj
                _pp_arm_objs.append((_pp_key, _pp_arm_obj, _pp_md5_cache[_pp_resolved]))

            bpy.context.view_layer.objects.active = _pp_arm_objs[0][1]
            bpy.ops.object.mode_set(mode='EDIT')
            for _pp_key, _pp_arm_obj, _pp_md5 in _pp_arm_objs:
                md5_addon._populate_armature_bones(_pp_arm_obj, _pp_md5, scale, rot_mat=None)
            bpy.ops.object.mode_set(mode='OBJECT')

            for _pp_key, _pp_arm_obj, _pp_md5 in _pp_arm_objs:
                md5_addon._finish_armature_object(_pp_arm_obj)
                _pp_arm_obj.hide_set(True)
                _pp_arm_obj.select_set(False)

    # ── park the import outside the view layer ──────────────────────
    # Everything from here to the "finishing up" step below is built
    # into root_col with the view layer excluded from it, and the whole
    # lot is revealed in one go at the end.
    #
    # The reason is that the import runs MODALLY: it hands control back
    # to Blender every _TICK_BUDGET so the status line updates and ESC
    # still cancels, and each of those returns costs time proportional
    # to how many objects the VIEW LAYER holds — Blender re-evaluates
    # the depsgraph, redraws the viewport and rebuilds the Outliner
    # before the next tick. Measured, depsgraph alone: 12ms at 2,000
    # objects, 61ms at 10,000. With the collection excluded the same
    # measurement is 0.07ms and 0.20ms — the objects still exist and are
    # still being built, the view layer simply has no view of them yet.
    # Revealing 10,000 of them at the end costs 0.24s, once.
    #
    # This is deliberately AFTER the armature pre-pass above, which
    # needs the view layer for its select/active/mode_set round trip and
    # already runs before anything else has been built for exactly the
    # same reason (see its own note). Everything the main loop does from
    # here on works on datablocks — bpy.data.objects.new(), collection
    # linking, meshes, modifiers, materials — none of which needs a view
    # layer. The two things that DO are hide_set(), which is deferred
    # through pending_hide, and the final selection, which happens after
    # the reveal. ANYTHING ADDED BELOW THAT REACHES FOR
    # context.view_layer WILL FIND IT EMPTY.
    root_layer_col = _find_layer_collection(view_layer.layer_collection, root_col)
    if root_layer_col is not None:
        root_layer_col.exclude = True
        _PARKED_LAYER_COLLECTIONS.append(root_layer_col)

    # resolved "ik_pose" .md5anim path -> {lowercase joint name: its
    # model-space Matrix at frame 0, scaled to match the armature}, or
    # None if that anim can't be read. See joint_bind_matrices.
    ik_pose_cache = {}

    def joint_bind_matrices(decl):
        """
        The joint transforms of *decl*'s "ik_pose" anim at frame 0 - the
        pose idActor::Spawn puts a character in immediately before
        attaching anything to it, and therefore the pose every offset on
        a "def_attach" was authored against. Returns None when the decl
        declares no "ik_pose" (or it can't be read), which leaves
        attachment_bind_local_matrix composing in the rest pose instead.

        Read straight out of the parsed .md5anim via
        build_frame_skeleton, which composes the joint hierarchy from
        data alone - no Armature, no pose assignment, no depsgraph, and
        nothing that disturbs the pose the body is actually being
        imported in. Cached per resolved anim file: one shared "ik_pose"
        (models/md5/chars/af_pose.md5anim, inherited by npc_base) covers
        most of a Doom 3 map's characters between them.
        """
        if not decl:
            return None
        rel = decl['anims'].get('ik_pose')
        if not rel:
            return None
        resolved = resolve_model_path(filepath, rel, search_roots)
        if resolved is None:
            return None
        if resolved not in ik_pose_cache:
            table = None
            try:
                anim = md5_addon.parse_md5anim(resolved)
                table = {}
                for info, mat in zip(anim['hierarchy'],
                                     md5_addon.build_frame_skeleton(anim, 0)):
                    mat = mat.copy()
                    # build_frame_skeleton works in raw MD5 units; the
                    # armature's bones are built at the import scale, and
                    # the two get composed together.
                    mat.translation = mat.translation * scale
                    table[info['name'].strip().lower()] = mat
            except Exception:
                table = None
            ik_pose_cache[resolved] = table
        return ik_pose_cache[resolved]

    def place_attachment_specs(ent_idx, ent_name, classname, class_col,
                               body_arm_obj, body_decl, specs):
        """
        Place one character's head and everything it carries or wears,
        each bound to a joint of whatever it hangs off - the body for
        most, another attachment for the nested cases (a hat on a head).
        *specs* comes from resolve_attachment_specs, which already
        ordered parents before children; *body_decl* is the .map
        entity's own model decl, which supplies the "ik_pose" the body's
        own attachments bind in (each attachment's decl then supplies it
        for anything nested under that).

        Returns (ok, skip, af_pose_used); every failure is described in
        `issues` before being counted, and never aborts the rest of the
        character - a missing hat is not a reason to drop the head.

        Each attachment gets the same treatment a .map entity's own
        model does: shared/cached Mesh datablocks, a per-instance
        Armature so it can carry its own animation, its skin swap
        applied, and its objects appended to all_objects so the material
        pass and the final selection see them. What differs is only the
        transform - bone_parent_to instead of parenting to the entity's
        empty - and that is deliberately the ONLY difference, so
        anything later added to model placement applies here too.
        """
        ok = skip = 0
        af_used = 0
        # spec index -> the Object anything nested under it binds to:
        # its Armature for a skeletal attachment, or its first mesh for
        # a static one (which has no skeleton, so a child of it binds to
        # the parent's own origin - see attachment_local_matrix). A spec
        # that placed nothing records no entry at all.
        spec_arms  = {}
        # spec index -> the object-name stem its models were given, so a
        # nested attachment's name carries the whole chain it hangs off
        # ("<entity>_def_head_def_attach1" for a hat on a head). Without
        # that, a hat declared as "def_attach1" on the head collides
        # with the body's own "def_attach1" and Blender silently
        # uniquifies one of them into a ".001".
        spec_names = {}
        # spec index -> the model decl it placed, so anything nested
        # under it can look up THAT model's "ik_pose" rather than the
        # body's. None for a static parent, which has no anims at all.
        spec_decls = {}

        for si, spec in enumerate(specs):
            if spec['parent'] is None:
                parent_arm  = body_arm_obj
                parent_stem = ent_name
                parent_decl = body_decl
            else:
                parent_stem = spec_names.get(spec['parent'], ent_name)
                parent_decl = spec_decls.get(spec['parent'])
                if spec['parent'] not in spec_arms:
                    # The parent failed to place at all, and said so —
                    # don't report the knock-on a second time.
                    skip += 1
                    continue
                parent_arm = spec_arms[spec['parent']]

            what = f'"{spec["key"]}" "{spec["value"]}"'

            if not spec['model']:
                if spec['keys']:
                    # The entityDef resolved and simply declares no
                    # model. That is how idTech4 hangs a LIGHT, a
                    # particle emitter or a sound source off a character
                    # (a flame on a carried torch, say) - routine, and
                    # nothing a mesh importer places. Reporting it would
                    # put a line in the report for every torch, lantern
                    # and candle in the map.
                    continue
                skip += 1
                issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                              f"{what} matches no \"model NAME {{ ... }}\" decl and "
                              f"no entityDef — nothing to place")
                continue

            # A skeletal parent is bound to by JOINT; a static one has no
            # joints, so its children bind to its own origin instead (see
            # attachment_local_matrix). Either way the spawnarg naming a
            # joint is simply unused in the second case, exactly as The
            # Dark Mod's idEntity::Attach ignores it.
            bone = None
            if parent_arm.type == 'ARMATURE':
                bone = _find_bone(parent_arm, spec['joint'])
                if bone is None:
                    skip += 1
                    issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                                  f"{what} binds to joint \"{spec['joint']}\", which is "
                                  f"not a bone of \"{parent_arm.name}\" — not placed")
                    continue

            # Composed in the parent's "ik_pose" (the pose the engine
            # binds in) and rebased into the rest pose Blender parents
            # in - see attachment_bind_local_matrix for why that is not
            # the same thing, and for what it costs to get wrong.
            joint_rest = bone.matrix_local if bone is not None else None
            joint_bind = None
            if bone is not None:
                bind_table = joint_bind_matrices(parent_decl)
                if bind_table:
                    joint_bind = bind_table.get(bone.name.strip().lower())
            local = attachment_bind_local_matrix(spec, joint_rest, joint_bind, scale)

            sub_name  = f"{parent_stem}_{spec['key']}"
            spec_names[si] = sub_name
            model_key = _norm_slashes(spec['model'])
            model_ext = os.path.splitext(model_key)[1].lower()
            decl      = model_decls.get(model_key.lower())
            spec_decls[si] = decl

            # "skin" precedence is the same three-step one a .map
            # entity's own model gets (see the main loop): the attached
            # entityDef's own spawnarg first, then whatever its model
            # decl declares as its default skin.
            skin_key   = spec['skin']
            if not skin_key and decl is not None:
                skin_key = (decl.get('skin') or '').strip() or None
            skin_pairs = None
            if apply_skins and skin_key:
                skin_pairs = skin_defs.get(engine_canonical_decl(skin_key))
                if skin_pairs is None:
                    issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                                  f"{what}: \"skin\" \"{skin_key}\" not found in any "
                                  f".skin file under the \"skins\" folder")

            # ---- skeletal (MD5) attachment ------------------------
            if decl is not None:
                mesh_rel = decl.get('mesh')
                if not mesh_rel:
                    skip += 1
                    issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                                  f"{what}: model decl \"{model_key}\" has no "
                                  f"\"mesh\" line — nothing to import")
                    continue
                resolved = resolve_model_path(filepath, mesh_rel, search_roots)
                if resolved is None:
                    skip += 1
                    issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                                  f"{what}: MD5 mesh file not found: \"{mesh_rel}\" "
                                  f"(from model decl \"{model_key}\")")
                    continue
                if md5_mesh_cache.get(resolved, False) is None:
                    skip += 1      # cached failure, already reported once
                    continue
                # Claimed out of the batch, not just read: what is
                # still in md5_prebuilt_armatures once the whole import
                # is over is exactly "built but never placed", which is
                # what the leftover sweep after the entity loop removes.
                # Put back on failure, for that same sweep to collect.
                prebuilt = md5_prebuilt_armatures.pop((ent_idx, si), None)
                try:
                    sub_arm, sub_meshes = _get_or_build_md5_instance(
                        md5_addon, md5_mesh_cache, resolved, scale,
                        class_col, sub_name, arm_obj=prebuilt,
                        hide_sink=pending_hide)
                except Exception as exc:
                    if prebuilt is not None:
                        md5_prebuilt_armatures[(ent_idx, si)] = prebuilt
                    md5_mesh_cache[resolved] = None
                    skip += 1
                    issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                                  f"{what}: exception loading MD5 model "
                                  f"\"{model_key}\" (\"{mesh_rel}\"): {exc}")
                    continue

                # Same "origin" bone compensation the body model gets:
                # shift the model in its own local space so that bone's
                # rest POSITION becomes its local (0,0,0), then apply
                # the joint placement on top. Position only, never the
                # bone's own rest orientation - see the body path.
                origin_bone = _find_bone(sub_arm, 'origin')
                sub_local = local
                if origin_bone is not None:
                    sub_local = local @ Matrix.Translation(
                        -origin_bone.matrix_local.translation)

                for obj in [sub_arm] + sub_meshes:
                    attach_parent_to(obj, parent_arm, bone, sub_local)
                    all_objects.append(obj)
                spec_arms[si] = sub_arm
                # The head's own copies of the body's neck joints, driven
                # by the body's (idAFAttachment::CopyJointsFromBody). The
                # bind above puts the head's ROOT on the joint; without
                # this the rest of it stays at the bind transform and the
                # head does not bend with a posed or animated neck -
                # 14 cm at the crown on Quake 4's more heavily posed
                # cinematic NPCs. Skipped when the parent has no skeleton
                # to copy from.
                if spec['copy_joints'] and parent_arm.type == 'ARMATURE':
                    _cj_ok, _cj_skip = apply_copy_joints(
                        sub_arm, parent_arm, spec['copy_joints'])
                    copy_joint_counts[0] += _cj_ok
                    copy_joint_counts[1] += _cj_skip
                if skin_pairs:
                    apply_skin_material_swap(sub_meshes, skin_pairs, mat_cache)
                ok += 1

                if import_md5_animations:
                    anim_id, anim_rel, explicit, had_animN, used_af = choose_decl_anim(
                        decl['anims'], spec['anim'] or '', spec['keys'])
                    if used_af:
                        af_used += 1
                    if not anim_rel:
                        if explicit or had_animN:
                            issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                                          f"{what}: \"anim\" \"{spec['anim'] or anim_id}\" "
                                          f"has no matching \"anim ...\" entry in model "
                                          f"decl \"{model_key}\"")
                    else:
                        resolved_anim = resolve_model_path(filepath, anim_rel, search_roots)
                        if resolved_anim is None:
                            issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                                          f"{what}: MD5 anim file not found: "
                                          f"\"{anim_rel}\" (anim \"{anim_id}\")")
                        elif md5_anim_cache.get(resolved_anim, False) is None:
                            pass       # cached failure, already reported once
                        else:
                            try:
                                _get_or_apply_md5_anim(md5_addon, md5_anim_cache,
                                                       resolved_anim, sub_arm, scale,
                                                       md5_first_frame_only)
                            except Exception as exc:
                                md5_anim_cache[resolved_anim] = None
                                issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                                              f"{what}: exception applying MD5 anim "
                                              f"\"{anim_id}\" (\"{anim_rel}\"): {exc}")
                continue

            # ---- static (.ase/.lwo) attachment ---------------------
            # Real content mixes the two freely on one character: The
            # Dark Mod hangs .lwo pauldrons off the same AI that carries
            # a skeletal sword, and Doom 3's cinematic props are .lwo as
            # often as MD5.
            if model_ext not in ('.ase', '.lwo'):
                skip += 1
                issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                              f"{what} resolves to model \"{model_key}\", which is "
                              f"neither a known MD5 model decl nor an .ase/.lwo file")
                continue
            if not import_models or model_addon is None:
                # Static-model import wasn't requested (or the addon
                # isn't there - reported once, up front). Same silence
                # the main loop keeps in that case.
                continue

            resolved = resolve_model_path(filepath, model_key, search_roots)
            if resolved is None:
                skip += 1
                issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                              f"{what}: model file not found: \"{model_key}\"")
                continue
            sub_meshes = model_mesh_cache.get(resolved)
            if sub_meshes is None:
                load_exc = None
                try:
                    sub_meshes = model_addon.load_model_meshes(
                        resolved, scale,
                        lwo_first_layer_only=lwo_first_layer_only,
                        name_hint=model_key,
                        shading=model_shading,
                        shading_overrides=model_shading_overrides(
                            model_addon, resolved, lwo_first_layer_only,
                            source_path, import_materials, model_shading))
                except Exception as exc:
                    sub_meshes = []
                    load_exc = exc
                    issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                                  f"{what}: exception loading model "
                                  f"\"{model_key}\": {exc}")
                if not sub_meshes and load_exc is None:
                    issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                                  f"{what}: model file \"{model_key}\" loaded but "
                                  f"produced no usable geometry")
                for _mname, mesh, mat_names in sub_meshes:
                    for mn in mat_names:
                        mesh.materials.append(get_or_create_material(mn, mat_cache))
                    apply_mesh_smoothing(mesh, model_shading == 'SMOOTH')
                # Cached even when empty, so a broken reference shared by
                # many characters is only attempted once.
                model_mesh_cache[resolved] = sub_meshes
            if not sub_meshes:
                # Either just reported above, or a cached failure whose
                # reason an earlier character already logged.
                skip += 1
                continue

            placed = []
            for m_idx, (_mname, mesh, _mat_names) in enumerate(sub_meshes):
                obj_name = (sub_name if len(sub_meshes) == 1
                            else f"{sub_name}{m_idx}")
                mobj = bpy.data.objects.new(obj_name, mesh)
                class_col.objects.link(mobj)
                attach_parent_to(mobj, parent_arm, bone, local)
                all_objects.append(mobj)
                placed.append(mobj)
            if skin_pairs:
                apply_skin_material_swap(placed, skin_pairs, mat_cache)
            # No skeleton, so anything nested under this binds to the
            # object itself. Every piece shares one transform, so which
            # piece is recorded doesn't matter.
            spec_arms[si] = placed[0]
            ok += 1

        return ok, skip, af_used


    for ent_idx, entity in enumerate(entities):
        classname = entity.keys.get('classname', 'unknown')
        # Prefer the entity's "name" key if present (e.g. "func_static_52824"),
        # fall back to "entity_NNNN_classname" for unnamed entities.
        name_key  = entity.keys.get('name', '').strip()
        ent_name  = name_key if name_key else f"entity_{ent_idx:04d}_{classname}"

        # Marks where THIS entity's own objects start in all_objects —
        # every brush/patch/model/armature Object placed below for this
        # entity gets appended there, in several different branches, so
        # this is the only reliable way to slice out "every Object this
        # one entity produced" afterward (see the "hide" handling right
        # before this iteration's yield, further down).
        entity_obj_start = len(all_objects)

        # "hide" spawnarg: same precedence idTech4 itself uses to spawn
        # an entity already hidden — verified against the Doom 3 BFG GPL
        # source (Entity.cpp: "fl.hidden = spawnArgs.GetBool( "hide",
        # "0" )", where spawnArgs is the entity's own keys layered over
        # its entityDef's defaults). The entity's own "hide" key wins if
        # present; otherwise falls back to its classname's entityDef
        # "hide" (following "inherit" chains the same way "model"/"skin"
        # do — see _entitydef_keys_cached). GetBool itself is
        # atoi(value) != 0, not a literal "1"-string check, so this uses
        # the same parsing (_spawnarg_bool) rather than testing for "1"
        # specifically.
        hide_val = entity.keys.get('hide', '').strip()
        if not hide_val and entity_defs:
            def_hide = _entitydef_keys_cached(classname).get('hide')
            if def_hide:
                hide_val = def_hide.strip()
        entity_hidden = _spawnarg_bool(hide_val)

        # Entity empty — all entities go into the root collection
        empty = bpy.data.objects.new(ent_name, None)
        empty.empty_display_type = 'ARROWS'
        empty.empty_display_size = 0.25
        class_col = get_class_col(classname)
        class_col.objects.link(empty)

        # ── Determine classname / origin BEFORE building brush geometry,
        # so brushDef3 UV computation can account for the origin offset
        # (see build_brush_mesh's origin_for_uv doc) at build time. ──────
        is_worldspawn = entity.keys.get('classname', '').strip().lower() == 'worldspawn'
        orig = entity.keys.get('origin', '').split()
        origin_vec = None
        if len(orig) >= 3:
            try:
                origin_vec = Vector((float(orig[0]), float(orig[1]), float(orig[2])))
            except ValueError:
                origin_vec = None
        # Matches the exact condition used for brush_parent_inverse below:
        # brushes on a non-worldspawn entity with an origin get it added.
        origin_for_uv = origin_vec if (origin_vec is not None and not is_worldspawn) else None

        # ── Build brush / patch meshes FIRST (before deciding the empty's
        # location) so we can inspect the raw geometry's bounding box. ────
        built_brushes = []   # [(b_idx, mesh, mat_names), ...]
        for b_idx, brush in enumerate(entity.brushes):
            try:
                mesh, mat_names = build_brush_mesh(brush, scale, origin_for_uv=origin_for_uv)
            except Exception as exc:
                brush_skip += 1
                issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                              f"brush {b_idx} raised an exception building geometry: {exc}")
                continue
            if mesh is None:
                brush_skip += 1
                issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                              f"brush {b_idx} produced no usable geometry (degenerate/invalid planes)")
                continue
            built_brushes.append((b_idx, mesh, mat_names))

        built_patches = []   # [(p_idx, mesh, mat_names), ...]
        for p_idx, patch in enumerate(entity.patches):
            try:
                mesh, mat_names = build_patch_mesh(patch, scale)
            except Exception as exc:
                patch_skip += 1
                issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                              f"patch {p_idx} raised an exception building geometry: {exc}")
                continue
            if mesh is None:
                patch_skip += 1
                issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                              f"patch {p_idx} produced no usable geometry (malformed control data)")
                continue
            # Patches are always shade-smoothed — matching the real engine
            # (idSurface_Patch::GenerateNormals computes a continuously
            # curved normal field unconditionally, with no mapper-facing
            # smoothing toggle at all). Skipped only for a fully flat
            # patch, already reduced to a single quad (see
            # _patch_is_entirely_flat), where smoothing has no visual
            # effect anyway.
            if not _patch_is_entirely_flat(patch.ctrl, patch.cols, patch.rows):
                apply_mesh_smoothing(mesh, True)
            built_patches.append((p_idx, mesh, mat_names))

        has_geometry = bool(built_brushes or built_patches)

        # Bounding box (Blender-space, already scaled) of all raw geometry,
        # used below only for entities with NO "origin" key at all (e.g.
        # worldspawn) to give their empty a sensible pivot.
        bbox_min = bbox_max = None
        for (_, mesh, _) in built_brushes + built_patches:
            for v in mesh.vertices:
                co = v.co
                if bbox_min is None:
                    bbox_min = Vector(co)
                    bbox_max = Vector(co)
                else:
                    bbox_min.x = min(bbox_min.x, co.x); bbox_max.x = max(bbox_max.x, co.x)
                    bbox_min.y = min(bbox_min.y, co.y); bbox_max.y = max(bbox_max.y, co.y)
                    bbox_min.z = min(bbox_min.z, co.z); bbox_max.z = max(bbox_max.z, co.z)

        # idTech4's "origin" handling is NOT the same for patches and
        # brushes — verified directly against the Doom 3 GPL source
        # (idlib/MapFile.cpp, tools/compilers/dmap/map.cpp):
        #
        #   Point entities (lights, speakers, info_player_start, etc.) have NO
        #   geometry; their "origin" IS their world position and must be applied
        #   to the empty so it appears in the right place in Blender.
        #
        #   Patches: idMapPatch::Parse ALWAYS subtracts the entity's origin
        #   from every control point when the .map file is loaded
        #   ("vert->xyz[0] = v[0] - origin[0]"), and idMapPatch::Write adds
        #   it straight back when saving. The engine positions the entity
        #   at runtime by adding origin back to that already-shifted local
        #   surface — so end to end, the raw coordinates written in the
        #   .map file are ALWAYS already the correct final position for a
        #   patch, unconditionally, for every entity. No exceptions.
        #
        #   Brushes: idMapBrush::Parse stores a brushDef3 plane exactly as
        #   written — untouched. But dmap's compiler separately calls
        #   AdjustEntityForOrigin, which unconditionally adds the entity's
        #   origin to every brush plane (and texture vector) belonging to
        #   any NON-worldspawn entity, permanently baking that shift into
        #   the compiled static geometry with no runtime undo. Worldspawn
        #   is explicitly special-cased to never receive an origin
        #   ("never put an origin on the world, even if the editor left
        #   one there").
        #
        # So: patches never get origin added, ever. Brushes on any
        # non-worldspawn entity always get it added, unconditionally —
        # regardless of the brush's own raw coordinate magnitude.
        # (is_worldspawn / origin_vec were already computed above, before
        # building brush geometry, so build_brush_mesh could account for
        # the origin offset in brushDef3 UV computation.)
        child_parent_inverse = Matrix.Identity(4)

        if origin_vec is not None:
            empty.location = to_bl(origin_vec, scale)
        elif has_geometry and bbox_min is not None:
            # No "origin" key at all (e.g. worldspawn, func_group): give
            # the empty AND every mesh Object built from this geometry a
            # shared pivot at the geometry's own bounding-box center,
            # instead of leaving every mesh Object's own origin collapsed
            # to world (0,0,0) — which is what moving only the empty used
            # to produce, since matrix_parent_inverse below cancelled the
            # empty's offset back out for every child before it could
            # reach them. This matters most for worldspawn, which can own
            # thousands of brushes/patches — without it, every one of
            # them draws a relationship line all the way back to a single
            # point at world origin, and every mesh Object's own pivot
            # sits there too, off in a corner of the map.
            #
            # child_parent_inverse is deliberately left at the Identity
            # set above rather than cancelling this offset: the geometry
            # itself is shifted by the same amount below instead, so the
            # pivot actually reaches every brush/patch Object built from
            # it further down (brush_parent_inverse/patch_parent_inverse
            # both ultimately derive from this same child_parent_inverse
            # whenever there's no real origin key).
            #
            # This pivot is NOT a real idTech4 "origin" — the entity had
            # no such key in the .map file, so a future .map exporter
            # must never write one back out for it. Marked explicitly via
            # the "map_origin_synthetic" custom property so that exporter
            # can tell the two apart directly, rather than having to infer
            # it from "map_origin" merely being absent.
            bbox_center = (bbox_min + bbox_max) * 0.5
            empty.location = bbox_center
            empty['map_origin_synthetic'] = True
            bbox_shift = Matrix.Translation(-bbox_center)
            for (_, mesh, _) in built_brushes:
                mesh.transform(bbox_shift)
                mesh.update()
            # Patches are NOT shifted here — unlike brushes, their raw
            # coordinates are already absolute regardless of origin_vec
            # (see the patch-vs-brush origin comment below), so they get
            # the exact same empty.location-relative pivot treatment
            # whether that location came from a real "origin" key or this
            # synthetic bbox-center fallback. Handled once, uniformly,
            # right below.

        # Patches never receive the "origin added" treatment brushes get
        # (see the big comment above, sourced from idMapPatch::Parse/
        # Write) — their raw coordinates are ALWAYS already the correct
        # final absolute position, whether or not this entity has an
        # "origin" key at all. That used to mean every patch Object's own
        # pivot was deliberately cancelled back to world (0,0,0) via
        # matrix_parent_inverse, regardless of how sensible a pivot
        # empty.location holds — which is exactly why entities like a
        # func_static built entirely from patches (a very common case)
        # still fanned relationship lines out to a single point at world
        # origin even after brushes/worldspawn were fixed to use their
        # entity's own pivot. Shift patch geometry the same way brushes
        # already are above, so every patch Object's pivot lands at
        # empty.location too, real origin or synthetic alike.
        if built_patches and empty.location.length_squared > 0.0:
            patch_shift = Matrix.Translation(-empty.location)
            for (_, mesh, _) in built_patches:
                mesh.transform(patch_shift)
                mesh.update()

        store_entity_keys(empty, entity.keys)
        all_objects.append(empty)

        # ── static / animated mesh model (.ase / .lwo), if any ─────────
        # func_static / misc_model / etc. entities reference an external
        # mesh file via the "model" key. Its placement is given by the
        # same "origin" key handled above, plus a "rotation"/"angles"/
        # "angle" key (idTech4 stores model rotation independently of the
        # runtime physics "origin" pivot used for brush entities).
        if import_models or import_md5_models:
            model_key = entity.keys.get('model', '').strip()
            model_from_def_fallback = False
            if not model_key and entity_defs:
                # No explicit "model" spawnarg on this entity — fall back
                # to whatever its classname's entityDef declares, exactly
                # matching real idTech4 spawnarg layering: the entity's
                # own key always wins if present, entityDef only supplies
                # the default when it's absent. Follows "inherit" chains
                # (see resolve_entitydef_keys) so a def that inherits its
                # model from a base def, rather than declaring its own,
                # still resolves correctly.
                def_model = _entitydef_keys_cached(classname).get('model')
                if def_model:
                    model_key = def_model.strip()
                    model_from_def_fallback = True
            # Normalize '\' vs '/' before any comparison/lookup below
            # (model_decls, is_self_reference, resolve_model_path all
            # key/compare on this string) — a .map entity's "model" and
            # a .def's "model NAME { }" decl mix both interchangeably in
            # real content (see _norm_slashes).
            model_key = _norm_slashes(model_key)
            model_ext = os.path.splitext(model_key)[1].lower()
            # The "model" spawnarg has TWO real, distinct meanings in
            # idTech4 .map files — not just "external mesh file
            # reference":
            #   1. An actual file path (e.g.
            #      "models/mapobjects/lab/diamondbox/diamondbox.lwo").
            #   2. A bare identifier with no file extension at all — most
            #      commonly the entity's own "name", a well-known
            #      idTech4/Doom3 authoring pattern for brush-built
            #      entities where "model" mirrors "name" purely for the
            #      engine/compiler's own bookkeeping of that entity's
            #      compiled brush geometry, not a reference to any
            #      loadable file. Also covers inline brush-model markers
            #      like "*1" and other bare decl-style identifiers.
            # Case 2 was being misdetected as an omission — any model_key
            # without a recognized .ase/.lwo extension got flagged as
            # "unsupported format", even though there was never a file
            # to load in the first place. Only something that actually
            # LOOKS like a file reference (a short, alphabetic extension)
            # and isn't just self-referencing the entity's own name gets
            # treated as an attempted — and possibly failed — file load.
            looks_like_file_ref = bool(re.match(r'^\.[a-z0-9]{1,10}$', model_ext))
            is_self_reference = bool(model_key) and model_key.lower() == ent_name.lower()

            # "skin" spawnarg: per-instance material swap for whichever
            # model THIS entity places (see apply_skin_material_swap).
            # Extension is stripped before lookup so "skins/foo.skin"
            # matches a decl named "skins/foo" (see engine_canonical_decl).
            #
            # Just like "model" above, "skin" is a normal entityDef
            # spawnarg too — real idTech4 content routinely gives a
            # monster/NPC's entityDef its own default "skin" key (often
            # inherited from a shared base def, e.g. a "_black" or
            # "_red" variant skin declared once on a base classname and
            # picked up by every subclass that doesn't override it) so
            # every placed instance of that classname wears it without
            # every mapper having to set "skin" by hand on every entity.
            # The entity's own explicit "skin" key still always wins;
            # this is only a fallback for entities that don't set one.
            # _entitydef_keys_cached() already walks the full "inherit"
            # chain (root-most ancestor first, nearest override last),
            # so an inherited "skin" resolves exactly like an inherited
            # "model" does.
            skin_key = entity.keys.get('skin', '').strip()
            if not skin_key and entity_defs:
                def_skin = _entitydef_keys_cached(classname).get('skin')
                if def_skin:
                    skin_key = def_skin.strip()
            if not skin_key and model_decls:
                # STILL nothing at the entity/entityDef spawnarg level —
                # fall back to the "model NAME { ... }" decl's own
                # "skin" line (see parse_model_decls' 'skin' field),
                # itself already resolved through THAT decl's own
                # "inherit" chain the same way "mesh" is. This exactly
                # matches real idTech4 engine precedence, verified
                # against the Doom 3 BFG GPL source
                # (idEntity::UpdateModel, neo/d3xp/Entity.cpp): an
                # explicit "skin" spawnarg — direct or entityDef-
                # inherited — always wins; only when spawnArgs has NONE
                # at all does the engine fall back to
                # modelDef->GetDefaultSkin(), i.e. the "skin" line on
                # the model decl the entity's "model" key names. Real
                # content relies on exactly this: e.g. "model
                # npc_security { skin skins/characters/npcs/regsec.skin
                # ... }" gives every character placed via a decl that
                # inherits npc_security (directly or several levels
                # down) that skin with no "skin" spawnarg anywhere in
                # its entityDef chain at all.
                model_decl_for_skin = model_decls.get(model_key.lower())
                if model_decl_for_skin and model_decl_for_skin.get('skin'):
                    skin_key = model_decl_for_skin['skin'].strip()
            skin_pairs = None
            if apply_skins and skin_key:
                skin_pairs = skin_defs.get(engine_canonical_decl(skin_key))
                if skin_pairs is None:
                    issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                                  f"\"skin\" \"{skin_key}\" not found in any .skin file "
                                  f"under the \"skins\" folder")

            if model_key and looks_like_file_ref and not is_self_reference and model_ext not in ('.ase', '.lwo'):
                if model_ext == '.prt':
                    # Not a model reference at all — .prt is idTech4's
                    # particle-system declaration format. Never a mesh,
                    # so never something this importer could load
                    # regardless of format support. Counted on its own
                    # (prt_skip) rather than as a model_skip/static_
                    # model_skip failure, and rolled up into a single
                    # report summary line instead of an issue per
                    # entity — a map can reference the same handful of
                    # particle systems on hundreds of entities, and none
                    # of those are a genuine import problem.
                    prt_skip += 1
                else:
                    # A real file IS referenced, just not a format this
                    # importer can load (e.g. .md5mesh, an animated model —
                    # those need a skeleton/anim to mean anything and aren't
                    # a static mesh this importer handles) — a genuine
                    # omission worth surfacing.
                    model_skip += 1
                    static_model_skip += 1
                    issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                                  f"references model \"{model_key}\" — unsupported format "
                                  f"(only .ase/.lwo are loaded)")
            elif model_key and looks_like_file_ref and not is_self_reference and model_ext in ('.ase', '.lwo'):
                _dbg_fmt_t0 = time.monotonic()
                if import_models and model_addon is not None:
                    resolved = resolve_model_path(filepath, model_key, search_roots)
                    sub_meshes = []
                    if resolved is None:
                        model_skip += 1
                        static_model_skip += 1
                        issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                                      f"model file not found: \"{model_key}\" (searched Game/Mod "
                                      f"Root and the .map file's own ancestor directories)")
                    elif resolved in model_mesh_cache:
                        # Another entity already referenced this exact file —
                        # reuse its Mesh datablocks instead of re-parsing the
                        # file and building duplicate geometry. This IS real
                        # Blender instancing: every entity that places this
                        # same model gets its own Object (its own transform),
                        # but every one of those Objects points at the same
                        # single Mesh datablock created here, only once, the
                        # first time this exact resolved file path is seen.
                        sub_meshes = model_mesh_cache[resolved]
                        if not sub_meshes:
                            # Cached failure from an earlier entity — don't
                            # log this AGAIN for every subsequent entity that
                            # references the same broken file; the first
                            # entity to hit it already logged the real reason
                            # below, at the point the cache was populated.
                            model_skip += 1
                            static_model_skip += 1
                    else:
                        load_exc = None
                        _shading_overrides = model_shading_overrides(
                            model_addon, resolved, lwo_first_layer_only,
                            source_path, import_materials,
                            model_shading)
                        try:
                            sub_meshes = model_addon.load_model_meshes(resolved, scale,
                                                           lwo_first_layer_only=lwo_first_layer_only,
                                                           name_hint=model_key,
                                                           shading=model_shading,
                                                           shading_overrides=_shading_overrides)
                        except Exception as exc:
                            sub_meshes = []
                            load_exc = exc
                            issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                                          f"exception loading model \"{model_key}\": {exc}")
                        if not sub_meshes:
                            model_skip += 1
                            static_model_skip += 1
                            if load_exc is None:
                                issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                                              f"model file \"{model_key}\" loaded but produced no "
                                              f"usable geometry")
                        # Materials/smoothing are set up exactly once per
                        # unique mesh here, not per entity that references
                        # it below — repeating this on a shared/cached mesh
                        # would duplicate its material slots on every use.
                        for mname, mesh, mat_names in sub_meshes:
                            for mn in mat_names:
                                mesh.materials.append(get_or_create_material(mn, mat_cache))
                            # Only Smooth asks for Blender's own averaging.
                            # Engine/File already wrote custom split normals
                            # (apply_mesh_smoothing skips a mesh that has
                            # them), and Flat means faceted — passing True
                            # unconditionally is what made Flat impossible
                            # here. Mirrors the standalone importer's own
                            # apply_mesh_smoothing(mesh, mode == SHADING_SMOOTH).
                            apply_mesh_smoothing(mesh, model_shading == 'SMOOTH')
                        # Cache the result even if empty, so a broken/missing
                        # reference isn't re-attempted for every entity that
                        # points at it.
                        model_mesh_cache[resolved] = sub_meshes

                    if sub_meshes:
                        rot_matrix = entity_rotation_matrix(entity)
                        model_origin = to_bl(origin_vec, scale) if origin_vec is not None else Vector((0.0, 0.0, 0.0))
                        target_world = Matrix.Translation(model_origin) @ rot_matrix
                        # Compensate for wherever the empty's own pivot ended up
                        # above, so the model lands at the correct world
                        # position/rotation regardless of that pivot logic.
                        # Computed algebraically from empty.location (rather
                        # than reading back empty.matrix_world) since the empty
                        # is always unparented and never rotated/scaled — this
                        # avoids any dependency on the transform having already
                        # been re-evaluated by Blender at this point.
                        parent_inv = Matrix.Translation(-empty.location)
                        placed_mesh_objs = []

                        for m_idx, (mname, mesh, mat_names) in enumerate(sub_meshes):
                            # Each entity gets its own Object with its own
                            # transform, instancing the (possibly shared/cached)
                            # Mesh datablock — materials/smoothing were already
                            # applied once above, whether on this entity or an
                            # earlier one that referenced the same file.
                            #
                            # Object name is derived from the ENTITY's own
                            # already-unique name (ent_name — see above), NOT
                            # the model name. The model name is deliberately
                            # reused across every entity that places the same
                            # prop (that's the whole point of sharing one Mesh
                            # datablock), so naming the Object after it would
                            # force Blender's own auto-uniquify to kick in on
                            # nearly every placement of a repeated prop,
                            # producing a ".001", ".002", ... suffix on almost
                            # every instance. Since ent_name is already unique
                            # per entity, appending a constant suffix keeps that
                            # guarantee without ever colliding with the entity's
                            # own empty (which uses ent_name with no suffix).
                            obj_name = f"{ent_name}_model" if len(sub_meshes) == 1 else f"{ent_name}_model{m_idx}"
                            mobj = bpy.data.objects.new(obj_name, mesh)
                            class_col.objects.link(mobj)
                            mobj.parent = empty
                            mobj.matrix_parent_inverse = parent_inv
                            mobj.matrix_basis = target_world
                            all_objects.append(mobj)
                            placed_mesh_objs.append(mobj)
                            model_ok += 1
                            static_model_ok += 1

                        if skin_pairs:
                            apply_skin_material_swap(placed_mesh_objs, skin_pairs, mat_cache)
                # else: static-model import wasn't requested (or the addon
                # isn't available — already reported once, up front) —
                # stay silent per-entity rather than repeating that
                # explanation on every single affected entity.
                static_model_time += time.monotonic() - _dbg_fmt_t0

            elif model_key and not is_self_reference and model_key.lower() in model_decls:
                _dbg_fmt_t0 = time.monotonic()
                # A bare identifier matching a "model NAME { ... }" decl
                # found in the .def files — idTech4's indirection for MD5
                # (skeletal) models: unlike .ase/.lwo, a .map entity's
                # "model" key never names an .md5mesh file directly, it
                # names one of these decls instead (see
                # parse_model_decls). model_decls is only ever populated
                # when import_md5_models is on AND the MD5 addon is
                # available (see the top of this generator), so reaching
                # this branch already implies both — no need to re-check
                # either flag here.
                decl = model_decls[model_key.lower()]
                mesh_rel = decl.get('mesh')
                if not mesh_rel:
                    model_skip += 1
                    md5_model_skip += 1
                    issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                                  f"model decl \"{model_key}\" has no \"mesh\" line "
                                  f"— nothing to import")
                else:
                    resolved = resolve_model_path(filepath, mesh_rel, search_roots)
                    if resolved is None:
                        model_skip += 1
                        md5_model_skip += 1
                        issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                                      f"MD5 mesh file not found: \"{mesh_rel}\" (from "
                                      f"model decl \"{model_key}\"; searched Game/Mod "
                                      f"Root and the .map file's own ancestor "
                                      f"directories)")
                    elif resolved in md5_mesh_cache and md5_mesh_cache[resolved] is None:
                        # Cached failure from an earlier entity — the first
                        # entity to hit it already logged the real reason
                        # below, at the point the cache was populated.
                        model_skip += 1
                        md5_model_skip += 1
                    else:
                        _dbg_was_cached = resolved in md5_mesh_cache
                        # See place_attachment_specs for why this is
                        # popped rather than read, and put back below.
                        _dbg_prebuilt = md5_prebuilt_armatures.pop(ent_idx, None)
                        try:
                            _dbg_build_t0 = time.monotonic()
                            arm_obj, mesh_objs = _get_or_build_md5_instance(
                                md5_addon, md5_mesh_cache, resolved, scale,
                                class_col, ent_name, arm_obj=_dbg_prebuilt,
                                hide_sink=pending_hide)
                            _dbg_build_dt = time.monotonic() - _dbg_build_t0
                            md5_build_time += _dbg_build_dt
                            if _dbg_was_cached:
                                md5_build_cached_time  += _dbg_build_dt
                                md5_build_cached_count += 1
                            else:
                                md5_build_first_time  += _dbg_build_dt
                                md5_build_first_count += 1
                        except Exception as exc:
                            md5_build_time += time.monotonic() - _dbg_build_t0
                            if _dbg_prebuilt is not None:
                                md5_prebuilt_armatures[ent_idx] = _dbg_prebuilt
                            md5_mesh_cache[resolved] = None
                            model_skip += 1
                            md5_model_skip += 1
                            issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                                          f"exception loading MD5 model \"{model_key}\" "
                                          f"(\"{mesh_rel}\"): {exc}")
                        else:
                            rot_matrix = entity_rotation_matrix(entity)
                            model_origin = to_bl(origin_vec, scale) if origin_vec is not None else Vector((0.0, 0.0, 0.0))
                            target_world = Matrix.Translation(model_origin) @ rot_matrix

                            # idTech4 MD5 models place themselves via a
                            # dedicated "origin" bone, not the model's
                            # local (0,0,0) — the whole model must be
                            # shifted so THAT bone's rest POSITION, not
                            # the armature object's own local origin,
                            # ends up at the entity's placement. Case-
                            # insensitive lookup since real .md5mesh
                            # files aren't always authored with the
                            # joint literally lowercase "origin".
                            origin_bone = arm_obj.data.bones.get('origin')
                            if origin_bone is None:
                                for b in arm_obj.data.bones:
                                    if b.name.lower() == 'origin':
                                        origin_bone = b
                                        break

                            if origin_bone is not None:
                                # Shift the model (in its OWN local
                                # space) so the origin bone's rest
                                # position becomes the new local
                                # (0,0,0), THEN apply the entity's
                                # placement on top — the standard
                                # "rotate/place around an off-center
                                # pivot" composition. Deliberately only
                                # the bone's POSITION is used here, not
                                # its own rest ORIENTATION: the entity's
                                # rotation is applied directly to the
                                # model exactly like every other placed
                                # model type (.ase/.lwo/brushes)
                                # already does. Also canceling out the
                                # origin bone's own rest orientation
                                # (an earlier version of this code did,
                                # via a full matrix_local inversion)
                                # made every model's rotation wrong —
                                # most origin bones DO carry some rest
                                # orientation of their own that was
                                # never meant to be "undone" on
                                # placement, only compensated for
                                # position.
                                target_world = target_world @ Matrix.Translation(-origin_bone.matrix_local.translation)
                            # Falls back to the un-compensated placement
                            # (model's own local origin at the entity's
                            # placement) for any model with no "origin"
                            # bone at all — not every MD5 model has one.

                            parent_inv = Matrix.Translation(-empty.location)

                            # Armature and mesh piece(s) are placed
                            # identically — siblings under the entity's
                            # empty, all given the same transform — rather
                            # than mesh parented under armature. Either
                            # way the objects need the same matrix_world,
                            # which is what the Armature modifier (wired
                            # up inside _get_or_build_md5_instance) relies
                            # on: mesh vertex data and the armature's bone
                            # rest positions share the same local-space
                            # origin, so keeping every object's world
                            # transform identical is what keeps the mesh
                            # sitting correctly on its skeleton once moved
                            # to this entity's placement.
                            for obj in [arm_obj] + mesh_objs:
                                obj.parent = empty
                                obj.matrix_parent_inverse = parent_inv
                                obj.matrix_basis = target_world
                                all_objects.append(obj)
                            if skin_pairs:
                                apply_skin_material_swap(mesh_objs, skin_pairs, mat_cache)
                            model_ok += 1
                            md5_model_ok += 1

                            if import_md5_animations:
                                # The entity's own "anim" key wins, then
                                # the decl's/entityDef's defaults — see
                                # choose_decl_anim for the full priority.
                                (anim_id, anim_rel, explicit_anim,
                                 had_animN_pattern, used_af_pose) = choose_decl_anim(
                                    decl['anims'],
                                    entity.keys.get('anim', ''),
                                    _entitydef_keys_cached(classname))
                                if used_af_pose:
                                    af_pose_fallback_count += 1
                                # Every path that leaves anim_rel unset
                                # gets reported, but with different
                                # wording depending on whether there was
                                # something concrete that failed to
                                # resolve (the mapper set an explicit
                                # "anim" key that didn't match anything,
                                # or the classname's entityDef DOES
                                # declare a cinematic "animN" pattern
                                # whose values still didn't match any
                                # decl anim — both genuine data
                                # problems), or there was simply nothing
                                # to try at all (no "anim" key, no
                                # "idle"/"idleN"/"initial" in the decl,
                                # no "animN" pattern) — still worth
                                # surfacing (an MD5 model got placed
                                # with no animation on it at all), just
                                # not phrased as a mismatch.
                                if not anim_rel:
                                    if explicit_anim or had_animN_pattern:
                                        requested = entity.keys.get('anim', '').strip() or anim_id
                                        issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                                                      f"\"anim\" \"{requested}\" has no matching "
                                                      f"\"anim {requested} ...\" entry in model decl "
                                                      f"\"{model_key}\" — and the classname's entityDef "
                                                      f"has no numbered \"animN\" fallback that matches "
                                                      f"one either")
                                    else:
                                        issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                                                      f"MD5 model \"{model_key}\" placed, but no anim "
                                                      f"was found for it — no \"anim\" key on the "
                                                      f"entity, no \"idle\"/\"idle1\"-\"idle9\"/"
                                                      f"\"initial\" in the model decl, and the "
                                                      f"classname's entityDef declares no numbered "
                                                      f"\"animN\" fallback either")

                                if anim_rel:
                                    resolved_anim = resolve_model_path(filepath, anim_rel, search_roots)
                                    if resolved_anim is None:
                                        issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                                                      f"MD5 anim file not found: \"{anim_rel}\" "
                                                      f"(anim \"{anim_id}\")")
                                    elif resolved_anim in md5_anim_cache and md5_anim_cache[resolved_anim] is None:
                                        # Cached failure from an earlier
                                        # entity — the first entity to hit
                                        # it already logged the real
                                        # reason below, at the point the
                                        # cache was populated.
                                        pass
                                    else:
                                        try:
                                            # Bakes once and shares the
                                            # resulting Action across every
                                            # entity that resolves to this
                                            # same anim file — see
                                            # _get_or_apply_md5_anim for why
                                            # that's both safe and, for a
                                            # map with many repeated NPCs/
                                            # props, the single biggest
                                            # import-speed win available
                                            # here.
                                            _dbg_anim_t0 = time.monotonic()
                                            _get_or_apply_md5_anim(md5_addon, md5_anim_cache,
                                                                  resolved_anim, arm_obj, scale,
                                                                  md5_first_frame_only)
                                            md5_anim_time += time.monotonic() - _dbg_anim_t0
                                        except Exception as exc:
                                            md5_anim_time += time.monotonic() - _dbg_anim_t0
                                            md5_anim_cache[resolved_anim] = None
                                            issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                                                          f"exception applying MD5 anim "
                                                          f"\"{anim_id}\" (\"{anim_rel}\"): {exc}")

                            # The body is placed; now everything bound to
                            # it. Deliberately last, and deliberately only
                            # on the success path: a head binds to a bone
                            # of THIS armature, so there is nothing to
                            # attach to until the body model itself
                            # exists. resolve_attachment_specs already ran
                            # for this entity in the pre-pass above (which
                            # needed to know which extra armatures to
                            # batch); this call just reads that back.
                            _at_specs = entity_attach_specs(ent_idx, entity, classname)
                            if _at_specs:
                                _at_ok, _at_skip, _at_af = place_attachment_specs(
                                    ent_idx, ent_name, classname, class_col,
                                    arm_obj, decl, _at_specs)
                                attach_ok   += _at_ok
                                attach_skip += _at_skip
                                af_pose_fallback_count += _at_af
                md5_model_time += time.monotonic() - _dbg_fmt_t0

            elif model_key and not is_self_reference and model_from_def_fallback:
                # This model_key came from resolving the classname's
                # entityDef "model" key (following "inherit" — see
                # resolve_entitydef_keys), not straight from the
                # entity's own spawnarg. Unlike a bare identifier an
                # entity sets on ITSELF (routinely just brush-
                # bookkeeping — see the case-2 comment above), a value
                # pulled from an entityDef is always a deliberate
                # reference and SHOULD resolve to something loadable.
                # It matched neither a recognized static-model file
                # extension (.ase/.lwo) nor any parsed MD5 model decl
                # name, which usually means either the .def declaring
                # that decl wasn't found/scanned at all (wrong Game/Mod
                # Root, or "Parse defs"/"Import MD5 Models" left off)
                # or a genuine typo/omission in the content itself.
                model_skip += 1
                issues.append(f"entity {ent_idx} ({classname} \"{ent_name}\"): "
                              f"classname's entityDef resolves \"model\" to "
                              f"\"{model_key}\", but that's neither a "
                              f"recognized static-model file (.ase/.lwo) nor "
                              f"a known MD5 model decl name — check that "
                              f"\"Parse defs\" and \"Import MD5 Models\" are "
                              f"both on and the Game/Mod Root is correct")

        # Brushes on a non-worldspawn entity with an origin always get that
        # origin added (dmap's AdjustEntityForOrigin, unconditional) — let
        # it reach the mesh via an identity parent-inverse. Otherwise
        # (worldspawn, or no origin key) fall back to child_parent_inverse
        # decided above — also Identity, since the synthetic bbox-center
        # pivot case already baked its offset into the geometry itself
        # rather than relying on parent-inverse cancellation.
        if origin_vec is not None and not is_worldspawn:
            brush_parent_inverse = Matrix.Identity(4)
        else:
            brush_parent_inverse = child_parent_inverse

        # Patches never get origin added — see the comment block above —
        # but their mesh data was already shifted to be local to
        # empty.location (real origin or synthetic bbox-center alike) in
        # the "no origin_vec"/patch-shift step earlier, specifically so
        # their Object pivot would land there instead of at world
        # (0,0,0). Identity here is what lets that reach the Object,
        # mirroring brush_parent_inverse's own Identity case above. For
        # worldspawn specifically brush_parent_inverse is always this
        # same Identity too (is_worldspawn forces the else branch above,
        # and child_parent_inverse is always Identity) — which is what
        # makes combining brush and patch geometry into one shared
        # Object below safe: both already sit in the same local space.
        patch_parent_inverse = Matrix.Identity(4)

        # worldspawn_geo_grouping ('ALL'/'BRUSHES'/'PATCHES'/'NONE', UI
        # dropdown on the operator) controls how much of worldspawn's
        # brush/patch geometry gets combined into shared mesh Objects
        # instead of one Object per brush/patch — worldspawn routinely
        # owns thousands of them, and a separate Object each is both
        # wasteful and unwieldy to work with. Never applies to non-
        # worldspawn entities. Vertices are never welded/merged across
        # (or within) the source brushes/patches — see combine_brush_
        # meshes.
        group_brushes = is_worldspawn and worldspawn_geo_grouping in ('ALL', 'BRUSHES')
        group_patches = is_worldspawn and worldspawn_geo_grouping in ('ALL', 'PATCHES')
        combine_together = is_worldspawn and worldspawn_geo_grouping == 'ALL'

        if combine_together and (built_brushes or built_patches):
            # 'ALL': brushes and patches merged into a single Object.
            _combine_geo_objects(
                built_brushes + built_patches,
                f"worldspawn_geo_e{ent_idx:04d}", f"geo_e{ent_idx:04d}",
                class_col, empty, brush_parent_inverse, mat_cache, all_objects,
            )
            brush_ok += len(built_brushes)
            patch_ok += len(built_patches)
        else:
            if group_brushes and built_brushes:
                _combine_geo_objects(
                    built_brushes,
                    f"worldspawn_brushes_e{ent_idx:04d}", f"brushes_e{ent_idx:04d}",
                    class_col, empty, brush_parent_inverse, mat_cache, all_objects,
                )
                brush_ok += len(built_brushes)
            else:
                for b_idx, mesh, mat_names in built_brushes:
                    for mn in mat_names:
                        mesh.materials.append(get_or_create_material(mn, mat_cache))

                    obj = bpy.data.objects.new(f"brush_e{ent_idx:04d}_{b_idx:04d}", mesh)
                    class_col.objects.link(obj)  # link to collection first
                    obj.parent = empty
                    obj.matrix_parent_inverse = brush_parent_inverse
                    all_objects.append(obj)
                    brush_ok += 1

            # ── patches ──────────────────────────────────────────────
            if group_patches and built_patches:
                _combine_geo_objects(
                    built_patches,
                    f"worldspawn_patches_e{ent_idx:04d}", f"patches_e{ent_idx:04d}",
                    class_col, empty, patch_parent_inverse, mat_cache, all_objects,
                )
                patch_ok += len(built_patches)
            else:
                for p_idx, mesh, mat_names in built_patches:
                    for mn in mat_names:
                        mesh.materials.append(get_or_create_material(mn, mat_cache))

                    obj = bpy.data.objects.new(f"patch_e{ent_idx:04d}_{p_idx:04d}", mesh)
                    class_col.objects.link(obj)
                    obj.parent = empty
                    obj.matrix_parent_inverse = patch_parent_inverse
                    all_objects.append(obj)
                    patch_ok += 1

        if entity_hidden:
            # Viewport-hide only (the outliner's eye icon / Alt+H, not
            # hide_render), same convention used for MD5 armature
            # placements elsewhere in this file. Blender doesn't cascade
            # a parent's hidden state to its children on its own, so
            # every Object this entity produced (empty, brush/patch
            # mesh, placed model, MD5 armature + mesh pieces — whichever
            # apply) needs hiding individually; entity_obj_start (set at
            # the top of this iteration) is what makes
            # all_objects[entity_obj_start:] exactly "every Object this
            # one entity produced", regardless of which of the branches
            # above actually ran. Queued rather than hidden here: the
            # whole import is parked outside the view layer at this
            # point and hide_set() needs one. See pending_hide.
            pending_hide.extend(all_objects[entity_obj_start:])

        yield (ent_idx, len(entities),
               f"entity {ent_idx + 1}/{len(entities)}: {classname}  |  "
               f"brushes {brush_ok} ok/{brush_skip} skip  |  "
               f"patches {patch_ok} ok/{patch_skip} skip  |  "
               f"models {model_ok} ok/{model_skip} skip"
               + (f"  |  {len(issues)} issue(s)" if issues else ""))

    # ── Unclaimed pre-built armatures ───────────────────────────────
    # The batching pre-pass builds an Armature for everything it expects
    # the loop above to place, and the loop pops each one as it claims
    # it. Anything still here was built for a model that then wasn't
    # placed after all - a decl whose .md5mesh the loop failed to load,
    # or an attachment whose joint turned out not to be a bone of the
    # skeleton it binds to (each already reported as its own issue).
    # Those are real Objects in a real collection: left behind they show
    # up in the outliner as empty, nameless-looking skeletons belonging
    # to nothing.
    for _leftover in md5_prebuilt_armatures.values():
        _leftover_data = _leftover.data
        try:
            bpy.data.objects.remove(_leftover, do_unlink=True)
            if _leftover_data is not None and _leftover_data.users == 0:
                bpy.data.armatures.remove(_leftover_data)
        except Exception:
            pass
    md5_prebuilt_armatures.clear()

    # ── Material creation (post-pass) ───────────────────────────────────
    # Runs AFTER every entity above has been placed, not before, and
    # scoped to the material names actually assigned somewhere on THIS
    # import's own objects — rather than every material the whole
    # Game/Mod Root's .mtr source tree happens to contain. A real mod's
    # materials/ folder can hold many thousands of decls; building all of
    # them unconditionally (an earlier version of this) meant potentially
    # building thousands of unused materials before the very first
    # per-entity progress message could even be shown, appearing to hang
    # with no feedback at all.
    #
    # Scanning all_objects' own material_slots directly — instead of
    # mat_cache (an earlier version of this) — is deliberate: mat_cache
    # only ever sees names get_or_create_material() was called with,
    # which covers brush/patch faces and static (.ase/.lwo) models, but
    # NOT MD5 models — those get their mesh materials from
    # md5_addon.build_mesh() internally, a completely separate code path
    # in the companion MD5 Tools addon that never touches mat_cache at
    # all. Scoping off mat_cache silently skipped every MD5 model's
    # materials, leaving them on whatever placeholder build_mesh() itself
    # assigns — exactly the "some models are white" symptom. Reading
    # material_slots directly instead sees every material actually in
    # use, however it got there, matching what manually running the
    # material addon's own "Generate Materials" (All in Scene) does.
    #
    # Every name is also run through the material addon's own
    # engine_canonical_decl() before use — critical for .ase/.lwo-sourced
    # materials specifically: idTech4_ase_lwo_io's ASE parser
    # deliberately keeps the source bitmap's file extension on the
    # Blender material name it derives (e.g. "...chair2.tga"), matching
    # what the raw *BITMAP path looks like, but .mtr decls are always
    # named WITHOUT one (idTech4's own engine truncates a decl name at
    # the last '.', and also folds backslashes and case — see that
    # function's docstring).
    # Brush/patch face material names never carry an extension in the
    # first place, so this never came up until static models (e.g. the
    # "moveable_*" props) were tested — every .ase-sourced name failed
    # to match any .mtr entry and silently stayed on its blank
    # placeholder. generate_materials()'s own ALL_SCENE scope already
    # does this same stripping; this mirrors it.
    material_addon = get_material_import_addon() if import_materials else None
    material_report_lines = []
    if material_addon is not None:
        # BOTH the object's slots and the mesh's own material list, because
        # they are not the same set once a skin is involved. A .skin swap
        # (apply_skin_material_swap) sets slot.link = 'OBJECT' and points the
        # SLOT at the replacement, deliberately leaving the shared Mesh
        # untouched so other entities placing the same model keep their own
        # material. From then on obj.material_slots[i].material returns the
        # override and the mesh-level original is invisible to it, while
        # obj.data.materials still holds the original and never sees the
        # override. Verified disjoint: in hell1.map every monster_zombie_boney
        # carries skins/monsters/zombies/adrianboney01, so a slots-only scan
        # saw adrianboney01 and never models/monsters/zombie/boney/boney -
        # which was therefore never built and stayed flat white on the mesh.
        # A mesh-only scan would make the opposite mistake and drop every
        # skinned material. The union is the set actually in use.
        # material_targets() does this same slots-and-mesh union, and also
        # hands back WHICH datablocks each decl name is used under - so the
        # build fills the ones the geometry points at instead of a fresh
        # datablock named after the decl.
        material_targets = material_addon.material_targets(all_objects)
        target_names = sorted(material_targets)
    else:
        material_targets = {}
        target_names = []

    if target_names:
        # base_directory/source_path already resolved once, up front
        # (see the top of this function) — reused here rather than
        # calling _resolve_sources a second time. source_path is only
        # ever blank here when base_directory is too (_resolve_sources
        # defaults source_path to base_directory's own "materials"
        # subfolder whenever base_directory is set) — a default that
        # doesn't actually exist falls through to the "no .mtr materials
        # were found" branch below instead, not this one.
        if not search_roots:
            material_issues.append(
                "Material import was requested, but Base Directory isn't "
                "set — no materials were created.")
        else:
            if not material_addon.material_names(source_path, base_directory,
                                                 mod_dir=mod_directory):
                # Name every folder that was actually looked in: with a
                # Mod Base set source_path is one per root, and naming
                # only the first would send the user checking a directory
                # that was never the whole story.
                _searched = '", "'.join(source_path) if isinstance(
                    source_path, (list, tuple)) else source_path
                material_issues.append(
                    f"Material import was requested, but no .mtr materials "
                    f"were found under \"{_searched}\" — no materials "
                    f"were created.")
            else:
                mat_base_dir = base_directory

                # The case-insensitive name fallback that used to live
                # here is now redundant: build_materials() resolves
                # through the database's own find(), which already
                # lowercases the key and strips a trailing extension the
                # way idTech4's decl system does. Table priming is gone
                # too - load_database() publishes the tree's tables to
                # the driver registry itself.

                # Material Preview viewport shading recompiles a
                # shader per material as it's (re)built, which gets
                # dramatically slower as the material count climbs.
                # Switch any 3D viewport currently in Material
                # Preview to Solid for the duration of the loop
                # below, and back afterward (even if a build raises).
                _preview_spaces = [
                    space
                    for window in context.window_manager.windows
                    for area in window.screen.areas if area.type == 'VIEW_3D'
                    for space in area.spaces
                    if space.type == 'VIEW_3D' and space.shading.type == 'MATERIAL'
                ]
                for _space in _preview_spaces:
                    _space.shading.type = 'SOLID'
                try:
                    # Yielding after every single material — an
                    # earlier version of this — turned out much
                    # slower than calling the material addon
                    # manually: each yield is a full round trip back
                    # through the modal operator (wait for the next
                    # timer tick, redraw the status bar), and most
                    # materials build in well under a millisecond,
                    # so that per-yield overhead dominated the whole
                    # pass once material count climbed into the
                    # hundreds. Yielding only every ~0.15s instead —
                    # still often enough that nothing looks hung —
                    # lets many materials build back-to-back between
                    # yields, the same way a synchronous call would,
                    # while still returning control to Blender
                    # regularly.
                    _last_yield_t = time.monotonic()
                    # Drive the material addon's own build loop rather than
                    # calling it once per material: it keeps the accounting
                    # (counts, diagnostics, names with no .mtr declaration)
                    # that this importer previously threw away, and parses
                    # the tree once for the whole pass.
                    _build = material_addon.build_materials(
                        target_names, mode=material_mode,
                        params=material_parameters, context=context,
                        base_dir=mat_base_dir, mod_dir=mod_directory,
                        source=source_path,
                        progress=True, targets=material_targets)
                    for _mi, _total, _name in _build:
                        _now = time.monotonic()
                        _is_last = (_mi == _total)
                        if _is_last or (_now - _last_yield_t) >= 0.15:
                            _last_yield_t = _now
                            yield (len(entities), len(entities),
                                   f"building materials: {_mi}/{_total}")
                    mat_summary = _build.summary
                    # publish_report(), not mat_summary.publish(): publish()
                    # alone fills the material panel's Report section but
                    # never writes the "idTech4 Material Report" datablock
                    # that section's Open Full Report button opens, so a map
                    # import left that button reporting "No report yet" while
                    # the panel above it was full of findings. The MATERIALS
                    # section of this importer's own report is a summary of
                    # the same pass, not a substitute - it is scoped to the
                    # map and truncated to fit beside the geometry issues.
                    material_addon.publish_report(mat_summary, context)
                    material_report_lines = mat_summary.lines()
                    for _missing in mat_summary.not_found:
                        material_issues.append(
                            f"\"{_missing}\" is used by geometry but is not "
                            f"declared in any .mtr, and no image of that name "
                            f"is on disk - that surface keeps its blank "
                            f"placeholder (renders flat white)")
                    for _err in mat_summary.errors:
                        material_issues.append(
                            f"exception building {_err}")
                finally:
                    for _space in _preview_spaces:
                        _space.shading.type = 'MATERIAL'

                # Bring the material addon's own panel state in line with
                # what was just generated, so re-opening it doesn't show
                # a stale mode or an out-of-date Editor Textures list:
                #  - generation_mode drives that panel's table/parameter
                #    list visibility (only relevant for Standard) — left
                #    on whatever it was previously set to, it would show
                #    or hide the wrong controls for what's actually in
                #    the scene now.
                #  - parameter_policy the same way: the panel's shader
                #    parm sliders only do anything for materials built
                #    under DYNAMIC, so it has to agree with what this
                #    import actually built or the sliders lie.
                #  - refresh_editor_textures() rescans for editor-
                #    texture-only materials (qer_editorimage-only —
                #    clip volumes and the like) and resyncs each one's
                #    visibility/opacity with the panel's Global settings;
                #    without it, the panel's hide/transparency controls
                #    don't know about any material just (re)built here.
                context.scene.idtech4_settings.generation_mode = material_mode
                context.scene.idtech4_settings.parameter_policy = material_parameters
                material_addon.refresh_editor_textures(context)

                # Switch to Material Preview so the just-built materials
                # are actually visible without a manual mode change.
                _mat_preview_areas = [a for a in context.screen.areas if a.type == 'VIEW_3D']
                if _mat_preview_areas:
                    for _space in _mat_preview_areas[0].spaces:
                        if _space.type == 'VIEW_3D':
                            _space.shading.type = 'MATERIAL'
                            break

    yield (len(entities), len(entities), "finishing up (selection, view framing, outliner)...")

    # ── reveal it ───────────────────────────────────────────────────
    # Un-park what the whole import was built into (see the exclude
    # above), then apply the hides that could not happen while it was
    # parked, then select — in that order, because select_set() on an
    # already-hidden object raises, which is what the try below has
    # always been swallowing.
    _unpark_layer_collections()
    for o in pending_hide:
        try:
            o.hide_set(True)
        except RuntimeError:
            pass

    for o in all_objects:
        try:
            o.select_set(True)
        except RuntimeError:
            pass

    view3d_areas = [a for a in context.screen.areas if a.type == 'VIEW_3D']
    if view3d_areas:
        with context.temp_override(area=view3d_areas[0]):
            try:
                bpy.ops.view3d.view_selected()
            except Exception:
                pass
    # Turn off "Relationship Lines" in every 3D viewport: with worldspawn
    # alone often owning thousands of brush/patch children, these dashed
    # parent->child lines turn the viewport into an unreadable tangle and
    # add no useful information for this kind of import.
    for area in view3d_areas:
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                try:
                    space.overlay.show_relationship_lines = False
                except Exception:
                    pass

    # Collapse the per-classname collections under the root, while
    # leaving the root collection (named after the imported file) itself
    # expanded. Confirmed empirically: calling collapse 4x collapsed
    # BOTH the root collection and everything under it — show_one_level
    # appears to close the DEEPEST currently-expanded level first and
    # only work shallower on repeated calls, so a single call should
    # close just the deepest level (the per-classname collections)
    # without touching the root collection's own row. If this needs
    # tuning (e.g. it turns out to need 2 calls instead of 1, or the
    # direction is reversed), adjust the count below.
    collapse_outliner_levels(context, 1)

    # The MAP half of the report: what came out of this .map file — its
    # brushes, its patches, the models its entities reference. The MATERIALS
    # half is built below, and MapImportReport keeps the two apart so each
    # can be shown with its own issues rather than with the other's counts.
    #
    # One line per geometry/asset type instead of one long concatenated line
    # — the combined line got hard to scan once a map had several of these
    # categories at once.
    report_lines = []

    brush_line = f"  Brushes: {brush_ok} imported"
    if brush_skip:
        brush_line += f", {brush_skip} skipped"
    report_lines.append(brush_line)

    if patch_ok or patch_skip:
        patch_line = f"  Patches: {patch_ok} imported"
        if patch_skip:
            patch_line += f", {patch_skip} skipped"
        report_lines.append(patch_line)

    if model_ok or model_skip:
        model_line = f"  Models: {model_ok} imported"
        if model_skip:
            model_line += f", {model_skip} failed/missing"
        report_lines.append(model_line)
        # Per-format breakdown — the combined total above doesn't say
        # whether those are static (.ase/.lwo) or MD5 models, which
        # matters a lot for import time (MD5 costs far more per model:
        # each needs its own Armature, and possibly an anim bake).
        if static_model_ok or static_model_skip:
            part = f"    Static (.ase/.lwo): {static_model_ok} imported"
            if static_model_skip:
                part += f", {static_model_skip} failed"
            part += f" in {static_model_time:.1f}s"
            report_lines.append(part)
        if md5_model_ok or md5_model_skip:
            part = f"    MD5: {md5_model_ok} imported"
            if md5_model_skip:
                part += f", {md5_model_skip} failed"
            if af_pose_fallback_count:
                part += f", {af_pose_fallback_count} used the \"af_pose\" fallback anim"
            # Which of the three animation modes ran is not otherwise
            # visible after the fact — a model posed at frame 1 and one
            # left in its bind pose look equally static in the outliner.
            part += {
                'FIRST_FRAME': ", posed at frame 1 (no Actions kept)",
                'NONE':        ", no animation applied",
            }.get(md5_animation_mode, "")
            part += (f" in {md5_model_time:.1f}s "
                     f"[{md5_build_time:.1f}s armature/mesh "
                     f"({md5_build_first_count} first-build {md5_build_first_time:.1f}s, "
                     f"{md5_build_cached_count} cached {md5_build_cached_time:.1f}s), "
                     f"{md5_anim_time:.1f}s anim]")
            report_lines.append(part)

        if attach_ok or attach_skip:
            part = (f"    heads/attachments: {attach_ok} placed"
                    + (f", {attach_skip} failed" if attach_skip else ""))
            if copy_joint_counts[0] or copy_joint_counts[1]:
                part += (f"; {copy_joint_counts[0]} head joint(s) driven "
                         f"from the body")
                if copy_joint_counts[1]:
                    part += (f", {copy_joint_counts[1]} left alone "
                             f"(the two rigs rest the joint differently)")
            report_lines.append(part)

    if prt_skip:
        # Particle systems are counted, not reported as failures — see
        # the prt_skip comment above: never a mesh this importer could
        # load regardless of format support, so an issue per referencing
        # entity would just be noise.
        report_lines.append(f"  {prt_skip} .prt particle definitions not "
                             f"loaded — unsupported by importer")

    if material_report_lines:
        # BuildSummary.lines() is already scoped to the materials this
        # import built, and is deliberately unindented and untruncated — it is
        # indented into place here and nowhere else.
        material_report_lines = ["  " + line for line in material_report_lines]
    else:
        material_report_lines = [f"  {len(mat_cache)} material(s) in use by "
                                 f"the imported geometry (none were "
                                 f"generated)"]

    return MapImportReport(f"OK: {len(entities)} entities",
                           report_lines, issues,
                           material_report_lines, material_issues)


def import_map(context, filepath, scale=SCALE, import_models=True,
                lwo_first_layer_only=True, parse_defs=True,
                import_materials=True, material_mode='SIMPLE',
                material_parameters='BAKED', model_shading='ENGINE',
                worldspawn_geo_grouping='ALL'):
    """
    Synchronous wrapper around import_map_generator() — drives it to
    completion in one call and returns the same MapImportReport
    import_map_generator() itself returns (see its docstring).
    Runs the entire import in one uninterrupted call with no
    progressive status updates. Kept for any caller that doesn't need
    progressive UI updates (e.g. driving this addon's import from the
    Python console or another script, rather than through
    IMPORT_OT_idtech4_map's modal UI).
    """
    gen = import_map_generator(context, filepath, scale=scale, import_models=import_models,
                               lwo_first_layer_only=lwo_first_layer_only, parse_defs=parse_defs,
                               import_materials=import_materials, material_mode=material_mode,
                               material_parameters=material_parameters,
                               model_shading=model_shading,
                               worldspawn_geo_grouping=worldspawn_geo_grouping)
    while True:
        try:
            next(gen)
        except StopIteration as e:
            return e.value


# ─────────────────────────────────────────────────────────────────────
#  OPERATOR
# ─────────────────────────────────────────────────────────────────────

def show_map_import_report(context, title, report):
    """
    Show the detailed .map import report in a dedicated Text Editor
    window, rather than a small popup. A popup (invoke_popup) turned out
    to be the wrong tool for this: long issue messages got truncated
    with no way to see the rest, there was no scrollbar for a long
    issue list (mouse-wheel only), and no close/OK button (click-outside
    to dismiss). Blender's built-in Text Editor solves all three
    natively: it wraps long lines instead of cutting them off (word wrap
    is turned on below), it scrolls a real text buffer of any length,
    the window resizes like any other window, and closing it is just...
    closing the window, same as closing any other window. It also
    leaves a real bpy.data.texts datablock behind, so the report can be
    reopened later from any Text Editor's datablock browser, not just
    in the moment right after import.

    *title*: the report's heading line, and the name of the datablock it
    is written into — "idTech4 Map Import Report for mars_city1". The map's
    name is in it because that is what tells two reports apart: importing a
    second map used to overwrite the first one's report, and the datablock
    browser offered one row called "idTech4 Map Import Report" with no way
    to know which map it was about. Naming it after the map means each map
    keeps its own, and re-importing the SAME map still overwrites just that
    one rather than piling up .001 copies.
    *report*: the MapImportReport import_map_generator() returned. It
    formats its own body — one MAP section (counts, then the geometry/
    model/entity issues) followed by one MATERIALS section (counts, then
    the material issues), so each section's numbers sit directly above the
    lines that explain them.
    """
    text_block = bpy.data.texts.get(title)
    if text_block is None:
        text_block = bpy.data.texts.new(title)
    text_block.clear()

    text_block.write(report.body(title))

    # Open a new window and turn it into a Text Editor showing this
    # datablock. wm.window_new() duplicates the CURRENT window's whole
    # screen layout into a new OS window; only the FIRST area found in
    # that new window is converted to a Text Editor (the rest of that
    # duplicated layout is irrelevant here, but changing just one area's
    # type is enough — the person can resize/close the new window like
    # any other).
    try:
        bpy.ops.wm.window_new()
        new_window = context.window_manager.windows[-1]
        area = new_window.screen.areas[0]
        area.type = 'TEXT_EDITOR'
        space = area.spaces.active
        space.text = text_block
        space.show_word_wrap = True
        space.top = 0
    except Exception:
        # Best-effort: the text datablock itself still exists and holds
        # the full report even if opening a dedicated window for it
        # failed for some reason — it can still be viewed manually by
        # switching any area to a Text Editor and selecting "idTech4 Map
        # Import Report" from its datablock browser.
        pass


# The half of the two source-path tooltips below that is the same for
# both. Blender renders a text field's tooltip as the property's
# description followed by the field's own full value, which is the point
# of drawing these as locked fields rather than as labels: a path too
# long for the dialog's width is truncated on screen but shown whole on
# hover, and the sentence below says why it can't be typed over.
_SOURCE_DISPLAY_TIP = (
    "Source directories cannot be modified from the import dialog. To "
    "change the source directories, cancel the import and use the idTech4 "
    "side (N) panel, or derive from the imported file path")


def _get_import_base_display(self):
    """Read-only Base Directory for the import dialog's locked field.

    Same effective value _resolve_sources would pick for this import,
    minus the derive-from-model walk (which needs a chosen .map file and
    is what the field greys out for anyway): the sources gate's one-off
    override if there is one, else the shared config's Base Directory.

    A StringProperty given a `get` but no `set` is read-only at the RNA
    level, which is what draws the field locked — the same pattern the
    Sources panel's own fields use (see _get_shared_base_display)."""
    shared_base, _mod, _ = get_shared_paths()
    return (self.override_base_directory or shared_base) or "(not set)"


def _get_import_mod_base_display(self):
    """Read-only Mod Base for the import dialog's locked field — see
    _get_import_base_display. A gate override replaces the configured
    Base outright and so replaces the Mod Base that went with it, which
    is why an override reads as no mod rather than as the stored one:
    that is exactly what _resolve_sources will do."""
    _base, shared_mod, _ = get_shared_paths()
    if self.override_base_directory:
        return "(none - overridden for this import)"
    return shared_mod or "(none - using Base Directory only)"


def _get_import_source_display(self):
    """Read-only Materials Source for the import dialog's locked field —
    see _get_import_base_display. Spells out the unset-but-defaulted case
    the way _resolve_sources resolves it, so the field shows the path
    that will actually be read rather than an empty box."""
    shared_base, shared_mod, shared_source = get_shared_paths()
    source = self.override_source_path or shared_source
    if source:
        return source
    base = self.override_base_directory or shared_base
    mod = '' if self.override_base_directory else shared_mod
    roots = shared_search_roots(base, mod)
    if roots:
        return "(defaults to %s)" % ", ".join(
            os.path.join(r, 'materials') for r in roots)
    return "(not set)"


# How long the modal import spends advancing the generator before handing
# control back to Blender (see IMPORT_OT_idtech4_map.modal).
#
# Every one of those returns costs time proportional to how many objects
# the VIEW LAYER holds — Blender re-evaluates the depsgraph, redraws the
# viewport and rebuilds the Outliner before the next TIMER event. Fewer
# returns is therefore straightforwardly cheaper, and the budget is what
# sets how many there are. Measured on airdefense1.map (2,195 entities,
# 8,761 objects, ~22s of actual import work), counting only the
# depsgraph half of each return:
#
#                       40ms budget     250ms budget
#     returns cost         19.0s            5.8s
#
# A tick is not "budget worth of work": the loop only checks the clock
# BETWEEN entities, and one entity can overshoot badly on its own (this
# map's worldspawn spends ~5.7s in a single step), so the real return
# count is far below work/budget — 108 at 40ms, 32 at 250ms.
#
# Since 1.10.4 the import also builds outside the view layer, which
# takes those same numbers to ~0.1s and ~0.0s: the two are independent
# and both are worth having, because parking cannot cover the Outliner
# still listing the collections and this is the only lever that reduces
# the number of redraws at all.
#
# The counterweight is only responsiveness, and there is a lot of slack
# in it: this is a status line and an ESC check, not an interactive tool.
# 250ms still repaints the progress message four times a second and
# still cancels within a quarter second of the key — neither is a
# latency a person reads as "stalled", which is the whole reason the
# import runs modally at all.
_TICK_BUDGET = 0.25


class IMPORT_OT_idtech4_map(bpy.types.Operator, ImportHelper, ImportFileGuardMixin,
                            MD5_ImportScaleRotMixin):
    """Import an idTech4 / Quake 3 / Valve 220 .map file"""
    bl_idname  = "import_scene.idtech4_map"
    bl_label   = "Import idTech4 .map"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".map"
    filter_glob: StringProperty(default="*.map", options={'HIDDEN'}, maxlen=255)

    worldspawn_geo_grouping: EnumProperty(
        name="Worldspawn Geo Grouping",
        description="Worldspawn routinely owns thousands of brushes/"
                    "patches — how much of that geometry to combine into "
                    "shared mesh objects instead of importing one object "
                    "per brush/patch",
        items=[
            ('ALL', "All", "Combine all worldspawn brushes and patches "
             "together into a single mesh object"),
            ('PATCHES', "Patches", "Combine all worldspawn patches into "
             "one mesh object; brushes stay one object each"),
            ('BRUSHES', "Brushes", "Combine all worldspawn brushes into "
             "one mesh object; patches stay one object each"),
            ('NONE', "None", "No grouping — every worldspawn brush and "
             "patch becomes its own object"),
        ],
        default='ALL',
    )

    #scale_factor: FloatProperty(
    #    name="Scale",
    #    description="Unit scale. Default 0.0254 = game inches to metres.",
    #    default=SCALE, min=0.000001, max=1000.0, precision=6,
    #)

    import_models: BoolProperty(
        name="Import Static Models (.ase / .lwo)",
        description="Load and place the .ase / .lwo meshes referenced by "
                    "entities' \"model\" key (func_static, misc_model, ...), "
                    "using each entity's origin/rotation",
        default=True,
    )
    lwo_first_layer_only: BoolProperty(
        name="Only import 1st layer of .lwo models",
        description="For referenced .lwo static models, import only the "
                    "first layer in the file. This matches real idTech4/"
                    "Doom 3 engine behavior: verified against the GPL "
                    "source (idRenderModelStatic::ConvertLWOToModelSurfaces"
                    "), the engine only ever reads the FIRST layer of a "
                    ".lwo — any additional layer is parsed but never "
                    "rendered. Disable to import every layer of every "
                    "referenced .lwo instead. Has no effect on .ase models "
                    "(.ase has no equivalent \"layer\" concept — every "
                    "GEOMOBJECT is always used)",
        default=True,
    )
    # The same Shading choice the standalone .ase/.lwo import dialog
    # offers, applied to every static model this import places. Item ids
    # and wording mirror idTech4_ase_lwo_io's SHADING_ITEMS, which is
    # what load_model_meshes(shading=...) is handed verbatim — duplicated
    # rather than imported for the same reason material_mode's list is:
    # that addon may not be installed/enabled when this class is defined,
    # and an EnumProperty's items have to exist at that point.
    model_shading: EnumProperty(
        name="Shading",
        description="How to shade the static models this import places",
        items=[
            ('ENGINE', "Engine Shading",
             "Shade the way idTech4 will actually render it. For a .lwo "
             "that is the file's SURF SMAN angle and PTAG SMGP groups "
             "(lwGetVertNormals); for an .ase it is *MESH_NORMALS, or -- "
             "when the file has none -- the normals the engine regenerates "
             "from geometry, which ignore *MESH_SMOOTHING entirely. With "
             "material generation on, a renderBump or unsmoothedTangents "
             "material is honoured too"),
            ('FILE', "File Shading",
             "Shade the way the file intends, which is what a non-idTech4 "
             "model wants. Identical to Engine Shading for every .lwo -- "
             "LWO stores no normals, so its smoothing angle is both -- and "
             "for any .ase that has *MESH_NORMALS. Differs only for an "
             ".ase without them, where this honours the *MESH_SMOOTHING "
             "groups the engine throws away"),
            ('SMOOTH', "Smooth",
             "No custom normals: every face smooth-shaded with Blender's "
             "own averaging. Matches no idTech4 asset -- Blender weights by "
             "corner angle and crosses material boundaries, the engine does "
             "neither -- but it is the only way to get an unconstrained "
             "mesh, since the two modes above write custom split normals "
             "that override Blender's shading controls"),
            ('FLAT', "Flat",
             "No custom normals, every face faceted. Same reasoning as "
             "Smooth: with custom normals applied you would otherwise have "
             "to clear them by hand to get here"),
        ],
        default='ENGINE',
    )
    parse_defs: BoolProperty(
        name="Parse defs",
        description="For entities with no explicit \"model\" key of their "
                    "own, look up their classname in the .def files' "
                    "entityDef declarations (following \"inherit\" chains) "
                    "and import whichever model that entityDef declares, "
                    "at the entity's position/orientation from the .map — "
                    "matching how idTech4 itself falls back to entityDef "
                    "defaults for spawnargs the entity doesn't override. "
                    "The def files are expected under a \"def\" folder, "
                    "found the same way models are: under the Game/Mod "
                    "Root if set, otherwise auto-detected from the .map "
                    "file's location",
        default=True,
    )
    apply_skins: BoolProperty(
        name="Apply Skins",
        description="For entities with a \"skin\" — checked in the same "
                    "order the real engine does: the .map entity's own "
                    "\"skin\" key, else (if unset) its classname's "
                    "entityDef \"skin\" (following \"inherit\" chains the "
                    "same way \"model\" does), else (for MD5 models, if "
                    "still unset) the \"skin\" line on the \"model NAME "
                    "{ ... }\" decl the entity's model resolves to — "
                    "swap specific materials on the model THAT entity "
                    "places (only that one instance — every other entity "
                    "placing the same model is unaffected). The swap "
                    "table comes from a \"skin NAME { old new  old new  "
                    "... }\" decl in a .skin file, found the same way "
                    ".def files are: under the Game/Mod Root if set, "
                    "otherwise auto-detected from the .map file's "
                    "location, under a \"skins\" folder",
        default=True,
    )
    import_md5_models: BoolProperty(
        name="Import MD5 Models",
        description="Load and place MD5 (skeletal) models referenced by "
                    "entities' \"model\" key. Unlike .ase/.lwo, a .map "
                    "entity never names an .md5mesh file directly — it "
                    "names a \"model NAME { ... }\" decl in a .def file "
                    "(found the same way as entityDef declarations), and "
                    "that decl's \"mesh\" line gives the actual .md5mesh "
                    "path. Each placed entity gets its own Armature (so "
                    "each can carry its own pose/animation independently) "
                    "but instances of the same underlying mesh data, "
                    "matching how .ase/.lwo models are already instanced",
        default=True,
    )
    import_attachments: BoolProperty(
        name="Heads & Attachments",
        description="Also place each character's head and whatever it "
                    "carries or wears. An idTech4 character is several "
                    "models, not one: the .map entity places only the "
                    "body, while its entityDef's \"def_head\" and "
                    "\"def_attach*\" keys name the head and each "
                    "attached prop (weapon, PDA, pauldron, hat, chair), "
                    "each bound to a named joint of the body's skeleton "
                    "with its own offset and rotation. Bone-parented, so "
                    "they follow the body through an imported animation. "
                    "Needs \"Import MD5 Models\" on — there is no "
                    "skeleton to bind anything to without it",
        default=True,
    )
    # How each placed MD5 model's animation is resolved is unchanged by this
    # setting — the entity's "anim" key, then the decl's "idle"/"idleN"/
    # "initial", then a numbered "animN", then "af_pose" — only what is DONE
    # with the resolved anim differs. Deliberately map-import only: the
    # File > Import > MD5 Anim dialogs always want a real Action, and a
    # single-pose option there would just be a broken import.
    md5_animation_mode: EnumProperty(
        name="MD5 Animations",
        description="What to do with the animation resolved for each placed "
                    "MD5 model. Has no effect with \"Import MD5 Models\" off",
        items=[
            ('FIRST_FRAME', "First Frame Only",
             "Pose each model at the first frame of its animation and keep "
             "no Action. Models stand in their idle pose but do not move, "
             "and nothing about them is re-evaluated as the timeline plays. "
             "Measured on mars_city1: 22 fps in the viewport against 13 fps "
             "for Full, and a much smaller .blend. Not the same as None — "
             "most models' first frame differs from their raw MD5 bind "
             "pose, so None leaves them visibly wrong"),
            ('FULL', "Full",
             "Bake the whole animation onto each model's Armature as an "
             "Action, so the models animate with the timeline. The faithful "
             "option, and by far the slowest to play back: every animated "
             "Armature makes Blender re-evaluate its pose and re-deform its "
             "meshes on every single frame change"),
            ('NONE', "None",
             "Place the models with no animation applied at all, leaving "
             "each in its raw MD5 bind pose. No faster than First Frame "
             "Only, and usually less accurate — any model whose idle pose "
             "differs from its bind pose is left visibly wrong"),
        ],
        default='FIRST_FRAME',
    )
    import_materials: BoolProperty(
        name="Import Materials",
        description="Build real, fully-textured Blender materials from "
                    "the .mtr source tree found under Materials below "
                    "(or derived from the .map file's own location), for "
                    "every brush/patch face's material name — instead of "
                    "the blank placeholder materials they'd otherwise "
                    "get. Needs the companion \"idTech4 Materials\" addon",
        default=True,
    )
    material_mode: EnumProperty(
        name="Material Mode",
        description="Which of the \"idTech4 Materials\" addon's own "
                    "generation modes to build every material with",
        items=[
            ('MAXIMUM', "Maximum", "The engine's own draw order: one "
             "Principled BSDF per interaction pass, engine-exact specular, "
             "cube maps and full ambient compositing. The reference rung, "
             "and the most expensive"),
            ('GOOD', "Good", "About 2x faster than Maximum. One merged "
             "interaction pass on a Diffuse BSDF, keeping normals, "
             "heightmaps and the roughness estimate; no specular highlight "
             "and no cube maps"),
            ('BASIC', "Basic", "About 2.2x faster, and loads no specular "
             "texture at all. Builds only the ambient stages that carry "
             "alpha - decals, overlays and cutouts - and caps the stack"),
            ('SIMPLE', "Simple", "About 2.9x faster: the diffuse texture on "
             "a lit Diffuse BSDF and nothing else. No normals, no "
             "heightmaps, two fewer textures resident per material"),
        ],
        default='SIMPLE',
    )
    # The other half of the material addon's own build settings: what to
    # do with every `time`/parm/global/sound/table term a material's
    # expressions reach. Listed here rather than left to whatever the
    # Materials panel happens to be set to, because it is often a bigger
    # performance lever than a whole fidelity mode is — 219 drivers were
    # 74ms of a 91ms frame on mars_city1 — and a whole-map import is
    # exactly the case where that matters most. Item ids match
    # idTech4_material_import's PARAMS_* constants, which is what gets
    # passed straight through to build_materials(params=...); duplicated
    # rather than imported for the same reason material_mode's list is
    # (the addon may not be installed when this class is defined).
    material_parameters: EnumProperty(
        name="Parameters",
        description="What the \"idTech4 Materials\" addon does with time, "
                    "parm0..11, global0..7, sound and table lookups while "
                    "building each material",
        items=[
            ('BAKED', "Baked", "Fold every expression once, now, against "
             "the current frame and the Materials panel's slider values. No "
             "drivers at all, so nothing re-runs per frame and nothing "
             "depends on the driver namespace surviving a file load. The "
             "mode to use for a whole-map import; moving a slider "
             "afterwards needs a rebuild"),
            ('DYNAMIC', "Dynamic", "Drivers for expressions that reach "
             "`time`, and a recorded expression re-folded on slider moves "
             "for the rest. Costs real frame time: on mars_city1, 219 "
             "drivers were 74ms of a 91ms frame"),
            ('SKIP', "Skip", "Refuse every parameter. Conditional stages "
             "are dropped without being evaluated and dynamic terms use "
             "their neutral defaults. The cheapest result, and the least "
             "faithful"),
        ],
        default='BAKED',
    )
    derive_from_model: BoolProperty(
        name="Derive from model",
        description="Ignore Base Directory below and instead derive it "
                    "for THIS import by walking up from the .map file's "
                    "own directory looking for a \"materials\" folder "
                    "(Base = that folder's parent)",
        default=False,
    )
    # Display only — read-only mirrors of what _resolve_sources will use,
    # drawn as locked text fields so a long path is truncated on screen
    # but readable in full on hover. Read-only at the RNA level (a `get`
    # with no `set`), so nothing here can be typed over or saved; the
    # editable copies live in the 3D Viewport sidebar's Sources panel.
    display_base_directory: StringProperty(
        name="Base",
        description="Base Directory this import will read models, defs, "
                    "skins and materials under. " + _SOURCE_DISPLAY_TIP,
        get=_get_import_base_display,
    )
    display_mod_base_directory: StringProperty(
        name="Mod Base",
        description="Optional Mod Base Directory, searched BEFORE Base "
                    "Directory for models, defs, skins and materials; "
                    "anything the mod does not supply falls back to Base "
                    "Directory. " + _SOURCE_DISPLAY_TIP,
        get=_get_import_mod_base_display,
    )
    display_materials_source: StringProperty(
        name="Materials",
        description="The .mtr file, or folder of them, this import will "
                    "build materials from. Search order: this path if set, "
                    "and nothing else; otherwise <Mod Base>/materials if a "
                    "Mod Base is set, then <Base>/materials. "
                    + _SOURCE_DISPLAY_TIP,
        get=_get_import_source_display,
    )
    # Internal, not shown as editable UI — set only by the sources gate
    # operator (IMPORT_OT_idtech4_map_sources_gate) when the user picks
    # "Select now" without also checking "Set as default", so this one
    # import still uses what they typed without it being written into
    # the shared idTech4 config file. See _resolve_sources.
    override_base_directory: StringProperty(default='', options={'HIDDEN', 'SKIP_SAVE'})
    override_source_path: StringProperty(default='', options={'HIDDEN', 'SKIP_SAVE'})
    # Internal — set by the sources gate when the user picks "Don't set
    # up materials for this import", to force Import Materials off AND
    # non-interactive for this one invocation (rather than just
    # defaulting it off, which the user could still re-check). Also
    # doubles as "the gate already ran and decided to proceed without a
    # Base Directory" for the def/skins/model-only case — see
    # _needs_sources_gate.
    materials_setup_skipped: BoolProperty(default=False, options={'HIDDEN', 'SKIP_SAVE'})
    # Internal — set by the sources gate's "...and save settings" derive
    # choice. Threaded through to _resolve_sources, which is the only
    # place the actual derived paths become known (this import's own
    # filepath isn't known until now, after the gate's popup has
    # already closed).
    save_derived_as_default: BoolProperty(default=False, options={'HIDDEN', 'SKIP_SAVE'})
    # Internal — set once by the sources gate right before it re-invokes
    # this operator with its resolved values, so execute() knows not to
    # evaluate _needs_sources_gate() a second time (which could re-fire
    # the gate forever if derive_from_model or Select Now still leaves
    # Base Directory blank — e.g. derive failing to find a "materials"
    # folder at all). The gate only ever runs once per import attempt,
    # regardless of what was actually resolved.
    gate_resolved: BoolProperty(default=False, options={'HIDDEN', 'SKIP_SAVE'})

    def draw(self, context):
        layout = self.layout
        self.draw_transforms(layout)
        #layout.use_property_split = False
        #layout.use_property_decorate = False
        #layout.prop(self, "scale_factor")
        layout.separator()
        layout.label(text="Worldspawn Geo Grouping:")
        layout.prop(self, "worldspawn_geo_grouping", text="")
        layout.separator()

        # Source Directories — shown up front, not tucked under Import
        # Materials below, since Base Directory now backs def/skins/
        # model resolution too (Import Static Models, Parse defs, Import
        # MD5 Models/Animations, Apply Skins), not just materials.
        layout.label(text="Source Directories")
        box = layout.box()
        # Derive from model heads the box because it decides what the
        # three fields under it are even showing. It must stay OUTSIDE
        # the greyed-out part: the fields grey out when it is checked
        # (they no longer describe the import), and a checkbox that
        # greys ITSELF out the moment it is ticked cannot be unticked.
        box.prop(self, "derive_from_model")
        paths = box.column()
        paths.enabled = not self.derive_from_model
        # Same shape as the Sources panel's own block: a label above a
        # locked field rather than one "Base: <path>" line. A label
        # never wraps and never truncates — it just ran off the dialog's
        # edge, taking the tail of the path (the part that identifies
        # it) with it. The field truncates instead, and hovering it
        # shows the whole path plus why it can't be edited here (see
        # _SOURCE_DISPLAY_TIP), which is what replaced the four
        # "To change these:" lines this used to spell out inline.
        col = paths.column(align=True)
        col.label(text="Base:")
        col.prop(self, "display_base_directory", text="")
        col = paths.column(align=True)
        col.label(text="Mod Base:")
        col.prop(self, "display_mod_base_directory", text="")
        col = paths.column(align=True)
        col.label(text="Materials:")
        col.prop(self, "display_materials_source", text="")
        layout.separator()
        # Static-model (.ase/.lwo) placement needs the companion
        # idTech4_ase_lwo_io addon — see get_model_import_addon(). If
        # it isn't installed/enabled, grey the option out entirely
        # rather than let the user enable something that will silently
        # do nothing (import_map_generator would otherwise have to
        # explain the same thing per-entity in the report).
        model_addon_available = get_model_import_addon() is not None
        row = layout.row()
        row.enabled = model_addon_available
        row.prop(self, "import_models")
        if not model_addon_available:
            layout.label(text="Needs the \"idTech4 .ase / .lwo Importer\" addon "
                              "(not installed/enabled)", icon='ERROR')
        col = layout.column()
        col.enabled = self.import_models and model_addon_available
        col.prop(self, "lwo_first_layer_only")
        col.prop(self, "parse_defs")
        # The standalone .ase/.lwo dialog's own Shading control, in the
        # same label-above-dropdown shape it uses there. Enabled with
        # the rest of this block: with static models off there is
        # nothing for it to shade (MD5 models carry their own normals
        # and never go through load_model_meshes).
        col.separator()
        col.label(text="Shading:")
        col.prop(self, "model_shading", text="")
        layout.separator()

        # MD5 (skeletal) model placement needs the companion MD5 Tools
        # addon — see get_md5_import_addon(). Same greyed-out treatment
        # as the static-model row above when it isn't available.
        md5_addon_available = get_md5_import_addon() is not None
        row = layout.row()
        row.enabled = md5_addon_available
        row.prop(self, "import_md5_models")
        if not md5_addon_available:
            layout.label(text="Needs the \"idTech4 MD5 tools\" addon "
                              "(not installed/enabled)", icon='ERROR')
        anim_row = layout.row()
        anim_row.enabled = self.import_md5_models and md5_addon_available
        anim_row.prop(self, "md5_animation_mode")
        attach_row = layout.row()
        attach_row.enabled = self.import_md5_models and md5_addon_available
        attach_row.prop(self, "import_attachments")
        layout.separator()
        layout.prop(self, "apply_skins")
        # Material creation needs the companion idTech4_material_import
        # addon — see get_material_import_addon(). Same greyed-out
        # treatment as the static-model/MD5 rows above when it isn't
        # available.
        material_addon_available = get_material_import_addon() is not None
        forced_off = material_addon_available and self.materials_setup_skipped
        row = layout.row()
        row.enabled = material_addon_available and not forced_off
        row.prop(self, "import_materials")
        if not material_addon_available:
            layout.label(text="Needs the \"idTech4 Materials\" addon "
                              "(not installed/enabled)", icon='ERROR')
        elif forced_off:
            layout.label(text="Materials setup was skipped for this import",
                         icon='INFO')
        col = layout.column()
        col.enabled = self.import_materials and material_addon_available and not forced_off
        # Label above each dropdown rather than beside it, matching the
        # Shading and Worldspawn Geo Grouping controls above: the two
        # enum names are long enough that Blender's own split gave the
        # dropdown itself almost no width in a normal-width dialog.
        col.label(text="Material Mode:")
        col.prop(self, "material_mode", text="")
        col.label(text="Parameters:")
        col.prop(self, "material_parameters", text="")

    _timer      = None
    _gen        = None
    _result     = None
    _finished   = False
    _start_time = None

    def _needs_sources_gate(self):
        """True if Base Directory is needed by something currently
        checked in this import and isn't already resolvable — i.e.
        showing the sources gate popup would actually accomplish
        something. Factors in each feature's own companion-addon
        availability, matching the same checks draw() uses to grey out
        its row: a checkbox left checked while greyed-out (its addon not
        installed) doesn't "need" anything, since it can't do anything
        either way. Materials Source is NOT checked here — it's
        optional, defaulting to Base Directory's own "materials"
        subfolder (see _resolve_sources), so its own absence never
        blocks anything on its own."""
        if self.materials_setup_skipped or self.gate_resolved:
            return False

        model_addon_available    = get_model_import_addon() is not None
        md5_addon_available      = get_md5_import_addon() is not None
        material_addon_available = get_material_import_addon() is not None

        needs_base = (
            (self.import_models and model_addon_available) or
            self.parse_defs or
            (self.import_md5_models and md5_addon_available) or
            (self.import_md5_models and self.md5_animation_mode != 'NONE'
             and md5_addon_available) or
            self.apply_skins or
            (self.import_materials and material_addon_available)
        )
        if not needs_base:
            return False

        base_directory, mod_directory, _ = _resolve_sources(
            self.filepath, self.derive_from_model,
            self.override_base_directory, self.override_source_path)
        # Roots, not base alone: a Mod Base on its own is a usable tree,
        # so prompting for a Base Directory the user deliberately left
        # out would be a gate that can never be satisfied.
        return not shared_search_roots(base_directory, mod_directory)

    def execute(self, context):
        # Refuse an unusable selection before anything else - before the
        # sources gate (which would otherwise pop a Base Directory dialog
        # for a path that was never going to open) and before the modal
        # timer below, which would otherwise return {'RUNNING_MODAL'} and
        # surface the failure from inside a timer tick.
        _paths, status = self.guard_input_files()
        if status:
            return status
        if self._needs_sources_gate():
            # Bail out without touching the scene and hand off to the
            # sources gate, passing this operator's own already-chosen
            # property values through _pending_import_kwargs (a module
            # global rather than properties mirrored onto the gate
            # class — self.as_keywords() already hands back every
            # property Blender knows how to re-apply via another
            # bpy.ops call, so there's no need to redeclare all ~15 of
            # them a second time just for passthrough). The gate
            # re-invokes this same operator via EXEC_DEFAULT (filepath
            # is already known — no need to reopen the file browser)
            # once it has resolved values, with gate_resolved=True so
            # this check is never evaluated twice for one import.
            global _pending_import_kwargs
            _pending_import_kwargs = self.as_keywords(ignore=('filepath',))
            bpy.ops.import_scene.idtech4_map_sources_gate(
                'INVOKE_DEFAULT', filepath=self.filepath)
            return {'CANCELLED'}

        # Runs modally rather than synchronously to completion: Blender
        # is single-threaded for UI/Python, so a long synchronous import
        # gives no opportunity for the status bar (or anything else) to
        # visibly update along the way. import_map_generator() yields
        # after each entity is processed; the modal() handler below
        # advances it in small time-boxed bursts on each timer tick,
        # letting Blender process events and redraw between bursts.
        self._gen = import_map_generator(
            context, self.filepath, scale=self.get_scale(),
            import_models=self.import_models,
            lwo_first_layer_only=self.lwo_first_layer_only,
            parse_defs=self.parse_defs,
            apply_skins=self.apply_skins,
            import_md5_models=self.import_md5_models,
            md5_animation_mode=self.md5_animation_mode,
            import_attachments=self.import_attachments,
            import_materials=self.import_materials,
            material_mode=self.material_mode,
            material_parameters=self.material_parameters,
            model_shading=self.model_shading,
            worldspawn_geo_grouping=self.worldspawn_geo_grouping,
            derive_from_model=self.derive_from_model,
            override_base_directory=self.override_base_directory,
            override_source_path=self.override_source_path,
            save_derived_as_default=self.save_derived_as_default,
        )
        self._result     = None
        self._finished   = False
        self._start_time = time.monotonic()

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        context.workspace.status_text_set("idTech4 .map import: starting...")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        # Defensive: shouldn't normally be reached (the timer is removed
        # before self._gen is cleared in _finish()), but a nested
        # bpy.ops call made from _finish() could in principle still pump
        # Blender's event loop and re-invoke modal() reentrantly on an
        # already-queued TIMER event before _finish() has fully
        # returned. self._finished (checked in _finish() itself) is the
        # primary guard against that; this is just a backstop so a
        # stray reentrant call can never crash on a cleared generator.
        if self._gen is None:
            return {'FINISHED'}

        if event.type == 'ESC':
            return self._finish(context, cancelled=True)

        if event.type != 'TIMER':
            return {'PASS_THROUGH'}

        # Time-box each tick: keep advancing the generator (each
        # next() processes one entity) until either it finishes or
        # _TICK_BUDGET of wall-clock time has been used, then hand
        # control back to Blender. Bounds how long a single tick can
        # block event processing/redraws, without limiting throughput to
        # "one entity per timer interval" on large maps.
        tick_deadline = time.monotonic() + _TICK_BUDGET
        try:
            while True:
                ent_idx, total, message = next(self._gen)
                if time.monotonic() >= tick_deadline:
                    break
        except StopIteration as e:
            self._result = e.value
            return self._finish(context, cancelled=False)
        except BaseException:
            # The import builds into a collection parked out of the view
            # layer and un-parks it on its way out; an exception from the
            # middle of the run never reaches that step, and would leave
            # the user looking at a scene that appears to have imported
            # nothing. Put it back before the traceback goes anywhere.
            _unpark_layer_collections()
            self._finish(context, cancelled=True)
            raise

        context.workspace.status_text_set(f"idTech4 .map import: {message}")
        return {'RUNNING_MODAL'}

    def _finish(self, context, cancelled):
        if self._finished:
            # Already ran once — see the reentrancy note on modal()'s
            # guard above. Nothing left to do on a second call.
            return {'CANCELLED'} if cancelled else {'FINISHED'}
        self._finished = True

        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        context.workspace.status_text_set(None)
        self._gen = None
        # A completed import has already put its own collection back, so
        # this is a no-op on the success path. It is here for ESC, which
        # drops the generator wherever it happened to be — with the
        # collection still parked out of the view layer, and everything
        # imported so far invisible.
        _unpark_layer_collections()

        if cancelled:
            self.report({'WARNING'}, "Import cancelled")
            return {'CANCELLED'}

        # self._result is the MapImportReport import_map_generator()
        # returned — see its docstring for what it holds.
        report = self._result or MapImportReport("")

        # Wall-clock duration of the whole modal import (start of
        # execute() to here) — the whole point of running modally is to
        # let Blender's event loop keep breathing between timer-driven
        # bursts, so this is the actual time a person waited, not just
        # CPU time spent inside import_map_generator() itself. Folded
        # into the result string so it shows up everywhere that string
        # does: the report window and the status-bar INFO/WARNING/ERROR
        # message below.
        #
        # Onto the headline, which is the one line that is about the
        # import as a whole; every other line in the report is about one
        # section of it.
        if self._start_time is not None:
            elapsed = time.monotonic() - self._start_time
            elapsed_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{int(elapsed // 60)}m {elapsed % 60:.1f}s"
            report.headline += f" (in {elapsed_str})"
        result = report.result

        if report.issue_count:
            # Auto-show the detailed report only when there's something
            # to report — a clean import shouldn't interrupt with an
            # extra window confirming nothing went wrong.
            #
            # Deferred via bpy.app.timers rather than called directly:
            # show_map_import_report() calls bpy.ops.wm.window_new(),
            # and calling ANY operator from inside this operator's own
            # modal()/_finish() can pump Blender's event loop and
            # re-invoke modal() reentrantly on an already-queued TIMER
            # event before this call has even returned — that's exactly
            # what previously crashed here with "TypeError: 'NoneType'
            # object is not an iterator" (self._gen already cleared,
            # above, by the time the reentrant call landed). Deferring
            # via a timer runs the window-creation call after this
            # operator has fully finished and Blender's call stack has
            # unwound — the same fix already proven for
            # collapse_outliner_levels earlier in this file.
            map_name = os.path.splitext(os.path.basename(self.filepath))[0]
            title = f"idTech4 Map Import Report for {map_name}"

            def _show_report(title=title, report=report):
                show_map_import_report(bpy.context, title, report)
                return None   # one-shot: don't reschedule

            bpy.app.timers.register(_show_report, first_interval=0.05)

        if result.startswith("ERROR"):
            self.report({'ERROR'}, result[6:])
            return {'CANCELLED'}
        if result.startswith("WARNING"):
            self.report({'WARNING'}, result[8:])
            return {'FINISHED'}
        suffix = (f" — {report.issue_count} issue(s), see report window"
                  if report.issue_count else "")
        self.report({'INFO'}, result[3:] + suffix)
        return {'FINISHED'}


# ─────────────────────────────────────────────────────────────────────
#  MESH POST-PROCESSING
# ─────────────────────────────────────────────────────────────────────

def apply_mesh_smoothing(mesh, smooth):
    """Set (or clear) the per-polygon smooth-shading flag on a built mesh
    — the same effect as Blender's Object > Shade Smooth. Duplicated in
    idTech4_ase_lwo_io.py for its own standalone .ase/.lwo import —
    see the "COMPANION ADDON DETECTION" comment near the top of this
    file for why each idTech4 addon keeps its own copy of small shared
    helpers rather than depending on another addon's internals.

    Skipped entirely for a mesh that already has custom split normals —
    a placed .lwo/.ase model built under Engine or File Shading
    (idTech4_ase_lwo_io.py's build_lwo_meshes/build_ase_meshes already
    applied them before this ever runs). Two separate reasons, both in
    that module's mark_smooth_for_custom_normals:

      - those meshes are ALREADY smooth-flagged, deliberately and
        throughout, so that Blender's mikktspace has continuous smooth
        fans to build a tangent basis over. A flat-flagged face is a fan
        of one, which gives a per-FACE tangent — invisible until a
        bumpmap is applied, at which point the model reads as faceted.
      - setting use_smooth here would CHANGE the normals anyway. Blender
        stores a custom normal as an offset relative to the fan frame it
        was written in, so flipping the flag afterwards re-decodes the
        same bytes against a different frame (43.9° mean on cone.ASE).

    Only affects .lwo/.ase-sourced meshes; brush/patch geometry never has
    custom normals from this addon, so this is a no-op there."""
    if mesh.has_custom_normals:
        return
    for poly in mesh.polygons:
        poly.use_smooth = smooth


# ─────────────────────────────────────────────────────────────────────
#  SOURCES GATE
# ─────────────────────────────────────────────────────────────────────
# File > Import points straight at IMPORT_OT_idtech4_map now — unlike
# the old design, this gate no longer runs in front of the menu entry.
# It only fires from INSIDE that operator's own execute(), and only once
# the .map file AND every checkbox are already chosen: execute() checks
# _needs_sources_gate() (does anything actually checked need Base
# Directory, factoring in which companion addons are even installed)
# before doing any scene work, and only launches this gate if something
# both needs it and doesn't already have it. That means the popup below
# no longer needs its own "is anything even needed" check — by the time
# it's shown, that's already been decided.
#
# self.filepath plus every one of the real operator's own already-set
# properties are handed off via _pending_import_kwargs (a plain module
# global — see execute()'s comment for why that's simpler here than
# mirroring ~15 properties onto this class just to receive them).
# _launch_target merges in whatever this popup resolved and re-invokes
# the real operator via EXEC_DEFAULT (filepath is already known, so
# there's no reason to reopen the file browser), with gate_resolved=True
# so execute() never re-evaluates the need a second time for the same
# import — that matters because Derive Automatically can still come back
# empty (no "materials" folder found walking up from the .map file) and
# Select Now's fields are optional to fill in Base Directory only; both
# are meant to proceed with whatever was resolved, not loop forever.

_pending_import_kwargs = {}


def _gate_base_directory_update(self, context):
    """Property update callback for manual_base_directory: the instant a
    Base Directory is picked (typed or browsed), auto-fill Materials
    from its own "materials" subfolder if one exists — but only when
    Materials is still blank, so this never overwrites something the
    user already entered."""
    if self.manual_source_path or not self.manual_base_directory:
        return
    candidate = os.path.normpath(os.path.join(
        bpy.path.abspath(self.manual_base_directory), 'materials'))
    if os.path.isdir(candidate):
        self.manual_source_path = candidate


class MaterialsSourceGateMixin:
    """Shared popup + dispatch logic for the gate operator below."""

    action: EnumProperty(
        name="",
        items=[
            ('MANUAL',      "Select now", ""),
            ('DERIVE',      "Derive automatically from the imported file's "
                            "location, just this time", ""),
            ('DERIVE_SAVE', "Derive automatically from the imported file's "
                            "location, and save as the default", ""),
            ('SKIP',        "Don't set up materials for this import", ""),
        ],
        default='MANUAL',
    )
    manual_base_directory: StringProperty(
        name="Base Directory",
        description="A standard Doom 3-style \"base\" directory — the "
                    "folder containing the game/mod's normal materials/, "
                    "models/, textures/, etc. subfolders",
        subtype='DIR_PATH',
        default='',
        update=_gate_base_directory_update,
    )
    manual_source_path: StringProperty(
        name="Materials (optional)",
        description="Where to search for .mtr material declarations. "
                    "Leave blank to use the \"materials\" subfolder under "
                    "the Base Directory above (standard setup) — only "
                    "set this to point somewhere else instead, e.g. a "
                    "different materials directory or one specific .mtr "
                    "file. These two paths are used on their own for this "
                    "import: a Mod Base configured in the Sources panel "
                    "pairs with the Base Directory stored there, not with "
                    "one entered here",
        subtype='FILE_PATH',
        default='',
    )
    set_as_default: BoolProperty(
        name="Set as default for future imports",
        description="Also save these into the shared idTech4 config "
                    "file (visible from the 3D Viewport sidebar's "
                    "idTech4 tab), so future imports don't need this "
                    "prompt",
        default=True,
    )

    def _target_op(self):
        """Subclass returns the bpy.ops.* callable for the real import
        operator this gate leads to."""
        raise NotImplementedError

    def _launch_target(self, context, **kwargs):
        global _pending_import_kwargs
        merged = dict(_pending_import_kwargs)
        merged.update(kwargs)
        merged['gate_resolved'] = True
        merged['filepath'] = self.filepath
        _pending_import_kwargs = {}
        self._target_op()('EXEC_DEFAULT', **merged)
        return {'FINISHED'}

    def invoke(self, context, event):
        # No "is it even needed" check here — the real operator's own
        # execute() already decided that before launching this gate at
        # all (see _needs_sources_gate), so showing the popup is always
        # the right call once this operator has been invoked.
        return context.window_manager.invoke_props_dialog(self, width=480)

    def draw(self, context):
        layout = self.layout
        layout.label(text="idTech4 Base Directory / Materials Source", icon='INFO')
        layout.label(text="not defined - needed for what you've")
        layout.label(text="selected to import.")
        layout.separator()
        # layout.prop(self, "action", expand=True) renders each enum
        # item's own long "name" as its button's text — that came out
        # blank in practice regardless of row/column layout, so each
        # choice is built explicitly here via prop_enum with its own
        # short, hardcoded button label instead (full description stays
        # on the enum item itself, still shown as this button's tooltip).
        # No explicit Cancel choice — invoke_props_dialog's own built-in
        # Cancel button (and Escape) already ends the operator without
        # calling execute() at all, which is exactly equivalent.
        col = layout.column(align=True)
        col.prop_enum(self, "action", 'MANUAL', text="Select Now")
        col.prop_enum(self, "action", 'DERIVE',
                      text="Derive Automatically (from the imported file's location) - just this time")
        col.prop_enum(self, "action", 'DERIVE_SAVE',
                      text="Derive Automatically (from the imported file's location) and save settings")
        col.prop_enum(self, "action", 'SKIP',
                      text="Skip - Don't Set Up Materials For This Import")
        if self.action == 'MANUAL':
            layout.separator()
            box = layout.box()
            box.label(text="Base Directory: a standard Doom 3-style \"base\" folder")
            box.label(text="(contains materials/, models/, textures/, etc.)")
            box.prop(self, "manual_base_directory")
            box.separator()
            box.label(text="Materials (optional): leave blank to use Base")
            box.label(text="Directory's own \"materials\" subfolder, or set this")
            box.label(text="to point somewhere else instead")
            box.prop(self, "manual_source_path")
            box.separator()
            box.prop(self, "set_as_default")

    def execute(self, context):
        if self.action == 'DERIVE':
            return self._launch_target(context, derive_from_model=True)

        if self.action == 'DERIVE_SAVE':
            return self._launch_target(context, derive_from_model=True,
                                        save_derived_as_default=True)

        if self.action == 'MANUAL':
            if not self.manual_base_directory:
                self.report({'ERROR'}, "Base Directory must be set.")
                return {'CANCELLED'}
            if self.set_as_default:
                set_shared_paths(base_directory=self.manual_base_directory,
                                  materials_mtr_source=self.manual_source_path)
            return self._launch_target(
                context,
                override_base_directory=self.manual_base_directory,
                override_source_path=self.manual_source_path)

        # SKIP
        return self._launch_target(
            context, import_materials=False, materials_setup_skipped=True)


class IMPORT_OT_idtech4_map_sources_gate(bpy.types.Operator, MaterialsSourceGateMixin):
    """Resolve Base Directory / Materials Source for a .map import already
    in progress (file and options already chosen), then re-launch it"""
    bl_idname  = "import_scene.idtech4_map_sources_gate"
    bl_label   = ".map Import : Base Directory / Materials Source Setup"
    bl_options = {'INTERNAL'}

    filepath: StringProperty(subtype='FILE_PATH', options={'HIDDEN', 'SKIP_SAVE'})

    def _target_op(self):
        return bpy.ops.import_scene.idtech4_map


# ─────────────────────────────────────────────────────────────────────
#  MAP EXPORT — WORLDSPAWN BRUSH GEOMETRY
# ─────────────────────────────────────────────────────────────────────
#
# The exact inverse of the brushDef3 half of this file's importer. Every
# rule below is idTech4's own, taken from the GPL source rather than from
# what the format looks like:
#
#   idMapBrush::Write (idlib/MapFile.cpp:485)
#       ( nx ny nz d ) ( ( m00 m01 m02 ) ( m10 m11 m12 ) ) "material" 0 0 0
#     with d = -(n · p) for points on the plane and n the OUTWARD normal
#     (idPlane::FromPoints), so a point inside the brush satisfies
#     n · p + d < 0. That is the same convention _face_from_plane parses
#     into (face.normal, face.d = -d_raw), read the other way round.
#
#   FS_WriteFloatString (framework/File.cpp:41)
#     A bare "%f" is NOT printf's %f. The engine rewrites it as "%1.10f"
#     and then strips trailing zeros and a trailing '.', which is why
#     shipped maps read "( 0 0 -1 48 )" and "Version 2" rather than
#     "( 0.000000 0.000000 -1.000000 48.000000 )" and "Version 2.000000".
#     _fs_float reproduces that byte for byte.
#
#   idMapBrushSide::GetTextureVectors / ComputeAxisBase (MapFile.cpp:67,93)
#     dmap turns the 2x3 texture matrix into two world-space planes and
#     reads texture coordinates straight off them
#     (usurface.cpp:203  st = xyz · texVec + offset). No texture size
#     enters anywhere, so recovering the matrix from a Blender UV map is a
#     plain affine solve in the face's (s,t) axis basis — see
#     solve_face_texmat.
#
#   RemoveDuplicateBrushPlanes / MAX_BUILD_SIDES (dmap/map.cpp:53,245)
#     dmap drops a duplicate plane with a console warning and throws the
#     whole brush away on a mirrored pair. Both are caught here instead,
#     where the object and the faces responsible can still be named.
#
# WHAT A BRUSH HAS TO BE. A brushDef3 is an intersection of half-spaces,
# so the only mesh that survives the round trip is a closed convex
# polyhedron. Those four conditions - closed manifold, every face flat,
# every vertex behind every face plane, at least four distinct planes -
# are not merely necessary but sufficient: a closed convex mesh IS the
# intersection of its own facet half-spaces, so a component that passes
# them re-imports as the same solid. Everything in validate_component
# exists to establish exactly that, and to name what failed when it does
# not.

# The engine's own limits, all from the Doom 3 GPL source. Exceeding any
# of them is a dmap failure rather than something the writer can fix, so
# they are checked here where the offending object can still be named.
BRUSH_MAX_SIDES         = 300        # MAX_BUILD_SIDES, dmap/map.cpp:53
BRUSH_MAX_WINDING_POINTS = 64        # MAX_POINTS_ON_WINDING, Winding.h:279
BRUSH_MAX_WORLD_COORD   = 128 * 1024 # MAX_WORLD_COORD, idlib/Lib.h:98

# Tolerances. Planarity and convexity are RELATIVE to the size of the
# component being judged, with an absolute floor, because the validator
# has to give the same verdict whether the scene was imported 1:1 (this
# addon's default) or scaled to metres — a fixed 0.01 game-unit epsilon
# is 40x looser in one than the other, and would pass geometry in metres
# that it failed in inches. The relative figure is picked to sit at
# dmap's own DIST_EPSILON (0.01) for a 64-unit brush.
BRUSH_PLANAR_REL_EPS  = 1e-4
BRUSH_PLANAR_ABS_EPS  = 1e-6
# ...and never tighter than the coordinates can express. A Blender mesh
# vertex is float32, so a coordinate near 25000 is quantised in steps of
# about 0.002 - and a five-unit brush sitting out there therefore cannot
# be flatter than that, whatever it was authored as. Judging it against
# extent * REL (0.0005) demands more precision than the input carries,
# which produced exactly one false "non-flat face" on altham.map: a
# perfectly ordinary six-sided box that nobody could have fixed, because
# there was nothing wrong with it.
#
# The slack of 2 covers the handful of operations between the stored
# coordinate and the fitted plane. Even at the far corner of the world
# this stays under dmap's own DIST_EPSILON of 0.01, so nothing the
# engine would call a separate plane is being waved through.
BRUSH_FLOAT32_ULP     = 2.0 ** -23      # 24-bit significand, ~1.19e-7
BRUSH_FLOAT32_SLACK   = 2.0
# Faces are clustered onto a shared plane before flatness is judged, so
# that a quad triangulated into two very slightly divergent triangles is
# reported as ONE non-flat surface rather than as two planes that then
# fail convexity. The angular tolerance only has to be loose enough to
# survive float32 vertex coordinates; the per-vertex distance test below
# is what actually decides flatness.
BRUSH_CLUSTER_DOT     = 0.999999     # ~0.081 degrees
# A face whose area is under this (relative to the component) has no
# trustworthy normal at all.
BRUSH_DEGENERATE_REL_AREA = 1e-10
# How far a UV may stray from the single affine projection being fitted.
# Blender stores UVs as float32, so a face whose coordinates run to ~100
# repeats already carries ~1e-5 of representation error before anything
# is fitted; 1e-3 leaves room for that without accepting a genuinely
# unwrapped (non-projected) island.
BRUSH_UV_EPS          = 1e-3
# Near-axis normal snapping. A box modelled on the grid in Blender still
# produces normals like (4e-8, 0, 1) after a matrix multiply, and writing
# those makes every axis-aligned brush in the file look hand-rotated.
BRUSH_NORMAL_SNAP_EPS = 1e-5

# The engine's fallback decl name: idMaterial::DefaultDefinition
# (renderer/Material.cpp:2693) maps "_default", the checkerboard. A face
# with no material assigned gets this rather than common/caulk, so an
# oversight is loudly visible in the editor and in game instead of
# silently invisible. It is reported either way.
BRUSH_DEFAULT_MATERIAL = '_default'

# The texture matrix used when a mesh carries no UV map at all: each
# game's own editor default, so the fallback matches what a brush created
# in DoomEdit (Version 2) or Quake 4's editor (Version 3) would carry.
BRUSH_DEFAULT_TEXMAT = {'2': 0.0625, '3': 0.03125}

EXPORT_MAP_VERSION_ITEMS = [
    ('2', "Version 2  (Doom 3 / Prey / Dark Mod)",
     "Write \"Version 2\" and end every brush side with the trailing "
     "\"0 0 0\", exactly as idMapBrush::Write and these games' editors do"),
    ('3', "Version 3  (Quake 4)",
     "Write \"Version 3\" and omit the trailing \"0 0 0\" after each "
     "side's material, matching Quake 4's own shipped maps"),
]

EXPORT_MAP_SOURCE_ITEMS = [
    ('WORLDSPAWN', "Worldspawn Collection",
     "Walk up from the active object (or the selection) to its root "
     "collection, then export the meshes in that root's \"worldspawn\" "
     "child collection. This is the layout a .map import produces. "
     "Refuses, naming what it did find, when there is no such collection"),
    ('ACTIVE_COLLECTION', "Active Collection",
     "Export every mesh in the collection highlighted in the outliner, "
     "and in its child collections, as worldspawn geometry. For scenes "
     "built by hand rather than imported"),
    ('SELECTED', "Selected Objects",
     "Export only the selected mesh objects as worldspawn geometry"),
]

EXPORT_MAP_SNAP_ITEMS = [
    ('NORMALS', "Normals Only",
     "Snap a plane normal to an exact axis when it is already within "
     "1e-5 of one, then re-derive the plane distance from the snapped "
     "normal. Vertex positions are never moved, so this cannot break "
     "geometry - it only stops an axis-aligned brush being written as "
     "\"( 0.00000004 0 1 -63.9999992 )\""),
    ('NORMALS_GRID', "Normals + Grid",
     "Normal snapping as above, and additionally round every vertex "
     "position to the grid size below before any plane is derived. "
     "WARNING: moving vertices can make a face that was flat non-flat, "
     "or a brush that was convex concave - anything the snap breaks is "
     "reported rather than written"),
    ('GRID', "Grid Only",
     "Round every vertex position to the grid size below before any "
     "plane is derived, and write plane normals exactly as they come "
     "out. WARNING: moving vertices can break flatness and convexity - "
     "anything the snap breaks is reported rather than written"),
    ('NONE', "None",
     "Write the transformed values exactly. Most faithful to the scene; "
     "axis-aligned brushes may read as very slightly rotated when "
     "reopened in a level editor"),
]

# Rotation presets, the same table (and the same meaning) as every other
# idTech4 addon's: a pure Z rotation applied to positions on the way out.
# EXPORT_ROTATION_ITEMS names them from the export side, so the preset is
# applied directly here rather than inverted.
ROTATION_PRESETS = {
    'NONE':   None,
    'X_TO_Y': Matrix.Rotation(math.radians(90), 4, 'Z'),
    'Y_TO_X': Matrix.Rotation(math.radians(-90), 4, 'Z'),
    'R180':   Matrix.Rotation(math.radians(180), 4, 'Z'),
}

_BLENDER_UNIQUIFIER_RE = re.compile(r'\.\d{3}$')


# ─────────────────────────────────────────────────────────────────────
#  ENGINE-FAITHFUL NUMBER FORMATTING
# ─────────────────────────────────────────────────────────────────────

def _fs_float(value):
    """One float, formatted the way idTech4 writes a bare "%f".

    FS_WriteFloatString (framework/File.cpp:41) intercepts a "%f" with no
    width or precision and prints "%1.10f" instead, then runs
    idStr::StripTrailing('0') followed by StripTrailing('.'). That is why
    a shipped .map says "( 0 0 -1 48 )" and "Version 2": the same code
    path writes the plane numbers, the texture matrix and the version
    line.

    Reproducing it is not cosmetic. It is what lets a re-export of an
    imported map diff against the original and show only the brushes that
    actually changed, instead of every number in the file.

    "0" survives StripTrailing intact ("0.0000000000" -> "0" -> "0").

    Negative zero is folded to "0" rather than written as "-0". That is
    not a departure from the engine - sprintf would render an actual
    -0.0f as "-0", and so would this if one arrived - but nothing in a
    shipped map ever contains one, while the plane and texture-matrix
    solves here produce them constantly from products like 0.0 * -1. A
    "-0" is the same number as "0" to every parser involved, so writing
    it would add nothing except a diff against the original file on
    lines that did not change.
    """
    # Guard the non-finite cases sprintf would render as "inf"/"nan": a
    # degenerate plane that reached this far would poison the whole file.
    if value != value or value in (float('inf'), float('-inf')):
        return '0'
    text = '%1.10f' % value
    text = text.rstrip('0').rstrip('.')
    # Folded AFTER formatting, not before: the values that render as a
    # signed zero here are mostly not -0.0 itself but residues like
    # -1e-12 out of the plane and texture-matrix solves, which "%1.10f"
    # renders as "-0.0000000000" and StripTrailing reduces to "-0". Both
    # spellings parse back to the same number, so the shorter one is
    # written and the file diffs clean against a map it round-tripped.
    if text in ('', '-', '0', '-0'):
        return '0'
    return text


def _strip_blender_uniquifier(name):
    """Drop Blender's ".001" duplicate-name suffix.

    Kept as a local copy rather than imported from idTech4_ase_lwo_io for
    the reason every shared helper in this file is: the addons install
    independently, and a cross-addon import would turn "installed
    alongside" into a hard dependency.
    """
    return _BLENDER_UNIQUIFIER_RE.sub('', name or '')


def brush_material_name(mat):
    """The decl string to write for a Blender material, and whether one
    had to be invented.

    Returns (name, missing). The datablock NAME wins, exactly as
    material_decl_for_export does in the .ase/.lwo exporter and for the
    same reason: assigning a material is how the user says which
    declaration a surface should use, so a rename has to be honoured
    rather than silently discarded, and an untouched import already
    carries the decl as its name.

    idtech4_material - what the material addon stamps on everything it
    builds, holding the .mtr decl it was built from - is the fallback for
    a datablock whose name has been reduced to nothing.

    The name is written VERBATIM. It is deliberately not put through
    engine_canonical_decl: that truncates at the last dot anywhere in the
    string, which is right for modelling a lookup and wrong for authoring
    one, and a map material's capitalisation is the mapper's business.
    """
    if mat is not None:
        name = _strip_blender_uniquifier(mat.name or '').strip()
        if name:
            return name, False
        stored = (mat.get('idtech4_material') or '').strip()
        if stored:
            return stored, False
    return BRUSH_DEFAULT_MATERIAL, True


# ─────────────────────────────────────────────────────────────────────
#  TEXTURE MATRIX RECOVERY
# ─────────────────────────────────────────────────────────────────────

def _solve3(rows, targets):
    """Solve a 3x3 system by Gaussian elimination with partial pivoting.
    Returns the three unknowns, or None if the matrix is singular."""
    m = [[rows[i][0], rows[i][1], rows[i][2], targets[i]] for i in range(3)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-15:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(3):
            if r == col:
                continue
            f = m[r][col] / m[col][col]
            for k in range(col, 4):
                m[r][k] -= f * m[col][k]
    return (m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2])


def _pick_spanning_triple(rows):
    """Indices of three rows of (s, t, 1) that span the plane as widely as
    possible.

    Any three non-collinear samples determine an affine map exactly, so
    the fit does not need least squares — but WHICH three decides how much
    float error the answer carries. Picking the widest triangle available
    (farthest from the centroid, then farthest from that point, then
    farthest from the line joining them) keeps the system well
    conditioned on the long thin faces that trim geometry is made of,
    where three adjacent vertices would be very nearly collinear.

    Returns None when every sample lies on one line, which means the face
    has no area in texture space.
    """
    n = len(rows)
    if n < 3:
        return None
    cs = sum(r[0] for r in rows) / n
    ct = sum(r[1] for r in rows) / n
    i0 = max(range(n), key=lambda i: (rows[i][0] - cs) ** 2 + (rows[i][1] - ct) ** 2)
    s0, t0 = rows[i0][0], rows[i0][1]
    i1 = max(range(n), key=lambda i: (rows[i][0] - s0) ** 2 + (rows[i][1] - t0) ** 2)
    s1, t1 = rows[i1][0], rows[i1][1]
    dx, dy = s1 - s0, t1 - t0
    span = math.sqrt(dx * dx + dy * dy)
    if span < 1e-12:
        return None
    # Perpendicular distance from the i0-i1 line, without the division.
    def cross(i):
        return abs(dx * (rows[i][1] - t0) - dy * (rows[i][0] - s0))
    i2 = max(range(n), key=cross)
    if cross(i2) / span < 1e-9:
        return None
    return i0, i1, i2


def solve_face_texmat(points, uvs, normal):
    """Recover a brushDef3 2x3 texture matrix from Blender UVs.

    *points* are game-space positions, *uvs* the Blender UVs at those
    positions, *normal* the face's outward normal. Returns
    (row0, row1, residual), or (None, None, None) when the samples do not
    span the plane.

    This inverts uv_brushdef3, which inverts the engine's own
    GetTextureVectors. With (S, T) = ComputeAxisBase(normal) and
    s = p·S, t = p·T, the engine's texture coordinates are

        u        = m00*s + m01*t + m02
        v_engine = m10*s + m11*t + m12

    and the importer stores Blender's V as 1 - v_engine, so the second
    row is fitted against (1 - v_blender). Both rows are therefore plain
    affine functions of (s, t): three non-collinear samples give them
    exactly.

    *residual* is the largest error, in texture-repeat units, over EVERY
    sample rather than the three that were solved from. That is the whole
    point of solving rather than least-squaring: a face whose UVs really
    are one flat projection fits to float precision, and anything else -
    an unwrapped island, a packed atlas, two coplanar faces with
    unrelated UVs - shows up as a residual the caller can refuse on.
    idTech4 has no way to store the second kind, so accepting a
    least-squares approximation of it would silently reshuffle the
    texturing rather than report it.
    """
    s_axis, t_axis = _face_axes(normal)
    rows = [(p.dot(s_axis), p.dot(t_axis), 1.0) for p in points]
    triple = _pick_spanning_triple(rows)
    if triple is None:
        return None, None, None
    i0, i1, i2 = triple
    sub = [rows[i0], rows[i1], rows[i2]]

    us = [uv[0] for uv in uvs]
    vs = [1.0 - uv[1] for uv in uvs]      # back to the engine's V

    row0 = _solve3(sub, [us[i0], us[i1], us[i2]])
    row1 = _solve3(sub, [vs[i0], vs[i1], vs[i2]])
    if row0 is None or row1 is None:
        return None, None, None

    residual = 0.0
    for (s, t, _one), u, v in zip(rows, us, vs):
        residual = max(residual,
                       abs(row0[0] * s + row0[1] * t + row0[2] - u),
                       abs(row1[0] * s + row1[1] * t + row1[2] - v))
    return row0, row1, residual


# ─────────────────────────────────────────────────────────────────────
#  SNAPPING
# ─────────────────────────────────────────────────────────────────────

def snap_plane_normal(normal, eps=BRUSH_NORMAL_SNAP_EPS):
    """Snap a normal already within *eps* of an axis onto that axis.

    Deliberately NOT idVec3::FixDegenerateNormal, which compares against
    0.0 and 1.0 exactly and so only ever repairs a normal that is already
    perfect in two components. What makes an exported box look
    hand-rotated is the residue of a matrix multiply — 4e-8 where 0
    belongs — which the engine's own function leaves alone.

    Vertices are never moved, so this cannot change which side of a plane
    anything is on by more than *eps* times the model's size: the plane
    distance is re-derived from the snapped normal by the caller.
    """
    x, y, z = normal.x, normal.y, normal.z
    sx = 0.0 if abs(x) < eps else (1.0 if abs(x - 1.0) < eps else
                                   (-1.0 if abs(x + 1.0) < eps else x))
    sy = 0.0 if abs(y) < eps else (1.0 if abs(y - 1.0) < eps else
                                   (-1.0 if abs(y + 1.0) < eps else y))
    sz = 0.0 if abs(z) < eps else (1.0 if abs(z - 1.0) < eps else
                                   (-1.0 if abs(z + 1.0) < eps else z))
    out = Vector((sx, sy, sz))
    length = out.length
    if length < PLANE_EPS:
        return normal
    return out / length


def snap_to_grid(value, grid):
    """Round one coordinate to the nearest multiple of *grid*."""
    if grid <= 0.0:
        return value
    return round(value / grid) * grid


# ─────────────────────────────────────────────────────────────────────
#  VALIDATION MODEL
# ─────────────────────────────────────────────────────────────────────
#
# Every check reports through one BrushIssue shape so the export report,
# the sidebar list and the Highlight Issue button all read the same
# records. The element indices an issue carries are indices into the
# object's own MESH, not into any temporary bmesh, which is what lets the
# panel select them in Edit Mode afterwards.

ISSUE_NON_PLANAR        = 'NON_PLANAR'
ISSUE_DEGENERATE_FACE   = 'DEGENERATE_FACE'
ISSUE_OPEN              = 'OPEN'
ISSUE_CONCAVE           = 'CONCAVE'
ISSUE_NOT_SOLID         = 'NOT_SOLID'
ISSUE_MIRRORED_PLANE    = 'MIRRORED_PLANE'
ISSUE_COPLANAR_MATERIAL = 'COPLANAR_MATERIAL'
ISSUE_UV_PROJECTION     = 'UV_PROJECTION'
ISSUE_TOO_MANY_SIDES    = 'TOO_MANY_SIDES'
ISSUE_TOO_MANY_POINTS   = 'TOO_MANY_POINTS'
ISSUE_OUT_OF_BOUNDS     = 'OUT_OF_BOUNDS'
ISSUE_NO_MATERIAL       = 'NO_MATERIAL'
ISSUE_NO_UV             = 'NO_UV'
ISSUE_INVERTED          = 'INVERTED'

# type -> (label, severity, select mode, icon). Ordered as the sidebar
# list and the report present them: the failures that stop a brush being
# written first, then the warnings that do not.
BRUSH_ISSUE_INFO = (
    (ISSUE_NON_PLANAR,        ("Non-flat face",              'ERROR',   'FACE', 'MESH_DATA')),
    (ISSUE_DEGENERATE_FACE,   ("Degenerate face",            'ERROR',   'FACE', 'MESH_DATA')),
    (ISSUE_OPEN,              ("Open or non-manifold edge",  'ERROR',   'EDGE', 'MOD_EDGESPLIT')),
    (ISSUE_CONCAVE,           ("Concave brush",              'ERROR',   'FACE', 'MOD_SOLIDIFY')),
    (ISSUE_NOT_SOLID,         ("Not a solid (under 4 sides)", 'ERROR',  'FACE', 'MESH_PLANE')),
    (ISSUE_MIRRORED_PLANE,    ("Mirrored plane pair",        'ERROR',   'FACE', 'MOD_MIRROR')),
    (ISSUE_COPLANAR_MATERIAL, ("Coplanar faces, 2+ materials", 'ERROR', 'FACE', 'MATERIAL')),
    (ISSUE_UV_PROJECTION,     ("UVs are not a flat projection", 'ERROR', 'FACE', 'UV')),
    (ISSUE_TOO_MANY_SIDES,    ("Over %d brush sides" % BRUSH_MAX_SIDES,
                                                             'ERROR',   'FACE', 'MESH_ICOSPHERE')),
    (ISSUE_TOO_MANY_POINTS,   ("Over %d points on a side" % BRUSH_MAX_WINDING_POINTS,
                                                             'ERROR',   'FACE', 'MESH_CIRCLE')),
    (ISSUE_OUT_OF_BOUNDS,     ("Outside the world bounds",   'ERROR',   'VERT', 'WORLD')),
    (ISSUE_NO_MATERIAL,       ("No material assigned",       'WARNING', 'FACE', 'MATERIAL')),
    (ISSUE_NO_UV,             ("No UV map",                  'WARNING', 'FACE', 'UV')),
    (ISSUE_INVERTED,          ("Normals face inward",        'WARNING', 'FACE', 'NORMALS_FACE')),
)
BRUSH_ISSUE_LOOKUP = dict(BRUSH_ISSUE_INFO)
BRUSH_ISSUE_ORDER = {k: i for i, (k, _v) in enumerate(BRUSH_ISSUE_INFO)}


class BrushIssue(object):
    """One problem, in one object, with the mesh elements responsible."""
    __slots__ = ('kind', 'object_name', 'component', 'indices', 'detail')

    def __init__(self, kind, object_name, component, indices=(), detail=''):
        self.kind = kind
        self.object_name = object_name
        self.component = component
        self.indices = list(indices)
        self.detail = detail

    @property
    def label(self):
        return BRUSH_ISSUE_LOOKUP[self.kind][0]

    @property
    def severity(self):
        return BRUSH_ISSUE_LOOKUP[self.kind][1]

    @property
    def select_mode(self):
        return BRUSH_ISSUE_LOOKUP[self.kind][2]

    @property
    def icon(self):
        return BRUSH_ISSUE_LOOKUP[self.kind][3]

    @property
    def is_fatal(self):
        return self.severity == 'ERROR'

    def describe(self):
        where = "%s [brush %d]" % (self.object_name, self.component)
        text = "%s: %s" % (where, self.label)
        if self.detail:
            text += " — " + self.detail
        if self.indices:
            shown = ', '.join(str(i) for i in self.indices[:8])
            more = '' if len(self.indices) <= 8 else ' +%d more' % (len(self.indices) - 8)
            text += "  (%s %s%s)" % (self.select_mode.lower(), shown, more)
        return text


class BrushSide(object):
    """One finished brushDef3 side, ready to write."""
    __slots__ = ('normal', 'dist', 'row0', 'row1', 'material')

    def __init__(self, normal, dist, row0, row1, material):
        self.normal = normal        # outward
        self.dist = dist            # the written d, i.e. -(n · p)
        self.row0 = row0
        self.row1 = row1
        self.material = material


# ─────────────────────────────────────────────────────────────────────
#  COMPONENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────

def brush_components(bm):
    """Split a bmesh into edge-connected components, as lists of faces.

    This is what recovers individual brushes from an import: the map
    importer's default worldspawn grouping combines every brush into one
    mesh object, and combine_brush_meshes deliberately never welds across
    brush boundaries, so each original brush survives as its own island
    even where two of them share a wall. Loose vertices and wire edges
    belong to no brush and are skipped rather than reported — nothing in
    a .map can carry them.
    """
    seen = set()
    out = []
    for face in bm.faces:
        if face.index in seen:
            continue
        stack = [face]
        seen.add(face.index)
        group = []
        while stack:
            cur = stack.pop()
            group.append(cur)
            for edge in cur.edges:
                for other in edge.link_faces:
                    if other.index not in seen:
                        seen.add(other.index)
                        stack.append(other)
        out.append(group)
    return out


def _component_bounds(faces):
    """(bounding box diagonal, box centre) for a component.

    The diagonal is what every relative tolerance is measured against.
    The centre is the local origin that _signed_volume and
    _fit_cluster_plane subtract before they add anything up: both are
    sums of products of coordinates, and a brush sitting 8000 units from
    the world origin — routine in a Dark Mod mission — makes those
    products around 1e11 while the answer they are converging on is
    around 1e5. Double precision loses the low six digits of every term,
    and what comes out the far end is noise. Measured on altham.map,
    computing the enclosed volume in world coordinates got the SIGN wrong
    on 185 of 13111 brushes, each of which was then written with its
    planes reversed and duly rejected as concave. Subtracting the centre
    first makes every term the size of the brush, and the cancellation
    goes away.
    """
    lo = [float('inf')] * 3
    hi = [float('-inf')] * 3
    for f in faces:
        for v in f.verts:
            for i in range(3):
                c = v.co[i]
                if c < lo[i]:
                    lo[i] = c
                if c > hi[i]:
                    hi[i] = c
    if lo[0] > hi[0]:
        return 0.0, Vector()
    diagonal = math.sqrt(sum((hi[i] - lo[i]) ** 2 for i in range(3)))
    centre = Vector(((lo[0] + hi[0]) * 0.5,
                     (lo[1] + hi[1]) * 0.5,
                     (lo[2] + hi[2]) * 0.5))
    return diagonal, centre


def _signed_volume(faces, origin):
    """Six times the signed volume enclosed by *faces*, by the divergence
    theorem, measured about *origin*.

    Negative means the winding — and therefore every face normal — points
    into the solid rather than out of it, which is what a mirrored object
    transform or an inside-out model produces. The sign has to be settled
    before any texture matrix is fitted, because ComputeAxisBase keys the
    whole (S, T) basis off the normal, so getting it wrong does not
    merely reverse the planes: it re-projects the textures too.

    Translating the solid to *origin* changes nothing about the enclosed
    volume — a closed surface's divergence integral is
    translation-invariant — but it changes everything about whether the
    arithmetic can find it. See _component_bounds.
    """
    total = 0.0
    for f in faces:
        verts = f.verts
        v0 = verts[0].co - origin
        for i in range(1, len(verts) - 1):
            total += v0.dot((verts[i].co - origin).cross(
                verts[i + 1].co - origin))
    return total


def _cluster_planes(faces, flipped, planar_eps, origin):
    """Group faces onto shared planes.

    Grouping BEFORE flatness is judged is what makes the diagnostics
    readable. A quad the user triangulated is two faces whose normals
    differ by float noise; treating them as two planes would report
    neither as non-flat and instead fail convexity somewhere else
    entirely, naming a vertex nowhere near the actual mistake. Clustered
    first, the pair is one surface, and "is this surface flat?" is asked
    once, of every vertex on it.

    Returns [(normal, dist, [face, ...]), ...] with the plane fitted to
    the whole cluster and the normal outward.

    The grouping pass uses bmesh's own face normals, which is all a
    same-plane-or-not decision needs; the plane itself is then re-fitted
    from the vertex coordinates by Newell's method in Python floats.
    That second pass is not redundant. BMFace.normal is computed and
    stored in single precision, and a brush side reached through it
    drifts by around a part in 1e5 - which on a small face a thousand
    units from the origin moves the written plane by a hundredth of a
    unit, right at dmap's own DIST_EPSILON. Newell over the same
    coordinates in double precision costs nothing and gives back most of
    that margin. It cannot give back all of it: the coordinates are
    float32 in the mesh, and that is the floor for any exporter reading a
    Blender mesh.
    """
    clusters = []       # [normal, d, faces, area]
    for f in faces:
        n = -f.normal if flipped else f.normal.copy()
        if n.length_squared < 0.5:      # bmesh gives a zero normal for a
            continue                    # degenerate face; handled elsewhere
        n.normalize()
        d = n.dot(f.verts[0].co)
        for c in clusters:
            if c[0].dot(n) >= BRUSH_CLUSTER_DOT and abs(c[1] - d) <= planar_eps:
                area = f.calc_area()
                total = c[3] + area
                if total > 0.0:
                    blended = c[0] * c[3] + n * area
                    if blended.length_squared > PLANE_EPS * PLANE_EPS:
                        c[0] = blended.normalized()
                    c[1] = (c[1] * c[3] + d * area) / total
                c[2].append(f)
                c[3] = total
                break
        else:
            clusters.append([n, d, [f], f.calc_area()])

    out = []
    for rough_normal, rough_d, group, _area in clusters:
        fitted = _fit_cluster_plane(group, rough_normal, origin)
        if fitted is None:
            out.append((rough_normal, rough_d, group))
        else:
            out.append((fitted[0], fitted[1], group))
    return out


def _fit_cluster_plane(group, rough_normal, origin):
    """(normal, distance) for one plane cluster, by Newell + mean offset.

    Newell's method sums a cross-product term per polygon EDGE, so it
    uses every vertex of every face rather than three of them, and
    summing it across the faces of a cluster weights them by area for
    free. The result is oriented against the cluster's rough normal
    because Newell's sign follows the winding, which a flipped component
    has already had reversed.

    Coordinates are taken relative to *origin* for the normal, for the
    reason spelled out in _component_bounds: Newell's (a.z + b.z) term
    carries the full magnitude of the coordinates while the sum it feeds
    is the size of the face, so a face far from the world origin loses
    most of its significant digits. The DISTANCE is then measured back in
    world coordinates, which is what the plane equation needs and where
    no cancellation happens - it is a mean, not a difference of large
    numbers.

    Returns None when the cluster has no usable area, leaving the caller
    with the rough plane it came in with.
    """
    nx = ny = nz = 0.0
    ox, oy, oz = origin.x, origin.y, origin.z
    for f in group:
        verts = f.verts
        count = len(verts)
        for i in range(count):
            a = verts[i].co
            b = verts[(i + 1) % count].co
            ax, ay, az = a.x - ox, a.y - oy, a.z - oz
            bx, by, bz = b.x - ox, b.y - oy, b.z - oz
            nx += (ay - by) * (az + bz)
            ny += (az - bz) * (ax + bx)
            nz += (ax - bx) * (ay + by)
    normal = Vector((nx, ny, nz))
    if normal.length < PLANE_EPS:
        return None
    normal.normalize()
    if normal.dot(rough_normal) < 0.0:
        normal.negate()
    seen = {}
    for f in group:
        for v in f.verts:
            seen[v.index] = v.co
    dist = sum(normal.dot(co) for co in seen.values()) / len(seen)
    return normal, dist


def validate_component(faces, object_name, component_index, uv_layer,
                       materials, snap_normals, want_sides=True):
    """Check one connected component, and build its brush sides.

    Returns (sides, issues). *sides* is None whenever any issue is fatal —
    the component is not written — but the issue list is always complete
    rather than stopping at the first failure, because a mesh with two
    unrelated mistakes in it should show both.

    *materials* is the object's resolved material-slot list (a slot may
    hold None). *uv_layer* may be None, which is a warning rather than a
    failure: the side gets the editor's own default texture matrix.

    The order of the checks below is the order in which one failure makes
    the next check meaningless. Flatness comes before convexity because a
    plane fitted through a non-flat face is fiction, and convexity before
    the texture solve because there is no point recovering UVs for a
    brush that cannot be written.
    """
    issues = []

    def add(kind, indices=(), detail=''):
        issues.append(BrushIssue(kind, object_name, component_index,
                                 indices, detail))

    extent, origin = _component_bounds(faces)
    # How flat a face has to be to count as flat, and how far a vertex may
    # sit in front of a plane before the brush is concave. Three floors,
    # and the largest wins: an absolute one, one relative to the size of
    # the brush, and one relative to how far from the world origin it sits
    # — see BRUSH_FLOAT32_ULP for why that last one is not optional.
    resolution = max(abs(origin.x), abs(origin.y), abs(origin.z))
    planar_eps = max(BRUSH_PLANAR_ABS_EPS,
                     extent * BRUSH_PLANAR_REL_EPS,
                     resolution * BRUSH_FLOAT32_ULP * BRUSH_FLOAT32_SLACK)

    # ── Degenerate faces ────────────────────────────────────────────
    # A zero-area face has no usable normal, so every later check would
    # be reading noise. Reported first and the component abandoned.
    area_floor = max(1e-18, (extent * extent) * BRUSH_DEGENERATE_REL_AREA)
    degenerate = [f.index for f in faces if f.calc_area() <= area_floor]
    if degenerate:
        add(ISSUE_DEGENERATE_FACE, degenerate,
            "%d face(s) have no area" % len(degenerate))
        return None, issues

    # ── Closed and manifold ─────────────────────────────────────────
    # A brush is an intersection of half-spaces: it has no boundary and
    # no edge shared by three surfaces. This is also the check that
    # catches an imported patch, which is an open sheet.
    edges = {}
    for f in faces:
        for e in f.edges:
            edges[e.index] = e
    bad_edges = [i for i, e in edges.items() if len(e.link_faces) != 2]
    if bad_edges:
        counts = {}
        for i in bad_edges:
            counts[len(edges[i].link_faces)] = counts.get(len(edges[i].link_faces), 0) + 1
        how = ', '.join("%d edge(s) with %d face(s)" % (n, k)
                        for k, n in sorted(counts.items()))
        add(ISSUE_OPEN, bad_edges, how)
        return None, issues

    # ── Winding direction ───────────────────────────────────────────
    volume = _signed_volume(faces, origin)
    flipped = volume < 0.0
    if flipped:
        add(ISSUE_INVERTED, [f.index for f in faces],
            "exported with the planes reversed so the brush is solid")

    # ── Plane clustering and flatness ───────────────────────────────
    clusters = _cluster_planes(faces, flipped, planar_eps, origin)
    non_planar = []
    worst_dev = 0.0
    for normal, dist, group in clusters:
        for f in group:
            for v in f.verts:
                dev = abs(normal.dot(v.co) - dist)
                if dev > planar_eps:
                    if f.index not in non_planar:
                        non_planar.append(f.index)
                    worst_dev = max(worst_dev, dev)
    if non_planar:
        add(ISSUE_NON_PLANAR, sorted(non_planar),
            "off its plane by up to %.6g (tolerance %.6g)" % (worst_dev, planar_eps))

    # ── Enough distinct planes to enclose a volume ──────────────────
    if len(clusters) < 4:
        add(ISSUE_NOT_SOLID, [f.index for f in faces],
            "%d distinct plane(s); a brush needs at least 4" % len(clusters))
    if len(clusters) > BRUSH_MAX_SIDES:
        add(ISSUE_TOO_MANY_SIDES, [f.index for f in faces],
            "%d sides; dmap's MAX_BUILD_SIDES is %d"
            % (len(clusters), BRUSH_MAX_SIDES))

    # ── Mirrored planes ─────────────────────────────────────────────
    # dmap throws the whole brush away on these (RemoveDuplicateBrushPlanes),
    # so there is no point writing one. In practice it means a zero-
    # thickness sheet that the manifold test let through.
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            if (clusters[i][0].dot(clusters[j][0]) <= -BRUSH_CLUSTER_DOT
                    and abs(clusters[i][1] + clusters[j][1]) <= planar_eps):
                add(ISSUE_MIRRORED_PLANE,
                    [f.index for f in clusters[i][2] + clusters[j][2]],
                    "two sides share one plane facing opposite ways")
                break
        else:
            continue
        break

    # ── Convexity ───────────────────────────────────────────────────
    # Every vertex behind every plane. Together with closed + flat, this
    # is what makes the half-space intersection reproduce the mesh.
    verts = {}
    for f in faces:
        for v in f.verts:
            verts[v.index] = v
    if len(clusters) >= 4 and not non_planar:
        offenders = []
        worst_out = 0.0
        for normal, dist, group in clusters:
            outside = False
            for v in verts.values():
                over = normal.dot(v.co) - dist
                if over > planar_eps:
                    outside = True
                    worst_out = max(worst_out, over)
            if outside:
                for f in group:
                    if f.index not in offenders:
                        offenders.append(f.index)
        if offenders:
            add(ISSUE_CONCAVE, sorted(offenders),
                "geometry sits up to %.6g in front of %d side(s)"
                % (worst_out, len(offenders)))

    # ── World bounds ────────────────────────────────────────────────
    out_of_bounds = [i for i, v in verts.items()
                     if max(abs(v.co.x), abs(v.co.y), abs(v.co.z))
                     > BRUSH_MAX_WORLD_COORD]
    if out_of_bounds:
        add(ISSUE_OUT_OF_BOUNDS, sorted(out_of_bounds),
            "beyond +/-%d game units" % BRUSH_MAX_WORLD_COORD)

    # ── Per-side material, winding size and texture matrix ──────────
    sides = []
    unassigned = []       # faces with no material, collected for one issue
    for normal, dist, group in clusters:
        face_ids = [f.index for f in group]

        # One plane, one material: a brushDef3 side has room for exactly
        # one. Faces that agree merge silently; faces that disagree are
        # refused rather than resolved, because picking a winner would
        # quietly retexture the level.
        names = []
        for f in group:
            slot = f.material_index
            mat = materials[slot] if 0 <= slot < len(materials) else None
            name, missing = brush_material_name(mat)
            if missing:
                # Gathered for ONE issue per brush rather than raised per
                # side. A mesh nobody has assigned a material to has it
                # missing on every face, and six identical rows for one
                # cube - thousands for a map - buries the findings that
                # differ from each other.
                unassigned.append(f.index)
            if name not in names:
                names.append(name)
        if len(names) > 1:
            add(ISSUE_COPLANAR_MATERIAL, face_ids,
                "coplanar faces carry %s" % ', '.join('"%s"' % n for n in names))
            continue
        material = names[0]

        # Distinct points on the merged side. dmap clips a fixed-size
        # winding against the other planes, and MAX_POINTS_ON_WINDING is
        # a hard ceiling rather than a warning.
        side_verts = {}
        for f in group:
            for v in f.verts:
                side_verts[v.index] = v.co
        if len(side_verts) > BRUSH_MAX_WINDING_POINTS:
            add(ISSUE_TOO_MANY_POINTS, face_ids,
                "%d points; MAX_POINTS_ON_WINDING is %d"
                % (len(side_verts), BRUSH_MAX_WINDING_POINTS))

        plane_normal = snap_plane_normal(normal) if snap_normals else normal
        if plane_normal is not normal:
            # Re-derive the distance from the SNAPPED normal rather than
            # keeping the one fitted to the unsnapped one, so the plane
            # still passes through the face instead of near it. Rotating
            # a normal by up to BRUSH_NORMAL_SNAP_EPS tilts the plane by
            # that much times the face's size, which stays an order of
            # magnitude inside the flatness tolerance the face was
            # already judged against.
            dist = (sum(plane_normal.dot(co) for co in side_verts.values())
                    / len(side_verts))

        if uv_layer is None:
            k = BRUSH_DEFAULT_TEXMAT['2']
            row0, row1 = (k, 0.0, 0.0), (0.0, k, 0.0)
        else:
            points = []
            uvs = []
            for f in group:
                for loop in f.loops:
                    points.append(loop.vert.co)
                    uvs.append(tuple(loop[uv_layer].uv))
            row0, row1, residual = solve_face_texmat(points, uvs, plane_normal)
            if row0 is None:
                add(ISSUE_UV_PROJECTION, face_ids,
                    "the UVs on this side have no area to solve from")
                continue
            if residual > BRUSH_UV_EPS:
                add(ISSUE_UV_PROJECTION, face_ids,
                    "off a single projection by %.6g texture units "
                    "(tolerance %.6g)" % (residual, BRUSH_UV_EPS))
                continue

        if want_sides:
            # The written d is the engine's: n · p + d = 0, so d = -(n · p).
            sides.append(BrushSide(plane_normal, -dist, row0, row1, material))

    if unassigned:
        add(ISSUE_NO_MATERIAL, sorted(unassigned),
            "written as \"%s\", the engine's checkerboard fallback"
            % BRUSH_DEFAULT_MATERIAL)

    if any(i.is_fatal for i in issues):
        return None, issues
    return sides, issues


# ─────────────────────────────────────────────────────────────────────
#  OBJECT ANALYSIS
# ─────────────────────────────────────────────────────────────────────

def _object_export_bmesh(obj, depsgraph, transform, grid):
    """A bmesh of *obj* in target space, with mesh element indices intact.

    Indices matter beyond the writer: they are what the sidebar's
    Highlight Issue button selects in Edit Mode afterwards, so the bmesh
    is built straight from the object's own mesh (or its evaluated one)
    and never re-indexed.

    *transform* is the full 4x4 taking a local coordinate to game space,
    scale included. Face normals are recomputed from the transformed
    coordinates rather than carried over, so a mirrored or negatively
    scaled object arrives with the winding it actually has — the caller's
    signed-volume test then sees the flip and reverses the planes.
    """
    mesh = None
    temp_owner = None
    if depsgraph is not None:
        eva = obj.evaluated_get(depsgraph)
        mesh = eva.to_mesh()
        temp_owner = eva
    else:
        mesh = obj.data

    bm = bmesh.new()
    bm.from_mesh(mesh)
    if temp_owner is not None:
        temp_owner.to_mesh_clear()

    for v in bm.verts:
        co = transform @ v.co
        if grid > 0.0:
            co = Vector((snap_to_grid(co.x, grid),
                         snap_to_grid(co.y, grid),
                         snap_to_grid(co.z, grid)))
        v.co = co
    bm.verts.index_update()
    bm.edges.index_update()
    bm.faces.index_update()
    bm.normal_update()
    return bm


def analyze_object(obj, transform, snap_normals=True, grid=0.0,
                   depsgraph=None, want_sides=True):
    """Split one mesh object into brushes and check every one.

    Returns (brushes, issues) where *brushes* is a list of side lists.
    """
    brushes = []
    issues = []
    bm = _object_export_bmesh(obj, depsgraph, transform, grid)
    try:
        uv_layer = bm.loops.layers.uv.active
        materials = [slot.material for slot in obj.material_slots]
        components = brush_components(bm)
        if uv_layer is None and components:
            issues.append(BrushIssue(
                ISSUE_NO_UV, obj.name, 0,
                [f.index for f in bm.faces],
                "every side written with the editor default matrix "
                "(%.6g)" % BRUSH_DEFAULT_TEXMAT['2']))
        for index, faces in enumerate(components):
            sides, comp_issues = validate_component(
                faces, obj.name, index, uv_layer, materials, snap_normals,
                want_sides=want_sides)
            issues.extend(comp_issues)
            if sides:
                brushes.append(sides)
    finally:
        bm.free()
    return brushes, issues


# ─────────────────────────────────────────────────────────────────────
#  SCOPE RESOLUTION
# ─────────────────────────────────────────────────────────────────────

def _collection_mesh_objects(collection, skip_hidden, view_layer=None):
    """Every mesh object in *collection* and its children, in a stable
    order and with each object counted once however many collections it
    is linked into."""
    out = []
    seen = set()

    def walk(col):
        for obj in col.objects:
            if obj.type != 'MESH' or obj.name in seen:
                continue
            if skip_hidden and view_layer is not None:
                if not obj.visible_get(view_layer=view_layer):
                    continue
            seen.add(obj.name)
            out.append(obj)
        for child in col.children:
            walk(child)

    walk(collection)
    out.sort(key=lambda o: o.name)
    return out


def _root_collection_for(obj, scene):
    """The top-level collection an object belongs to — the one a .map
    import created and named after the file.

    Walks down from the scene collection rather than up from the object,
    because a Collection has no parent pointer: Blender models the link
    in one direction only, and the same collection can legitimately hang
    under several parents.
    """
    if obj is None:
        return None
    target = set()
    for col in obj.users_collection:
        target.add(col.name)
    if not target:
        return None

    def contains(col, wanted, guard):
        if col.name in wanted:
            return True
        if col.name in guard:
            return False
        guard.add(col.name)
        return any(contains(c, wanted, guard) for c in col.children)

    for root in scene.collection.children:
        if contains(root, target, set()):
            return root
    return None


def resolve_worldspawn_collection(context):
    """Find the "worldspawn" collection to export, or explain why not.

    Returns (collection, message). Exactly one is ever set.

    Starting from the active object and falling back to the selection is
    what the .map import's own layout makes natural: the import builds
    one root collection named after the file, with a child collection per
    entity classname under it, so "the worldspawn of the map this object
    came from" is unambiguous from any brush in it.
    """
    scene = context.scene
    # view_layer rather than context.active_object / selected_objects:
    # both of those are missing from a File Browser's context, and the
    # export dialog's own Check Geometry button runs from exactly there.
    obj = context.view_layer.objects.active
    if obj is None:
        selected = [o for o in context.view_layer.objects if o.select_get()]
        obj = selected[0] if selected else None
    if obj is None:
        return None, ("Select an object from the map first — the "
                      "worldspawn collection is found from whichever map "
                      "the active object belongs to.")

    root = _root_collection_for(obj, scene)
    if root is None:
        return None, ("\"%s\" is not inside any top-level collection of "
                      "this scene, so there is no map root to search."
                      % obj.name)

    for child in root.children:
        if child.name.strip().lower() == 'worldspawn':
            return child, None

    if root.name.strip().lower() == 'worldspawn':
        return root, None

    found = [c.name for c in root.children]
    detail = (', '.join('"%s"' % n for n in found[:12]) if found
              else 'no child collections at all')
    if len(found) > 12:
        detail += ' and %d more' % (len(found) - 12)
    return None, ("Collection \"%s\" has no child collection named "
                  "\"worldspawn\" — it contains %s. Pick a different "
                  "Source in the export options, or rename the collection "
                  "holding the world geometry to \"worldspawn\"."
                  % (root.name, detail))


def resolve_export_objects(context, source, skip_hidden):
    """(objects, label, message) for the chosen export Source."""
    view_layer = context.view_layer
    if source == 'SELECTED':
        objs = [o for o in view_layer.objects
                if o.type == 'MESH' and o.select_get()]
        if skip_hidden:
            objs = [o for o in objs if o.visible_get(view_layer=view_layer)]
        objs.sort(key=lambda o: o.name)
        if not objs:
            return [], '', "No mesh objects are selected."
        return objs, "selection", None

    if source == 'ACTIVE_COLLECTION':
        layer_col = view_layer.active_layer_collection
        col = layer_col.collection if layer_col else None
        if col is None:
            return [], '', "There is no active collection."
        objs = _collection_mesh_objects(col, skip_hidden, view_layer)
        if not objs:
            return [], col.name, ("Collection \"%s\" contains no mesh "
                                  "objects." % col.name)
        return objs, col.name, None

    col, message = resolve_worldspawn_collection(context)
    if col is None:
        return [], '', message
    objs = _collection_mesh_objects(col, skip_hidden, view_layer)
    if not objs:
        return [], col.name, ("Collection \"%s\" contains no mesh objects."
                              % col.name)
    return objs, col.name, None


def worldspawn_spawnargs(collection):
    """The worldspawn entity's key/values, recovered from the import.

    store_entity_keys stamps every spawnarg onto the entity's empty as
    "map_<key>", so a re-export can put back the "call", "script" and
    similar keys the map came in with instead of writing a bare
    worldspawn that silently drops the map's script hookup. classname is
    forced rather than trusted — this entity is worldspawn by definition
    of where it is being written.
    """
    keys = {}
    if collection is None:
        return keys
    for obj in collection.objects:
        if obj.type != 'EMPTY':
            continue
        found = {}
        for prop in obj.keys():
            if not prop.startswith('map_'):
                continue
            name = prop[4:]
            if not name or name.lower() == 'classname':
                continue
            value = obj[prop]
            if isinstance(value, str):
                found[name] = value
        if found or (obj.get('map_classname', '') or '').lower() == 'worldspawn':
            keys.update(found)
    return keys


# ─────────────────────────────────────────────────────────────────────
#  WRITER
# ─────────────────────────────────────────────────────────────────────

def format_brush(sides, primitive_num, map_version):
    """One brushDef3 primitive, formatted exactly as idMapBrush::Write
    does — including the leading "// primitive N" comment, the one-space
    indent on "brushDef3" and the two-space indent on each side.

    The trailing "0 0 0" after the material is Doom 3's; it is where
    Quake 2 kept per-side content/flag overrides, which idMapBrush::Parse
    reads and discards ("Q2 allowed override of default flags and values,
    but we don't any more"). Quake 4's Version 3 maps omit it, so it is
    written only for Version 2.
    """
    tail = ' 0 0 0' if map_version == '2' else ''
    out = ["// primitive %d" % primitive_num, "{", " brushDef3", " {"]
    for s in sides:
        out.append(
            "  ( %s %s %s %s ) ( ( %s %s %s ) ( %s %s %s ) ) \"%s\"%s"
            % (_fs_float(s.normal.x), _fs_float(s.normal.y),
               _fs_float(s.normal.z), _fs_float(s.dist),
               _fs_float(s.row0[0]), _fs_float(s.row0[1]), _fs_float(s.row0[2]),
               _fs_float(s.row1[0]), _fs_float(s.row1[1]), _fs_float(s.row1[2]),
               s.material, tail))
    out.append(" }")
    out.append("}")
    return out


def format_map(brushes, spawnargs, map_version):
    """The whole file, as one list of lines.

    Laid out the way idMapFile::Write lays it out: the version line, then
    each entity as "// entity N" plus a brace block of epairs followed by
    its primitives. Only entity 0 (worldspawn) is written — this exporter
    covers world brush geometry, and an entity block for anything else
    would be a promise the rest of the scene is not being kept.
    """
    lines = ["Version %s" % map_version, "// entity 0", "{",
             "\"classname\" \"worldspawn\""]
    for key, value in spawnargs.items():
        lines.append("\"%s\" \"%s\"" % (key, value))
    for i, sides in enumerate(brushes):
        lines.extend(format_brush(sides, i, map_version))
    lines.append("}")
    return lines


class MapExportReport(object):
    """One export's outcome: the counts, then the issues behind them."""
    __slots__ = ('headline', 'lines', 'issues')

    def __init__(self, headline, lines=None, issues=None):
        self.headline = headline
        self.lines = list(lines or [])
        self.issues = list(issues or [])

    @property
    def result(self):
        return '\n'.join([self.headline] + self.lines)

    def body(self, title):
        out = [title, '=' * len(title), '', self.headline, '']
        out.extend(self.lines)
        out.append('')
        if not self.issues:
            out.append('  No issues.')
            return '\n'.join(out) + '\n'
        out.append('  %d issue(s)' % len(self.issues))
        out.append('  ' + '-' * 60)
        for kind, _info in BRUSH_ISSUE_INFO:
            group = [i for i in self.issues if i.kind == kind]
            if not group:
                continue
            label, severity, _mode, _icon = BRUSH_ISSUE_LOOKUP[kind]
            out.append('')
            out.append('  %s — %s (%d)' % (severity, label, len(group)))
            out.extend('    ' + i.describe() for i in group)
        return '\n'.join(out) + '\n'


def export_map(context, filepath, source='WORLDSPAWN', map_version='2',
               scale=1.0, rotation='NONE', snap='NORMALS', grid=1.0,
               skip_hidden=True, apply_modifiers=False,
               write_spawnargs=True):
    """Write the chosen worldspawn geometry to *filepath*.

    Returns a MapExportReport. Nothing is written when the scope resolves
    to nothing or every brush fails validation; a partial success writes
    the brushes that passed and reports the rest, so one stray patch
    surface in an imported map cannot block the export of the map.
    """
    objects, label, message = resolve_export_objects(context, source, skip_hidden)
    if message:
        return MapExportReport("ERROR: " + message)

    rot = ROTATION_PRESETS.get(rotation) or Matrix.Identity(4)
    scale_matrix = Matrix.Diagonal((scale, scale, scale, 1.0))
    grid_size = grid if snap in ('GRID', 'NORMALS_GRID') else 0.0
    snap_normals = snap in ('NORMALS', 'NORMALS_GRID')

    depsgraph = context.evaluated_depsgraph_get() if apply_modifiers else None

    brushes = []
    issues = []
    per_object = []
    for obj in objects:
        transform = scale_matrix @ rot @ obj.matrix_world
        obj_brushes, obj_issues = analyze_object(
            obj, transform, snap_normals=snap_normals, grid=grid_size,
            depsgraph=depsgraph)
        brushes.extend(obj_brushes)
        issues.extend(obj_issues)
        per_object.append((obj.name, len(obj_brushes),
                           sum(1 for i in obj_issues if i.is_fatal)))

    fatal = [i for i in issues if i.is_fatal]
    warnings = [i for i in issues if not i.is_fatal]
    rejected = len(set((i.object_name, i.component) for i in fatal))

    spawnargs = {}
    if write_spawnargs and source == 'WORLDSPAWN':
        col, _msg = resolve_worldspawn_collection(context)
        spawnargs = worldspawn_spawnargs(col)

    lines = [
        "  source        : %s (%s)" % (label, source.replace('_', ' ').lower()),
        "  objects       : %d" % len(objects),
        "  brushes       : %d written" % len(brushes),
        "  rejected      : %d" % rejected,
        "  warnings      : %d" % len(warnings),
        "  sides         : %d" % sum(len(s) for s in brushes),
        "  map version   : %s" % map_version,
        "  scale         : %.6g" % scale,
        "  rotation      : %s" % rotation,
        "  snapping      : %s%s" % (snap, ("  grid %.6g" % grid)
                                    if grid_size else ''),
    ]
    if len(per_object) > 1:
        lines.append("")
        lines.append("  per object:")
        for name, count, bad in per_object:
            lines.append("    %-40s %4d brush(es)%s"
                         % (name, count, ", %d rejected" % bad if bad else ''))

    if not brushes:
        return MapExportReport(
            "ERROR: no brushes could be written — every component failed "
            "validation.", lines, issues)

    text = '\n'.join(format_map(brushes, spawnargs, map_version)) + '\n'
    with open(filepath, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(text)

    if rejected:
        headline = ("WARNING: wrote %d brush(es); %d component(s) were "
                    "rejected." % (len(brushes), rejected))
    elif warnings:
        headline = ("WARNING: wrote %d brush(es) with %d warning(s)."
                    % (len(brushes), len(warnings)))
    else:
        headline = "OK: wrote %d brush(es)." % len(brushes)
    return MapExportReport(headline, lines, issues)



# ─────────────────────────────────────────────────────────────────────
#  MAP EXPORT — EDIT MODE
# ─────────────────────────────────────────────────────────────────────

def map_to_object_mode(context):
    """Leave Edit Mode so the export reads the mesh as it is NOW.

    An object in Edit Mode holds its live geometry in an edit BMesh that
    obj.data has not been updated from, so anything reading obj.data
    silently exports the model as it was when Edit Mode was entered — no
    error, no warning, just the old geometry in the file. The mode round
    trip is what flushes it, and it flushes every object of a
    multi-object edit session rather than only the active one. This is
    the recipe Blender's own io_scene_fbx uses, and the same one the
    .ase/.lwo exporter in this toolchain carries.

    Where FBX guards the switch with poll() and exports the stale data
    anyway when it fails, this raises instead: a refused export is
    recoverable, a silently pre-edit one is the failure this exists to
    remove.
    """
    obj = context.view_layer.objects.active
    if obj is None or obj.mode == 'OBJECT':
        return None
    if not bpy.ops.object.mode_set.poll():
        pretty = obj.mode.replace('_', ' ').title()
        raise RuntimeError(
            "Cannot leave %s Mode automatically. Switch to Object Mode and "
            "export again — exporting from %s Mode would write the geometry "
            "as it was when you entered it." % (pretty, pretty))
    mode = obj.mode
    bpy.ops.object.mode_set(mode='OBJECT')
    return obj, mode


def map_restore_mode(context, saved):
    """Put back what map_to_object_mode left. Safe to call with None."""
    if not saved:
        return
    obj, mode = saved
    context.view_layer.objects.active = obj
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode=mode)


# ─────────────────────────────────────────────────────────────────────
#  VALIDATION RESULTS  (sidebar list state)
# ─────────────────────────────────────────────────────────────────────

class IDTECH4_PG_MapIssue(bpy.types.PropertyGroup):
    """One row of the Validate Collection list.

    The mesh element indices are kept as a comma-separated string rather
    than a nested CollectionProperty: a per-index PropertyGroup would
    cost one RNA struct per face on a scene with thousands of them, and
    nothing ever reads them except Highlight Issue, which parses the
    whole string in one go.

    verts/edges/faces record the size of the mesh at the moment it was
    checked. Editing the mesh invalidates every index in the list, so
    Highlight Issue refuses rather than selecting whatever now happens to
    live at those numbers.
    """
    kind: StringProperty(default='')
    label: StringProperty(default='')
    severity: StringProperty(default='ERROR')
    select_mode: StringProperty(default='FACE')
    icon: StringProperty(default='ERROR')
    object_name: StringProperty(default='')
    component: IntProperty(default=0)
    detail: StringProperty(default='')
    indices: StringProperty(default='')
    index_count: IntProperty(default=0)
    verts: IntProperty(default=0)
    edges: IntProperty(default=0)
    faces: IntProperty(default=0)


class IDTECH4_PG_MapValidate(bpy.types.PropertyGroup):
    issues: bpy.props.CollectionProperty(type=IDTECH4_PG_MapIssue)
    active_index: IntProperty(default=0)
    summary: StringProperty(default='')
    scope: StringProperty(default='')
    checked: BoolProperty(default=False)
    brush_count: IntProperty(default=0)
    object_count: IntProperty(default=0)


class IDTECH4_UL_map_issues(bpy.types.UIList):
    """Issues grouped by type.

    A UIList is a flat list with no header rows, so grouping is done by
    sorting on the issue type (BRUSH_ISSUE_ORDER puts the failures that
    stop a brush being written above the warnings that do not) and giving
    every row its type's own name and icon. Reading down the list, each
    type's rows sit together and are labelled, which is what grouping was
    for; the alternative - fake separator rows - would be selectable
    entries that Highlight Issue could do nothing with.
    """

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_prop, index):
        if self.layout_type in ('DEFAULT', 'COMPACT'):
            row = layout.row(align=True)
            split = row.split(factor=0.52, align=True)
            left = split.row(align=True)
            left.alert = (item.severity == 'ERROR')
            left.label(text=item.label, icon=item.icon or 'DOT')
            right = split.row(align=True)
            right.label(text="%s  [brush %d]  ×%d"
                             % (item.object_name, item.component,
                                item.index_count))
        else:
            layout.label(text=item.label, icon=item.icon or 'DOT')

    def filter_items(self, context, data, propname):
        items = getattr(data, propname)
        # A UIList's "neworder" is not the sorted sequence of indices: it
        # is, for each item in its ORIGINAL position, the row it should
        # move to. Returning the sort order directly reverses the mapping
        # on any list that is not its own inverse permutation, which
        # scrambles the grouping this exists to produce.
        keyed = sorted(range(len(items)),
                       key=lambda i: (BRUSH_ISSUE_ORDER.get(items[i].kind, 99),
                                      items[i].object_name,
                                      items[i].component))
        neworder = [0] * len(items)
        for position, original in enumerate(keyed):
            neworder[original] = position
        return [], neworder


def _store_issues(state, issues, obj_lookup):
    """Replace the sidebar list with *issues*."""
    state.issues.clear()
    for issue in issues:
        row = state.issues.add()
        row.kind = issue.kind
        row.label = issue.label
        row.severity = issue.severity
        row.select_mode = issue.select_mode
        row.icon = issue.icon
        row.object_name = issue.object_name
        row.component = issue.component
        row.detail = issue.detail
        # Capped: an out-of-bounds report on a dense mesh could otherwise
        # put a hundred thousand numbers into one StringProperty, and
        # selecting the first few thousand shows the problem just as well.
        capped = issue.indices[:4096]
        row.indices = ','.join(str(i) for i in capped)
        row.index_count = len(issue.indices)
        mesh = obj_lookup.get(issue.object_name)
        if mesh is not None:
            row.verts = len(mesh.vertices)
            row.edges = len(mesh.edges)
            row.faces = len(mesh.polygons)
    state.active_index = 0


def validate_objects(objects, snap_normals=False, grid=0.0):
    """Run the export's own checks over *objects*, geometry only.

    Deliberately runs at the objects' own world transform with no export
    scale, no rotation preset and (by default) no grid snapping: every
    check here is invariant under a uniform scale or a Z rotation, so
    feeding the export's transform options in would change nothing except
    to make the sidebar's verdict depend on settings that live in a
    dialog the sidebar cannot see. Grid snapping is the exception, since
    moving vertices genuinely can break flatness — the export dialog's
    own Check Geometry button passes it through.

    Modifiers are never applied, because the indices this returns have to
    address the object's own mesh for Highlight Issue to select them.
    """
    issues = []
    brushes = 0
    for obj in objects:
        obj_brushes, obj_issues = analyze_object(
            obj, obj.matrix_world, snap_normals=snap_normals, grid=grid,
            depsgraph=None)
        brushes += len(obj_brushes)
        issues.extend(obj_issues)
    issues.sort(key=lambda i: (BRUSH_ISSUE_ORDER.get(i.kind, 99),
                               i.object_name, i.component))
    return brushes, issues


def _validation_summary(objects, brushes, issues):
    fatal = [i for i in issues if i.is_fatal]
    rejected = len(set((i.object_name, i.component) for i in fatal))
    if not issues:
        return "%d object(s), %d brush(es) — all valid" % (len(objects), brushes)
    return ("%d object(s), %d brush(es) ok, %d rejected, %d warning(s)"
            % (len(objects), brushes, rejected, len(issues) - len(fatal)))


class IDTECH4_OT_map_validate(bpy.types.Operator):
    """Check every mesh in the active collection for anything that would
    stop it exporting as an idTech4 brush"""
    bl_idname = "idtech4.map_validate"
    bl_label = "Validate Collection"
    bl_options = {'REGISTER'}

    source: EnumProperty(items=EXPORT_MAP_SOURCE_ITEMS,
                         default='ACTIVE_COLLECTION', options={'SKIP_SAVE'})
    skip_hidden: BoolProperty(default=True, options={'SKIP_SAVE'})
    snap: StringProperty(default='NORMALS', options={'SKIP_SAVE'})
    grid: FloatProperty(default=0.0, options={'SKIP_SAVE'})

    def execute(self, context):
        state = context.scene.idtech4_map_validate
        objects, label, message = resolve_export_objects(
            context, self.source, self.skip_hidden)
        if message:
            state.issues.clear()
            state.checked = True
            state.scope = label
            state.brush_count = 0
            state.object_count = 0
            state.summary = message
            self.report({'WARNING'}, message)
            return {'CANCELLED'}

        try:
            saved = map_to_object_mode(context)
        except RuntimeError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        try:
            grid = self.grid if self.snap in ('GRID', 'NORMALS_GRID') else 0.0
            brushes, issues = validate_objects(
                objects, snap_normals=self.snap in ('NORMALS', 'NORMALS_GRID'),
                grid=grid)
        finally:
            map_restore_mode(context, saved)

        _store_issues(state, issues, {o.name: o.data for o in objects})
        state.checked = True
        state.scope = label
        state.brush_count = brushes
        state.object_count = len(objects)
        state.summary = _validation_summary(objects, brushes, issues)
        self.report({'WARNING'} if issues else {'INFO'}, state.summary)
        return {'FINISHED'}


class IDTECH4_OT_map_highlight_issue(bpy.types.Operator):
    """Enter Edit Mode on the selected issue's object and select exactly
    the geometry responsible for it"""
    bl_idname = "idtech4.map_highlight_issue"
    bl_label = "Highlight Issue"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        state = getattr(context.scene, 'idtech4_map_validate', None)
        return bool(state and 0 <= state.active_index < len(state.issues))

    def execute(self, context):
        state = context.scene.idtech4_map_validate
        row = state.issues[state.active_index]

        obj = bpy.data.objects.get(row.object_name)
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'}, "Object \"%s\" no longer exists."
                        % row.object_name)
            return {'CANCELLED'}
        if obj.name not in context.view_layer.objects:
            self.report({'ERROR'}, "\"%s\" is not in the current view layer."
                        % obj.name)
            return {'CANCELLED'}

        mesh = obj.data
        # Indices address the mesh as it was when Validate ran. An edit
        # since then renumbers everything, so selecting those numbers now
        # would highlight unrelated geometry and quietly claim it was the
        # problem.
        if (len(mesh.vertices), len(mesh.edges), len(mesh.polygons)) != \
                (row.verts, row.edges, row.faces):
            self.report({'ERROR'},
                        "\"%s\" has been edited since it was checked — run "
                        "Validate Collection again." % obj.name)
            return {'CANCELLED'}

        if context.mode != 'OBJECT':
            if not bpy.ops.object.mode_set.poll():
                self.report({'ERROR'}, "Switch to Object Mode first.")
                return {'CANCELLED'}
            bpy.ops.object.mode_set(mode='OBJECT')

        # visible_get() is False for anything hidden, excluded or on a
        # disabled collection, and mode_set refuses on all three — so say
        # which it is rather than letting the operator fail opaquely.
        obj.hide_set(False)
        if not obj.visible_get():
            self.report({'ERROR'},
                        "\"%s\" is not visible in this view layer (its "
                        "collection may be excluded or disabled)." % obj.name)
            return {'CANCELLED'}

        for other in context.view_layer.objects:
            if other.select_get():
                other.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj

        bpy.ops.object.mode_set(mode='EDIT')
        mode = row.select_mode
        context.tool_settings.mesh_select_mode = (mode == 'VERT',
                                                  mode == 'EDGE',
                                                  mode == 'FACE')
        bm = bmesh.from_edit_mesh(mesh)
        for elem in (bm.verts, bm.edges, bm.faces):
            elem.ensure_lookup_table()
        for elem in bm.verts:
            elem.select_set(False)
        for elem in bm.edges:
            elem.select_set(False)
        for elem in bm.faces:
            elem.select_set(False)

        pool = {'VERT': bm.verts, 'EDGE': bm.edges, 'FACE': bm.faces}[mode]
        wanted = [int(i) for i in row.indices.split(',') if i.strip()]
        hit = 0
        for i in wanted:
            if 0 <= i < len(pool):
                pool[i].select_set(True)
                hit += 1
        bm.select_flush(True)
        bmesh.update_edit_mesh(mesh)

        shown = ("%d of %d" % (hit, row.index_count)
                 if hit != row.index_count else str(hit))
        self.report({'INFO'}, "%s — selected %s %s(s) on \"%s\""
                    % (row.label, shown, mode.lower(), obj.name))
        return {'FINISHED'}


class IDTECH4_OT_map_clear_issues(bpy.types.Operator):
    """Empty the issue list"""
    bl_idname = "idtech4.map_clear_issues"
    bl_label = "Clear"
    bl_options = {'REGISTER'}

    def execute(self, context):
        state = context.scene.idtech4_map_validate
        state.issues.clear()
        state.checked = False
        state.summary = ''
        state.scope = ''
        return {'FINISHED'}


class IDTECH4_PT_map_validate(bpy.types.Panel):
    """Validate Collection — the export's own checks, run on demand."""
    bl_label = "Validate Collection"
    bl_idname = "IDTECH4_PT_map_validate"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "idTech4 Map"

    def draw(self, context):
        layout = self.layout
        state = context.scene.idtech4_map_validate

        layer_col = context.view_layer.active_layer_collection
        col_name = layer_col.collection.name if layer_col else '<none>'

        info = layout.box()
        info.label(text="Checks every mesh for the four things", icon='INFO')
        info.label(text="a brushDef3 has to be: closed, flat-")
        info.label(text="faced, convex, and one material and")
        info.label(text="one flat UV projection per plane.")

        col = layout.column(align=True)
        col.label(text="Active collection:  %s" % col_name)
        op = col.operator(IDTECH4_OT_map_validate.bl_idname,
                          text="Validate Collection", icon='CHECKMARK')
        op.source = 'ACTIVE_COLLECTION'
        op.skip_hidden = True
        op.snap = 'NORMALS'
        op.grid = 0.0

        if not state.checked:
            return

        box = layout.box()
        if not state.issues:
            row = box.row()
            row.label(text=state.summary or "No issues.", icon='CHECKMARK')
            box.operator(IDTECH4_OT_map_clear_issues.bl_idname, icon='X')
            return

        head = box.row()
        head.alert = any(i.severity == 'ERROR' for i in state.issues)
        head.label(text=state.summary, icon='ERROR')

        box.template_list("IDTECH4_UL_map_issues", "",
                          state, "issues", state, "active_index", rows=8)

        if 0 <= state.active_index < len(state.issues):
            row = state.issues[state.active_index]
            detail = box.column(align=True)
            detail.label(text="%s  [brush %d]" % (row.object_name, row.component))
            if row.detail:
                # label() never wraps, so a long explanation is broken on
                # word boundaries rather than running off the sidebar.
                for line in _wrap_words(row.detail, 34):
                    detail.label(text="    " + line)
            detail.label(text="    %d %s(s) affected"
                              % (row.index_count, row.select_mode.lower()))

        row = box.row(align=True)
        row.operator(IDTECH4_OT_map_highlight_issue.bl_idname,
                     text="Highlight Issue", icon='RESTRICT_SELECT_OFF')
        row.operator(IDTECH4_OT_map_clear_issues.bl_idname, text="", icon='X')


def _wrap_words(text, width):
    """Break *text* on spaces to fit *width* columns. layout.label() does
    no wrapping of its own, and Blender exposes no measurement of the
    region's real width in characters, so this is the same approach the
    Sources panel takes to its explanatory text."""
    words = text.split()
    lines = []
    current = ''
    for word in words:
        candidate = word if not current else current + ' ' + word
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


# ─────────────────────────────────────────────────────────────────────
#  EXPORT OPERATOR
# ─────────────────────────────────────────────────────────────────────

class EXPORT_OT_idtech4_map(bpy.types.Operator, ExportHelper,
                            MD5_ExportScaleRotMixin):
    """Export worldspawn brush geometry to an idTech4 .map file"""
    bl_idname = "export_scene.idtech4_map"
    bl_label = "Export idTech4 .map"
    bl_options = {'REGISTER', 'PRESET'}

    filename_ext = ".map"
    filter_glob: StringProperty(default="*.map", options={'HIDDEN'}, maxlen=255)

    source: EnumProperty(
        name="Source",
        description="Which meshes become worldspawn brushes",
        items=EXPORT_MAP_SOURCE_ITEMS,
        default='WORLDSPAWN',
    )
    map_version: EnumProperty(
        name="",
        description="Which .map dialect to write",
        items=EXPORT_MAP_VERSION_ITEMS,
        default='2',
    )
    snap: EnumProperty(
        name="",
        description="How exported planes are tidied up",
        items=EXPORT_MAP_SNAP_ITEMS,
        default='NORMALS',
    )
    grid_size: FloatProperty(
        name="Grid Size",
        description="Grid the vertex positions are rounded to, in idTech4 "
                    "game units, before any plane is derived",
        default=1.0, min=0.001, max=1024.0,
    )
    skip_hidden: BoolProperty(
        name="Skip Hidden Objects",
        description="Leave out anything whose outliner EYE is closed",
        default=True,
    )
    apply_modifiers: BoolProperty(
        name="Apply Modifiers",
        description="Export each object's modifier-evaluated mesh instead "
                    "of its raw edit-mode geometry. Off by default so the "
                    "mesh in the outliner is the mesh that gets written",
        default=False,
    )
    write_spawnargs: BoolProperty(
        name="Keep Worldspawn Spawnargs",
        description="Write back the key/values the worldspawn entity came "
                    "in with (\"call\", \"script\", and anything else the "
                    "import stamped onto its empty as map_*). Turn off to "
                    "write a bare worldspawn",
        default=True,
    )
    show_report: BoolProperty(
        name="Open Report",
        description="Open the full export report in a Text Editor window "
                    "afterwards. The report is written to a text datablock "
                    "either way and can be reopened from any Text Editor",
        default=True,
    )

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = False

        layout.label(text="Source:")
        box = layout.box()
        box.prop(self, "source", text="")
        box.prop(self, "skip_hidden")
        box.prop(self, "apply_modifiers")
        box.prop(self, "write_spawnargs")

        self.draw_transforms(layout)

        layout.label(text="Map Format:")
        box = layout.box()
        box.label(text="Version:")
        box.prop(self, "map_version")
        box.label(text="Snapping:")
        box.prop(self, "snap")
        if self.snap in ('GRID', 'NORMALS_GRID'):
            box.prop(self, "grid_size")

        layout.label(text="Geometry:")
        box = layout.box()
        op = box.operator(IDTECH4_OT_map_validate.bl_idname,
                          text="Check Geometry", icon='CHECKMARK')
        op.source = self.source
        op.skip_hidden = self.skip_hidden
        op.snap = self.snap
        op.grid = self.grid_size

        state = context.scene.idtech4_map_validate
        if state.checked:
            sub = box.column(align=True)
            sub.alert = any(i.severity == 'ERROR' for i in state.issues)
            for line in _wrap_words(state.summary, 44):
                sub.label(text=line)
            counts = {}
            for row in state.issues:
                counts[row.label] = counts.get(row.label, 0) + 1
            for label, count in sorted(counts.items()):
                box.label(text="    %s × %d" % (label, count))
            box.label(text="Full detail in the idTech4 Map sidebar tab.")
        else:
            box.label(text="Not checked yet.", icon='INFO')

        layout.prop(self, "show_report")

    def execute(self, context):
        started = time.time()
        try:
            saved = map_to_object_mode(context)
        except RuntimeError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        try:
            report = export_map(
                context, self.filepath,
                source=self.source,
                map_version=self.map_version,
                scale=self.get_scale(),
                rotation=self.get_rotation(),
                snap=self.snap,
                grid=self.grid_size,
                skip_hidden=self.skip_hidden,
                apply_modifiers=self.apply_modifiers,
                write_spawnargs=self.write_spawnargs,
            )
        except Exception as exc:
            traceback.print_exc()
            self.report({'ERROR'}, "Export failed: %s" % exc)
            return {'CANCELLED'}
        finally:
            map_restore_mode(context, saved)

        # The sidebar list is filled from the export's own findings, so
        # Highlight Issue works on whatever the export just rejected
        # without having to run the check a second time.
        state = context.scene.idtech4_map_validate
        objects, label, _msg = resolve_export_objects(
            context, self.source, self.skip_hidden)
        _store_issues(state, report.issues, {o.name: o.data for o in objects})
        state.checked = True
        state.scope = label
        state.object_count = len(objects)
        state.summary = report.headline

        report.lines.append("  time          : %.2fs" % (time.time() - started))
        title = ("idTech4 Map Export Report for %s"
                 % os.path.basename(self.filepath))
        if self.show_report:
            show_map_export_report(context, title, report)
        else:
            text_block = bpy.data.texts.get(title) or bpy.data.texts.new(title)
            text_block.clear()
            text_block.write(report.body(title))

        level = report.headline.split(':', 1)[0]
        self.report({'ERROR'} if level == 'ERROR'
                    else {'WARNING'} if level == 'WARNING'
                    else {'INFO'}, report.result)
        return {'CANCELLED'} if level == 'ERROR' else {'FINISHED'}


def show_map_export_report(context, title, report):
    """Show the export report in its own Text Editor window — the same
    treatment, and for the same reasons, as show_map_import_report gives
    the import's: a popup truncates long issue lines, cannot scroll a
    long list and leaves nothing behind afterwards, while a text
    datablock can be reopened from any Text Editor later."""
    text_block = bpy.data.texts.get(title)
    if text_block is None:
        text_block = bpy.data.texts.new(title)
    text_block.clear()
    text_block.write(report.body(title))
    try:
        bpy.ops.wm.window_new()
        new_window = context.window_manager.windows[-1]
        area = new_window.screen.areas[0]
        area.type = 'TEXT_EDITOR'
        space = area.spaces.active
        space.text = text_block
        space.show_word_wrap = True
        space.top = 0
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────
#  MENU + REGISTER
# ─────────────────────────────────────────────────────────────────────

def menu_func_import(self, context):
    self.layout.operator(IMPORT_OT_idtech4_map.bl_idname,
                         text="idTech4 .map", icon='MESH_GRID')


def menu_func_export(self, context):
    self.layout.operator(EXPORT_OT_idtech4_map.bl_idname,
                         text="idTech4 .map", icon='MESH_GRID')


classes = (
    IMPORT_OT_idtech4_map,
    IMPORT_OT_idtech4_map_sources_gate,
    # Order matters here and only here: IDTECH4_PG_MapValidate holds a
    # CollectionProperty of IDTECH4_PG_MapIssue, and a PropertyGroup
    # cannot reference a type that is not registered yet.
    IDTECH4_PG_MapIssue,
    IDTECH4_PG_MapValidate,
    IDTECH4_UL_map_issues,
    IDTECH4_OT_map_validate,
    IDTECH4_OT_map_highlight_issue,
    IDTECH4_OT_map_clear_issues,
    IDTECH4_PT_map_validate,
    EXPORT_OT_idtech4_map,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    # Scene rather than WindowManager: the issue list is about the
    # geometry in this .blend, and surviving a save is what lets someone
    # validate a large map, close Blender and pick the list up again.
    bpy.types.Scene.idtech4_map_validate = (
        bpy.props.PointerProperty(type=IDTECH4_PG_MapValidate))
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)
    _register_shared_ui()


def unregister():
    _unregister_shared_ui()
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    # Before the classes: the pointer's type is one of them.
    if hasattr(bpy.types.Scene, 'idtech4_map_validate'):
        del bpy.types.Scene.idtech4_map_validate
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
