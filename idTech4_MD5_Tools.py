"""
Blender 4.x / 5.x – idTech4 MD5 Tools
================================
Full import, export, and retargeting support for Doom 3 / id Tech 4 MD5 formats.
Supports .md5mesh and .md5anim Version 10 and Version 12, and .md5camera.

Spec references:
    http://tfc.duke.free.fr/coding/md5-specs-en.html  (V10)
    MD5v12_FORMAT_SPEC.md                              (V12 extensions)

Install
-------
1.  Edit > Preferences > Add-ons > Install…
2.  Select this file.
3.  Enable "idTech MD5 tools - .md5mesh,md5anim, and md5camera Import/Export".

Scale (all import/export types)
--------------------------------
Every importer and exporter (mesh, anim, and camera) shares the same Scale
option. It is a checkbox, OFF by default, so nothing is scaled unless you
turn it on. When enabled, choose either:
    - a custom numeric Scale Factor, or
    - the idTech4 <-> Blender unit preset (idTech4 units are inches,
      Blender's default unit is metres): on import this converts
      idTech4 -> Blender (inches to metres, x0.0254); on export this
      converts Blender -> idTech4 (metres to inches, x39.3701).
The preset is the default scale mode whenever scaling is enabled.

Menu entries after installation
--------------------------------
File > Import > MD5 Mesh (.md5mesh)
    Import a skeletal mesh and armature. Supports V10 and V12.
    Options: scale, rotation, V12 normal handling, merge verts by distance.

File > Import > MD5 Anim (.md5anim)
    Import one or more animation files onto an existing armature.
    Each file becomes its own Blender Action.
    Options: scale, rotation, action name prepend, overwrite confirmation.

File > Import > MD5 Camera (.md5camera)
    Import one or more camera animation files.
    Each file creates its own camera object named MD5_Cam_<stem> with
    keyframed location, rotation, and FOV. Cuts become timeline markers
    named after the action.
    Options: scale, reorient, XYZ offset, clear timeline markers.

File > Export > MD5 Mesh (.md5mesh)
    Export the selected armature and its meshes. Only bones in the
    MD5_export_bone_collection bone collection are written.
    Options: scale, rotation, bone influence limit, V12 mode, split sharp edges.

File > Export > MD5 Anim (.md5anim)
    Export one or more actions as individual .md5anim files.
    Options: scale, rotation, animation compression, delta threshold,
    action checklist, filename prepend/strip.

File > Export > MD5 Mesh/Anims (.md5mesh)
    Combined single-dialog export of mesh and animations.

File > Export > MD5 Camera (.md5camera)
    Export one or more camera actions as individual .md5camera files.
    Options: scale, frame rate, reorient, XYZ offset, action checklist,
    export all cuts or per-action cuts only.

Properties > Data (armature) > Manage MD5 Export Bone Collection
    Add, remove, replace, or clear bones in the MD5_export_bone_collection.
    Only bones in this collection are written on mesh and anim export.

3D Viewport > N-Panel > MD5 > MD5 Animation Retargeter
    Snapshot rest pose, realign bones after orientation edits, retarget all
    affected actions, backup and restore.

3D Viewport > N-Panel > idTech4 > Sources
    Base Directory / Materials Source, shared with every other idTech4
    addon (map import, .ase/.lwo import, materials) regardless of which
    subset of them is installed.
"""


bl_info = {
    "name":        "idTech4 MD5 tools - .md5mesh,md5anim, and md5camera Import/Export",
    "author":      "Samson using Claude Sonnet 4.6",
    "version":     (1, 0, 0),
    "blender":     (4, 0, 0),
    "location":    "File > Import > MD5 Mesh / MD5 Anim; 3D Viewport > N-Panel > MD5 / idTech4",
    "description": "Import/Export idTech4 MD5 mesh/animations/cameras ( V10 & 12 ) . Retarget MD5 animations to modified MD5 armatures. Compatible with Blender 4.x and 5.x.",
    "category":    "Import-Export",
}

import addon_utils
import bpy
import functools
import json
import math
import re
import os
import struct
import sys
from mathutils import Vector, Quaternion, Matrix
from bpy_extras.io_utils import ImportHelper, ExportHelper
from bpy.props import StringProperty, BoolProperty, FloatProperty, EnumProperty, IntProperty
from bpy.types import Operator


# ---------------------------------------------------------------------------
# Blender 4.x / 5.x Action & F-curve API compatibility
# ---------------------------------------------------------------------------
# Blender 4.4 introduced "slotted" Actions (Action -> layers -> strips ->
# channelbags -> fcurves, addressed per Slot). For backward compatibility,
# 4.4 kept the old direct-access properties (action.fcurves, action.groups,
# action.id_root) working as a deprecated alias for the first slot's data.
#
# Blender 5.0 REMOVED that legacy alias entirely. Code that does
# `action.fcurves` on 5.0+ now raises:
#     AttributeError: 'Action' object has no attribute 'fcurves'
#
# The helpers below branch on Blender's version so the rest of this addon
# can create/read F-curves the same way on both 4.x and 5.x.
#
# Docs:
#   https://developer.blender.org/docs/release_notes/4.4/python_api/
#   https://developer.blender.org/docs/release_notes/5.0/python_api/

BLENDER_HAS_SLOTTED_ACTIONS = bpy.app.version >= (4, 4, 0)   # slots/channelbags exist
BLENDER_HAS_LEGACY_FCURVES  = bpy.app.version < (5, 0, 0)    # action.fcurves still works


def _md5_get_or_create_slot(id_data, action):
    """Return an Action Slot on `action` suitable for animating `id_data`
    (an Object, Armature, Camera-data, etc.), creating one if needed, and
    make sure it is assigned as the active slot on id_data.animation_data.

    On Blender < 4.4 there is no slot concept, so this is a no-op that
    returns None.
    """
    if not BLENDER_HAS_SLOTTED_ACTIONS:
        return None

    slot = None
    for s in action.slots:
        if s.target_id_type == id_data.id_type:
            slot = s
            break
    if slot is None:
        slot = action.slots.new(id_type=id_data.id_type, name=id_data.name)

    if id_data.animation_data is not None:
        id_data.animation_data.action_slot = slot
    return slot


def _md5_new_fcurve(action, slot, data_path, index):
    """Create (and return) a new F-curve on `action` for `slot`.

    Works on Blender 4.x (legacy action.fcurves API, still present through
    4.x) as well as Blender 5.0+ (slotted-only API via ActionChannelbag).
    """
    if BLENDER_HAS_LEGACY_FCURVES:
        return action.fcurves.new(data_path, index=index)

    # Blender 5.0+: action.fcurves no longer exists — go through the slot's
    # channelbag instead. anim_utils.action_ensure_channelbag_for_slot()
    # creates the layer/strip/channelbag as needed.
    from bpy_extras import anim_utils
    channelbag = anim_utils.action_ensure_channelbag_for_slot(action, slot)
    return channelbag.fcurves.new(data_path, index=index)


def _md5_iter_action_fcurves(action):
    """Yield every F-curve stored on `action`, across all of its slots.

    Works on Blender 4.x (legacy action.fcurves API) as well as
    Blender 5.0+ (slotted-only API via ActionChannelbag). Use this any time
    you need to scan/read F-curves on an arbitrary action (e.g. actions
    pulled from bpy.data.actions) rather than write new ones.
    """
    if action is None:
        return
    if BLENDER_HAS_LEGACY_FCURVES:
        yield from action.fcurves
        return

    from bpy_extras import anim_utils
    for slot in action.slots:
        channelbag = anim_utils.action_get_channelbag_for_slot(action, slot)
        if channelbag is not None:
            yield from channelbag.fcurves


# ---------------------------------------------------------------------------
# Quaternion helpers  (spec: http://tfc.duke.free.fr/coding/md5-specs-en.html)
# ---------------------------------------------------------------------------

def quat_compute_w(x, y, z):
    """Compute w so that (w,x,y,z) is a unit quaternion."""
    t = 1.0 - x*x - y*y - z*z
    return 0.0 if t < 0.0 else -math.sqrt(t)


def quat_rotate_point(quat, point):
    """Rotate a 3-D point by a quaternion  R = Q · P · Q*."""
    return quat @ Vector(point)


# ---------------------------------------------------------------------------
# Shared tokeniser
# ---------------------------------------------------------------------------

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


def strip_comments(text):
    """Strip // line comments, but never inside a double-quoted string.
    Real content routinely has a literal "//" as part of a file path
    inside the informational "commandline" line (e.g. a doubled-
    separator quirk like "maps/fred/marscity//reception_redo.mb", seen
    throughout real Doom3 export history) — a plain
    re.sub(r'//[^\n]*', '', text) treats THAT as a comment start too,
    truncating the quoted string right there and leaving no closing
    quote anywhere in the rest of the file. consume_string() then just
    keeps consuming every following token hunting for one that ends in
    a quote, silently corrupting every value parsed after it (joint
    names, positions, everything) until it either stumbles onto some
    unrelated token that happens to itself start+end with a quote, or
    — on a small file — runs off the end of the token list entirely
    ("list index out of range").

    /* */ blocks are stripped here too. idLexer::ReadWhiteSpace handles
    both forms for every file it reads, .md5mesh included, and this only
    ever did the // half — a block comment anywhere in a .md5mesh had its
    contents fed to the tokeniser as data. An UNTERMINATED /* runs to the
    end of the file, which is not leniency but the engine's own behaviour:
    ReadWhiteSpace returns 0 at EOF while hunting for the closing */, and
    everything after the opener is gone.

    NOT handled, and deliberately: the md5 lexer flags
    (renderer/Model_md5.cpp:497) are ALLOWPATHNAMES | NOSTRINGESCAPECHARS,
    with NOSTRINGCONCAT left CLEAR — so two strings separated by whitespace
    are one token to the engine and two to this whitespace split. No
    .md5mesh or .md5anim in the five corpora has adjacent strings, and the
    only strings the format uses are the shader line, the joint names and
    the informational commandline, none of which can be adjacent to
    another. Noted rather than fixed."""
    out = []
    in_string = False
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            in_string = not in_string
            out.append(c)
            i += 1
            continue
        if not in_string and c == '/' and i + 1 < n:
            if text[i + 1] == '/':
                j = text.find('\n', i)
                if j == -1:
                    break
                i = j
                continue
            if text[i + 1] == '*':
                j = text.find('*/', i + 2)
                if j == -1:
                    break          # unterminated: the engine loses the rest
                # A space, not nothing: the engine's comment is whitespace,
                # so `mesh/*c*/name` is two tokens there and must not be
                # glued into one by this split-on-whitespace tokeniser.
                out.append(' ')
                i = j + 2
                continue
        out.append(c)
        i += 1
    return ''.join(out)


def make_tokeniser(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
        raw = fh.read()
    tokens = strip_comments(raw).split()
    state  = [0]

    def consume(expected=None):
        tok = tokens[state[0]]
        state[0] += 1
        if expected is not None and tok != expected:
            raise ValueError(
                f"Expected '{expected}', got '{tok}' (token #{state[0]})")
        return tok

    def consume_int():   return int(consume())
    def consume_float(): return float(consume())

    def consume_string():
        tok = consume()
        if tok.startswith('"'):
            result = tok[1:]
            while not result.endswith('"'):
                result += ' ' + consume()
            return result[:-1]
        return tok

    def consume_vec3():
        consume('(')
        x, y, z = consume_float(), consume_float(), consume_float()
        consume(')')
        return (x, y, z)

    def at_end(): return state[0] >= len(tokens)
    def peek():   return tokens[state[0]] if state[0] < len(tokens) else None

    return consume, consume_int, consume_float, consume_string, consume_vec3, at_end, peek


# ---------------------------------------------------------------------------
# MD5Mesh parser
# ---------------------------------------------------------------------------

def parse_md5mesh(filepath):
    consume, consume_int, consume_float, consume_string, consume_vec3, at_end, peek = \
        make_tokeniser(filepath)

    result = {'joints': [], 'meshes': [], 'version': 10}

    while not at_end():
        tok = consume()

        if tok == 'MD5Version':
            result['version'] = consume_int()
        elif tok == 'commandline':
            consume_string()
        elif tok in ('numJoints', 'numMeshes'):
            consume_int()

        elif tok == 'joints':
            consume('{')
            while peek() != '}':
                name       = consume_string()
                parent     = consume_int()
                pos        = consume_vec3()
                ox, oy, oz = consume_vec3()
                ow         = quat_compute_w(ox, oy, oz)
                result['joints'].append({
                    'name':   name,
                    'parent': parent,
                    'pos':    pos,
                    'orient': Quaternion((ow, ox, oy, oz)),
                })
            consume('}')

        elif tok == 'mesh':
            consume('{')
            mesh = {'shader': '', 'verts': [], 'tris': [],
                    'weights': [], 'colors': None}
            is_v12 = result['version'] >= 12
            while peek() != '}':
                sub = consume()
                if sub == 'shader':
                    mesh['shader'] = consume_string()
                elif sub == 'numverts':
                    consume_int()
                elif sub == 'vert':
                    consume_int()          # index (discard)
                    consume('(')
                    s, t = consume_float(), consume_float()
                    consume(')')
                    start, count = consume_int(), consume_int()
                    normal  = None
                    tangent = None
                    if is_v12:
                        # ( nx ny nz ) ( tx ty tz tw )
                        nx, ny, nz = consume_vec3()
                        normal = (nx, ny, nz)
                        consume('(')
                        tx = consume_float()
                        ty = consume_float()
                        tz = consume_float()
                        tw = consume_float()
                        consume(')')
                        tangent = (tx, ty, tz, tw)
                    mesh['verts'].append({
                        'uv':     (s, t),
                        'start':  start,
                        'count':  count,
                        'normal': normal,
                        'tangent':tangent,
                    })
                elif sub == 'numtris':
                    consume_int()
                elif sub == 'tri':
                    consume_int()
                    v0, v1, v2 = consume_int(), consume_int(), consume_int()
                    mesh['tris'].append((v0, v1, v2))
                elif sub == 'numweights':
                    consume_int()
                elif sub == 'weight':
                    consume_int()
                    joint = consume_int()
                    bias  = consume_float()
                    pos   = consume_vec3()
                    mesh['weights'].append(
                        {'joint': joint, 'bias': bias, 'pos': pos})
                elif sub == 'numvertexcolors':
                    num_vc = consume_int()
                    mesh['colors'] = []
                elif sub == 'vertexcolor':
                    consume_int()          # index (discard)
                    consume('(')
                    r = consume_float()
                    g = consume_float()
                    b = consume_float()
                    a = consume_float()
                    consume(')')
                    if mesh['colors'] is not None:
                        mesh['colors'].append((r, g, b, a))
            consume('}')
            result['meshes'].append(mesh)

    return result


# ---------------------------------------------------------------------------
# MD5Anim parser
# ---------------------------------------------------------------------------

def parse_md5anim(filepath):
    consume, consume_int, consume_float, consume_string, consume_vec3, at_end, peek = \
        make_tokeniser(filepath)

    result = {'hierarchy': [], 'baseframe': [], 'frames': []}

    while not at_end():
        tok = consume()

        if tok == 'MD5Version':
            result['version'] = consume_int()
        elif tok == 'commandline':
            consume_string()
        elif tok == 'numFrames':
            result['num_frames'] = consume_int()
        elif tok == 'numJoints':
            result['num_joints'] = consume_int()
        elif tok == 'frameRate':
            result['frame_rate'] = consume_int()
        elif tok == 'numAnimatedComponents':
            result['num_animated_components'] = consume_int()

        elif tok == 'hierarchy':
            consume('{')
            while peek() != '}':
                name        = consume_string()
                parent      = consume_int()
                flags       = consume_int()
                start_index = consume_int()
                result['hierarchy'].append({
                    'name': name, 'parent': parent,
                    'flags': flags, 'start_index': start_index,
                })
            consume('}')

        elif tok == 'bounds':
            consume('{')
            while peek() != '}':
                consume_vec3()
                consume_vec3()
            consume('}')

        elif tok == 'baseframe':
            consume('{')
            while peek() != '}':
                pos    = consume_vec3()
                orient = consume_vec3()
                result['baseframe'].append({'pos': pos, 'orient': orient})
            consume('}')

        elif tok == 'frame':
            consume_int()
            consume('{')
            frame_data = []
            while peek() != '}':
                frame_data.append(consume_float())
            consume('}')
            result['frames'].append(frame_data)

    return result


# ---------------------------------------------------------------------------
# Binary MD5 (.bmd5mesh / .bmd5anim) parser
# ---------------------------------------------------------------------------
# Doom 3 BFG can convert plaintext .md5mesh/.md5anim into binary
# .bmd5mesh/.bmd5anim caches (generated/rendermodels, generated/anim) for
# faster loading. The layout below was reverse engineered from the BFG
# (VR fork) engine source — idRenderModelStatic::WriteBinaryModel and
# idRenderModelMD5::WriteBinaryModel in renderer/Model.cpp/Model_md5.cpp,
# and idMD5Anim::WriteBinary in d3xp/anim/Anim.cpp.
#
# The files mix TWO different endian conventions, inherited from idFile:
#   - idFile::WriteString / WriteVec3 / WriteVec4 / WriteFloat / WriteInt
#     use LittleLong/LittleFloat/LittleRevBytes, which are no-ops on a
#     little-endian host — so these fields are plain native (LE) data.
#   - The idFile::WriteBig<T>() template (and WriteBigArray) always byte
#     swaps to true big-endian regardless of host, because it was shared
#     with Xbox 360/PS3 ports. Multi-byte scalars (int/float/int64/short)
#     written this way must be swapped back when read on a PC.
#   - 1-byte fields (bool, byte, char) are unaffected by either path.
# Which convention applies is per call-site, not per type — e.g. a joint's
# translation is written with WriteVec3 (native LE) while a mesh vertex's
# xyz is written as part of an idDrawVert via WriteBigArray (true BE),
# even though both are idVec3.
# Only the 3-character tag is checked; the low byte is a cache-format
# version number that has moved between engine builds (the game's actual
# shipped binaries were seen carrying versions the reverse-engineered VR
# fork source doesn't define — e.g. BRM version 110 instead of 108), so
# rather than reject files a newer/older compiler wrote, we trust that the
# field layout traced from source is stable across nearby versions and
# just skip the version byte.
BRM_TAG        = (ord('B') << 16) | (ord('R') << 8) | ord('M')
MD5B_TAG       = (ord('5') << 16) | (ord('D') << 8) | ord('M')
B_ANIM_MD5_TAG = (ord('B') << 16) | (ord('M') << 8) | ord('D')


class _BinReader:
    """Sequential cursor over a binary MD5 file's bytes, with separate
    helpers for the native-little-endian fields (WriteString/WriteVec3/
    WriteFloat/WriteInt call sites) and the true-big-endian fields
    (WriteBig<T>/WriteBigArray call sites) — see module note above."""
    __slots__ = ('data', 'pos')

    def __init__(self, data):
        self.data = data
        self.pos  = 0

    def _read(self, n):
        p = self.pos
        end = p + n
        if end > len(self.data):
            raise EOFError("Unexpected end of file while parsing binary MD5 data")
        self.pos = end
        return self.data[p:end]

    # -- native little-endian (WriteString/WriteVec3/WriteVec4/WriteFloat/WriteInt) --
    def i32_le(self):
        return struct.unpack_from('<i', self._read(4))[0]

    def f32_le(self):
        return struct.unpack_from('<f', self._read(4))[0]

    def vec3_le(self):
        return struct.unpack_from('<3f', self._read(12))

    def string(self):
        n = self.i32_le()
        if n <= 0:
            return ''
        return self._read(n).decode('utf-8', errors='replace')

    # -- true big-endian (WriteBig<T> / WriteBigArray) --
    def u32_be(self):
        return struct.unpack_from('>I', self._read(4))[0]

    def i32_be(self):
        return struct.unpack_from('>i', self._read(4))[0]

    def i64_be(self):
        return struct.unpack_from('>q', self._read(8))[0]

    def f32_be(self):
        return struct.unpack_from('>f', self._read(4))[0]

    def u16_be(self):
        return struct.unpack_from('>H', self._read(2))[0]

    def byte(self):
        return self._read(1)[0]

    def bytes_raw(self, n):
        return self._read(n)

    def i32_be_array(self, count):
        if count <= 0:
            return []
        return list(struct.unpack_from('>%di' % count, self._read(4 * count)))

    def u16_be_array(self, count):
        if count <= 0:
            return []
        return list(struct.unpack_from('>%dH' % count, self._read(2 * count)))


def _half_to_float(h):
    """Decode a GPU half-float bit pattern, matching F16toF32() in
    idlib/geometry/DrawVert.h exactly (including its handling of
    subnormals/Inf/NaN, which that function folds into one branch)."""
    e = (h >> 10) & 0x1F
    m = h & 0x3FF
    s = -1.0 if (h & 0x8000) else 1.0
    if 0 < e < 31:
        return s * (2.0 ** (e - 15)) * (1.0 + m / 1024.0)
    elif m == 0:
        return s * 0.0
    return s * (2.0 ** -14) * (m / 1024.0)


def _skip_generic_surface(r):
    """Skip one entry of idRenderModelStatic's serialized 'surfaces' list.
    Ordinary compiled MD5 models write zero of these; the only known
    producer is the VR fork's PDA hitscan-surface hack (see
    receptioncin1_smallpda.md5mesh in this folder), which adds one
    surface per PDA-shader mesh before the binary cache is written. We
    don't need this geometry for import (mesh geometry comes from the
    MD5-specific block that follows), just need to walk past it
    correctly if a converted PDA model is ever fed in."""
    r.i32_be()                 # id
    r.string()                 # shader name
    if not r.byte():           # isGeometry
        return
    r.vec3_le(); r.vec3_le()   # bounds[0], bounds[1]  (WriteVec3 -> native LE)
    r.i32_be()                 # ambientViewCount
    r.byte(); r.byte(); r.byte(); r.byte()   # generateNormals/tangentsCalculated/perfectHull/referencedIndexes (bool)
    num_verts   = r.i32_be()
    num_in_file = r.i32_be()
    if num_in_file > 0:
        for _ in range(num_verts):
            r.vec3_le()                # xyz -- native LE here (WriteVec3), unlike the MD5 mesh vertex path below
            r.u16_be(); r.u16_be()     # st (half floats)
            r.bytes_raw(4); r.bytes_raw(4); r.bytes_raw(4); r.bytes_raw(4)  # normal/tangent/color/color2
    num_shadow = r.i32_be()
    if num_shadow > 0:
        for _ in range(num_shadow):
            r.f32_le(); r.f32_le(); r.f32_le(); r.f32_le()   # WriteVec4 -> native LE
    num_indexes = r.i32_be()
    if num_indexes > 0:
        r.u16_be_array(num_indexes)
    num_sil_indexes = r.i32_be()
    if num_indexes > 0 and num_sil_indexes > 0:
        r.u16_be_array(num_indexes)
    num_mirrored = r.i32_be()
    if num_mirrored > 0:
        r.i32_be_array(num_mirrored)
    num_dup = r.i32_be()
    if num_dup > 0:
        r.i32_be_array(num_dup * 2)
    num_sil_edges = r.i32_be()
    if num_sil_edges > 0:
        for _ in range(num_sil_edges):
            r.u16_be(); r.u16_be(); r.u16_be(); r.u16_be()   # p1, p2, v1, v2
    if r.byte():                # dominantTris != NULL
        for _ in range(num_verts):
            r.u16_be(); r.u16_be()             # v2, v3
            r.f32_le(); r.f32_le(); r.f32_le()  # normalizationScale (WriteFloat -> native LE)
    r.i32_be(); r.i32_be(); r.i32_be()   # numShadowIndexesNoFrontCaps, numShadowIndexesNoCaps, shadowCapPlaneBits


def _read_binary_render_model_header(r):
    """Consume idRenderModelStatic::WriteBinaryModel's header block, common
    to every .b<ext> binary render model (shared with .bmodel etc.)."""
    magic = r.u32_be()
    if (magic >> 8) != BRM_TAG:
        raise ValueError(
            "Not a Doom 3 BFG binary render model (bad header magic) — "
            "expected a .bmd5mesh produced by the game's binary model cache")
    r.i64_be()                      # timeStamp (ID_TIME_T) — unused for import
    num_surfaces = r.i32_be()
    for _ in range(num_surfaces):
        _skip_generic_surface(r)
    r.vec3_le(); r.vec3_le()        # bounds[0], bounds[1]
    r.i32_be(); r.i32_be(); r.i32_be()  # overlaysAdded, lastModifiedFrame, lastArchivedFrame
    r.string()                      # name
    for _ in range(9):              # isStaticWorldModel .. hasShadowCastingSurfaces (bools)
        r.byte()


def parse_bmd5mesh(filepath):
    """Parse a compiled .bmd5mesh (idRenderModelMD5::WriteBinaryModel).

    Unlike the plaintext .md5mesh format, the binary cache stores fully
    baked bind-pose vertices (already skinned against the default pose,
    already split at UV/tangent seams) with up to 4 packed (joint index,
    byte weight) pairs per vertex instead of a raw weight list — so the
    returned dict intentionally has a different 'meshes' shape than
    parse_md5mesh()'s (see build_mesh_binary()). The 'joints' list,
    however, is normalised to exactly the same shape parse_md5mesh()
    produces (absolute object-space 'pos'/'orient') so build_armature()
    can be reused unchanged.
    """
    with open(filepath, 'rb') as fh:
        data = fh.read()
    r = _BinReader(data)

    _read_binary_render_model_header(r)

    magic = r.u32_be()
    if (magic >> 8) != MD5B_TAG:
        raise ValueError("Not a valid .bmd5mesh (bad MD5 block magic)")

    num_joints     = r.i32_be()
    joint_names    = []
    joint_parents  = []
    for _ in range(num_joints):
        joint_names.append(r.string())
        joint_parents.append(r.i32_be())     # -1 == root

    num_pose   = r.i32_be()
    local_pose = []   # parent-relative (qx, qy, qz, qw, (tx, ty, tz))
    for _ in range(num_pose):
        qx = r.f32_be(); qy = r.f32_be(); qz = r.f32_be(); qw = r.f32_be()
        t  = r.vec3_le()
        local_pose.append((qx, qy, qz, qw, t))

    num_inverted = r.i32_be()
    for _ in range(num_inverted):
        r.bytes_raw(4 * 12)   # invertedDefaultPose (cached GPU skinning matrix) — unused for import

    # The default pose is serialized parent-relative (idRenderModelMD5::
    # WriteBinaryModel converts the absolute joints parsed from .md5mesh
    # text into this form before writing) — the mirror image of what
    # build_frame_skeleton() below already does to compose .md5anim's
    # parent-relative hierarchy/baseframe into object space. Re-use that
    # exact composition so the result matches parse_md5mesh()'s absolute
    # 'pos'/'orient' convention.
    #
    # One extra wrinkle: unlike the plaintext format (where w is always
    # dropped and reconstructed via quat_compute_w()'s -sqrt, matching
    # this addon's convention throughout), the binary defaultPose stores
    # every joint's TRUE w (from idQuat::CalcW() for the root, or from a
    # idMat3::ToQuat() matrix conversion for every other joint) — and
    # engine-internal quaternions built that way turned out (verified
    # empirically against parse_md5mesh() on a real model) to be the
    # inverse rotation of what this addon's convention expects. Taking
    # the conjugate (negate x/y/z, keep the file's real w) corrects that
    # — simply forcing w negative like quat_compute_w() does is NOT
    # equivalent and silently mis-rotates any joint whose true w already
    # happened to be negative.
    joints     = []
    abs_pos    = [None] * num_joints
    abs_orient = [None] * num_joints
    for i in range(num_joints):
        parent = joint_parents[i]
        qx, qy, qz, qw, t = local_pose[i]
        local_q = Quaternion((qw, -qx, -qy, -qz))
        if parent < 0:
            abs_orient[i] = local_q
            abs_pos[i]    = Vector(t)
        else:
            abs_orient[i] = (abs_orient[parent] @ local_q).normalized()
            abs_pos[i]    = abs_orient[parent] @ Vector(t) + abs_pos[parent]
        joints.append({
            'name':   joint_names[i],
            'parent': parent,
            'pos':    tuple(abs_pos[i]),
            'orient': abs_orient[i],
        })

    num_meshes = r.i32_be()
    meshes     = []
    for _ in range(num_meshes):
        shader = r.string()
        r.i32_be()                     # numVerts (pre-dedup source count) — unused
        r.i32_be()                     # numTris  (source tri count) — unused, recomputed from indexes
        num_mesh_joints = r.i32_be()
        r.bytes_raw(num_mesh_joints)   # meshJoints — global joint indices used by this mesh;
                                        # not needed since color[] below already stores global indices
        r.f32_be()                     # maxJointVertDist

        r.i32_be()                     # deform.numSourceVerts
        num_output_verts = r.i32_be()
        num_indexes  = r.i32_be()
        num_mirrored = r.i32_be()
        num_dup      = r.i32_be()
        num_sil_edges = r.i32_be()

        positions = []
        uvs       = []
        vgroups   = []
        if num_output_verts > 0:
            for _ in range(num_output_verts):
                x = r.f32_be(); y = r.f32_be(); z = r.f32_be()
                sh = r.u16_be(); th = r.u16_be()
                r.bytes_raw(4)              # normal (bind-pose, object-space) — unused, Blender recalculates
                r.bytes_raw(4)              # tangent — unused
                color  = r.bytes_raw(4)     # up to 4 GLOBAL joint indices
                color2 = r.bytes_raw(4)     # matching weight bytes (0-255, normalized to sum 255)
                positions.append((x, y, z))
                uvs.append((_half_to_float(sh), _half_to_float(th)))
                vg = []
                for k in range(4):
                    wbyte = color2[k]
                    if wbyte > 0:
                        vg.append((color[k], wbyte / 255.0))
                vgroups.append(vg)

        indexes = []
        if num_indexes > 0:
            indexes = r.u16_be_array(num_indexes)
            r.u16_be_array(num_indexes)      # silIndexes — unused

        if num_mirrored > 0:
            r.i32_be_array(num_mirrored)     # mirroredVerts — unused
        if num_dup > 0:
            r.i32_be_array(num_dup * 2)      # dupVerts — unused
        if num_sil_edges > 0:
            for _ in range(num_sil_edges):
                r.u16_be(); r.u16_be(); r.u16_be(); r.u16_be()   # p1, p2, v1, v2 — unused

        r.i32_be()   # surfaceNum

        tris = [tuple(indexes[i:i + 3]) for i in range(0, len(indexes) - 2, 3)]

        meshes.append({
            'shader':    shader,
            'positions': positions,
            'uvs':       uvs,
            'tris':      tris,
            'vgroups':   vgroups,
        })

    return {'joints': joints, 'meshes': meshes, 'version': 'binary'}


def parse_bmd5anim(filepath):
    """Parse a compiled .bmd5anim (idMD5Anim::WriteBinary).

    Unlike .bmd5mesh, this format is a near-verbatim field-for-field
    serialization of the same data the plaintext .md5anim parser
    produces (parent-relative hierarchy + baseframe + raw per-frame
    component stream) — so the returned dict matches parse_md5anim()'s
    shape exactly, and every downstream consumer (check_skeleton_
    compatibility, build_frame_skeleton, build_action) works unchanged.
    """
    with open(filepath, 'rb') as fh:
        data = fh.read()
    r = _BinReader(data)

    magic = r.u32_be()
    if (magic >> 8) != B_ANIM_MD5_TAG:
        raise ValueError("Not a valid .bmd5anim (bad header magic)")
    r.i64_be()   # timeStamp (ID_TIME_T) — unused for import

    num_frames               = r.i32_be()
    frame_rate                = r.i32_be()
    r.i32_be()                # animLength (derived; Blender import recomputes fps/frame range itself)
    num_joints                = r.i32_be()
    num_animated_components   = r.i32_be()

    num_bounds = r.i32_be()
    for _ in range(num_bounds):
        r.bytes_raw(4 * 3 * 2)   # per-frame culling bounds (true BE idVec3 pair) — unused for import

    num_hier  = r.i32_be()
    hierarchy = []
    for _ in range(num_hier):
        name        = r.string()
        parent      = r.i32_be()
        flags       = r.i32_be()
        start_index = r.i32_be()
        hierarchy.append({'name': name, 'parent': parent,
                           'flags': flags, 'start_index': start_index})

    num_base  = r.i32_be()
    baseframe = []
    for _ in range(num_base):
        qx = r.f32_be(); qy = r.f32_be(); qz = r.f32_be()
        r.f32_be()            # qw — discarded; build_frame_skeleton recomputes it via
                               # quat_compute_w(), exactly as it does for the text format
        t = r.vec3_le()
        baseframe.append({'pos': t, 'orient': (qx, qy, qz)})

    num_components = r.i32_be()
    # componentFrames carries one extra trailing pad float (JOINT_FRAME_PAD
    # in Anim.cpp) that must be consumed but is never referenced.
    flat = [r.f32_le() for _ in range(num_components + 1)]
    flat = flat[:num_components]

    r.vec3_le()   # totaldelta — root joint's baked movement delta; build_frame_skeleton derives
                   # root motion itself from componentFrames, so this is unused here too (matches
                   # parse_md5anim(), which never reads this value from the text format either)

    frames = []
    if num_animated_components > 0:
        for i in range(num_frames):
            start = i * num_animated_components
            frames.append(flat[start:start + num_animated_components])
    else:
        frames = [[] for _ in range(num_frames)]

    return {
        'hierarchy':                hierarchy,
        'baseframe':                baseframe,
        'frames':                   frames,
        'num_frames':               num_frames,
        'num_joints':               num_joints,
        'frame_rate':               frame_rate,
        'num_animated_components':  num_animated_components,
        'version':                  'binary',
    }


# ---------------------------------------------------------------------------
# Build per-frame skeleton in armature (object) space
# Implements the spec algorithm exactly.
# ---------------------------------------------------------------------------

def build_frame_skeleton(anim, frame_index):
    """
    Returns a list of 4x4 Matrix values (one per joint) representing each
    joint's transform in armature/object space for the given frame.
    """
    hierarchy  = anim['hierarchy']
    baseframe  = anim['baseframe']
    frame_data = anim['frames'][frame_index]
    num_joints = anim['num_joints']

    # world_mats[i] = joint i's 4x4 matrix in armature space
    world_mats = [None] * num_joints

    for i in range(num_joints):
        info  = hierarchy[i]
        base  = baseframe[i]
        flags = info['flags']
        si    = info['start_index']

        # Start from baseframe, selectively override with frame data
        pos    = list(base['pos'])
        orient = list(base['orient'])   # x, y, z

        j = 0
        if flags & 1:  pos[0]    = frame_data[si + j]; j += 1
        if flags & 2:  pos[1]    = frame_data[si + j]; j += 1
        if flags & 4:  pos[2]    = frame_data[si + j]; j += 1
        if flags & 8:  orient[0] = frame_data[si + j]; j += 1
        if flags & 16: orient[1] = frame_data[si + j]; j += 1
        if flags & 32: orient[2] = frame_data[si + j]; j += 1

        ow = quat_compute_w(orient[0], orient[1], orient[2])
        q  = Quaternion((ow, orient[0], orient[1], orient[2]))
        p  = Vector(pos)

        parent = info['parent']
        if parent < 0:
            # Root joint: position and orientation are already in object space
            final_pos    = p
            final_orient = q
        else:
            # Child joint: transform animated local values into object space
            # using the already-computed parent matrix.
            # pos is in parent-local space; rotate it by parent orientation
            # and add parent position (spec formula).
            par_mat      = world_mats[parent]
            par_orient   = par_mat.to_quaternion()
            par_pos      = par_mat.translation

            final_pos    = par_orient @ p + par_pos
            final_orient = (par_orient @ q).normalized()

        # Store as 4x4 matrix for easy reuse by children
        mat             = final_orient.to_matrix().to_4x4()
        mat.translation = final_pos
        world_mats[i]   = mat

    return world_mats


# ---------------------------------------------------------------------------
# Skeleton compatibility check
# ---------------------------------------------------------------------------

def check_skeleton_compatibility(anim, arm_obj):
    hierarchy  = anim['hierarchy']
    pose_bones = arm_obj.pose.bones

    anim_names = [j['name'] for j in hierarchy]
    bone_names = [pb.name for pb in pose_bones]

    if len(anim_names) != len(bone_names):
        return False, (
            f"Joint count mismatch: anim has {len(anim_names)} joints, "
            f"armature has {len(bone_names)} bones.")

    mismatches = [
        f"  [{i}] anim='{a}'  bone='{b}'"
        for i, (a, b) in enumerate(zip(anim_names, bone_names)) if a != b
    ]
    if mismatches:
        detail = '\n'.join(mismatches[:10])
        if len(mismatches) > 10:
            detail += f'\n  … and {len(mismatches) - 10} more'
        return False, f"Joint name mismatches:\n{detail}"

    for i, info in enumerate(hierarchy):
        pb      = pose_bones[i]
        exp_par = info['parent']
        if exp_par < 0:
            if pb.parent is not None:
                return False, (
                    f"Joint '{info['name']}' should be a root joint "
                    f"but bone has parent '{pb.parent.name}'.")
        else:
            if pb.parent is None:
                return False, (
                    f"Joint '{info['name']}' should have parent "
                    f"'{hierarchy[exp_par]['name']}' but bone has no parent.")
            if pb.parent.name != hierarchy[exp_par]['name']:
                return False, (
                    f"Joint '{info['name']}' parent mismatch: "
                    f"anim expects '{hierarchy[exp_par]['name']}', "
                    f"bone parent is '{pb.parent.name}'.")

    return True, ""


# ---------------------------------------------------------------------------
# Bake anim into a Blender Action
# ---------------------------------------------------------------------------

def build_action(anim, arm_obj, action_name, scale=1.0, rot_mat=None, afu=True):
    """
    Convert per-frame joint armature-space matrices into pose-bone
    matrix_basis values and write them as F-curve keyframes.

    Blender's pose evaluation chain for a child bone B with parent P is:
        pose_bone.matrix = parent_pose.matrix
                           @ (R_p.inv @ R_b)
                           @ matrix_basis

    Where R_b = arm_data.bones[B].matrix_local  (rest pose, armature space)
          R_p = arm_data.bones[P].matrix_local

    Solving for matrix_basis, and substituting pose_bone.matrix = joint_world
    and parent_pose.matrix = parent_joint_world:

        Child:  basis = R_b.inv @ R_p @ parent_joint_world.inv @ joint_world
        Root:   basis = R_b.inv @ joint_world

    These both return identity when the joint is at its rest pose, which is
    the correct sanity check.
    """
    if arm_obj.animation_data is None:
        arm_obj.animation_data_create()

    if action_name in bpy.data.actions:
        bpy.data.actions.remove(bpy.data.actions[action_name])
    action = bpy.data.actions.new(name=action_name)
    action.use_fake_user = afu
    arm_obj.animation_data.action = action
    slot = _md5_get_or_create_slot(arm_obj, action)

    num_frames = anim['num_frames']
    hierarchy  = anim['hierarchy']
    arm_data   = arm_obj.data
    pose_bones = arm_obj.pose.bones

    # Cache per-bone rest matrices and set rotation mode
    rest_local     = {}   # bone name -> matrix_local (armature space)
    rest_local_inv = {}   # bone name -> matrix_local inverted
    for info in hierarchy:
        bname = info['name']
        pose_bones[bname].rotation_mode = 'QUATERNION'
        ml = arm_data.bones[bname].matrix_local.copy()
        rest_local[bname]     = ml
        rest_local_inv[bname] = ml.inverted()

    # Pre-create F-curves
    loc_curves = {}
    rot_curves = {}
    for info in hierarchy:
        bname  = info['name']
        dp_loc = f'pose.bones["{bname}"].location'
        dp_rot = f'pose.bones["{bname}"].rotation_quaternion'
        loc_curves[bname] = [_md5_new_fcurve(action, slot, dp_loc, i) for i in range(3)]
        rot_curves[bname] = [_md5_new_fcurve(action, slot, dp_rot, i) for i in range(4)]

    # Pre-allocate keyframe points for performance
    for info in hierarchy:
        bname = info['name']
        for fc in loc_curves[bname]:
            fc.keyframe_points.add(num_frames)
        for fc in rot_curves[bname]:
            fc.keyframe_points.add(num_frames)

    # Write keyframes
    for fi in range(num_frames):
        world_mats    = build_frame_skeleton(anim, fi)
        blender_frame = float(fi + 1)   # Blender frames are 1-based

        for bi, info in enumerate(hierarchy):
            bname       = info['name']
            joint_world = world_mats[bi].copy()
            parent_idx  = info['parent']

            # Apply scale to the translation component of every joint matrix.
            # Rotation (if any) is applied by pre-multiplying the full matrix
            # with the rotation matrix so both position and orientation rotate.
            joint_world.translation *= scale
            if rot_mat is not None:
                joint_world = rot_mat @ joint_world

            if parent_idx < 0:
                # Root bone
                # basis = R_b.inv @ joint_world
                basis = rest_local_inv[bname] @ joint_world
            else:
                # Child bone
                # basis = R_b.inv @ R_p @ parent_joint_world.inv @ joint_world
                par_bname     = hierarchy[parent_idx]['name']
                par_rest      = rest_local[par_bname]
                # Also transform the parent world matrix the same way
                par_world     = world_mats[parent_idx].copy()
                par_world.translation *= scale
                if rot_mat is not None:
                    par_world = rot_mat @ par_world
                par_world_inv = par_world.inverted()
                basis = rest_local_inv[bname] @ par_rest @ par_world_inv @ joint_world

            loc   = basis.to_translation()
            rot_q = basis.to_quaternion()

            for ch, val in enumerate(loc):
                kp = loc_curves[bname][ch].keyframe_points[fi]
                kp.co            = (blender_frame, val)
                kp.interpolation = 'LINEAR'

            for ch, val in enumerate((rot_q.w, rot_q.x, rot_q.y, rot_q.z)):
                kp = rot_curves[bname][ch].keyframe_points[fi]
                kp.co            = (blender_frame, val)
                kp.interpolation = 'LINEAR'

    for info in hierarchy:
        bname = info['name']
        for fc in loc_curves[bname] + rot_curves[bname]:
            fc.update()

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end   = num_frames
    bpy.context.scene.render.fps  = anim.get('frame_rate', 24)

    return action


def build_frame_pose(anim, arm_obj, frame_index=0, scale=1.0, rot_mat=None):
    """Pose arm_obj at ONE frame of anim, creating no Action at all.

    The same matrix_basis solve build_action() performs - see its docstring
    for the derivation - evaluated for a single frame and written straight
    into the pose bones instead of into F-curves.

    What matters here is what it does NOT create. A pose lives on the
    Object and is saved with it, so it survives with no animation data
    whatsoever, while an Action makes Blender re-evaluate the animation,
    the pose, and every dependent Armature modifier on every frame change.
    Keyframe count has nothing to do with it - an F-curve is a continuous
    function, evaluated at whatever frame is current. Measured over a
    whole-map import (mars_city1: 68 MD5 models, 3331 pose bones, 140
    Armature modifiers), viewport playback at 1048x621:

        full Actions      (4,315,647 keyframes)    13.1 - 13.5 fps
        one keyframe each (   15,911 keyframes)    13.0 - 13.3 fps
        posed, no Action  (        0 keyframes)    22.0 - 22.2 fps

    Truncating the Actions recovers nothing whatsoever; only removing them
    does. That is the whole reason this function exists rather than a
    single-keyframe build_action().

    Deliberately does not touch scene.frame_start/frame_end/render.fps the
    way build_action() does; there is no animation here to set a range for.

    Only the .map importer uses this - the .md5anim/.bmd5anim import
    dialogs always want a real Action. Returns the number of bones posed.
    """
    if not anim.get('num_frames'):
        return 0
    hierarchy  = anim['hierarchy']
    arm_data   = arm_obj.data
    pose_bones = arm_obj.pose.bones
    frame_index = max(0, min(int(frame_index), anim['num_frames'] - 1))
    world_mats = build_frame_skeleton(anim, frame_index)

    rest_local     = {}
    rest_local_inv = {}
    for info in hierarchy:
        bname = info['name']
        ml = arm_data.bones[bname].matrix_local.copy()
        rest_local[bname]     = ml
        rest_local_inv[bname] = ml.inverted()

    for bi, info in enumerate(hierarchy):
        bname       = info['name']
        pose_bone   = pose_bones[bname]
        pose_bone.rotation_mode = 'QUATERNION'
        joint_world = world_mats[bi].copy()
        joint_world.translation *= scale
        if rot_mat is not None:
            joint_world = rot_mat @ joint_world

        parent_idx = info['parent']
        if parent_idx < 0:
            basis = rest_local_inv[bname] @ joint_world
        else:
            par_bname = hierarchy[parent_idx]['name']
            par_world = world_mats[parent_idx].copy()
            par_world.translation *= scale
            if rot_mat is not None:
                par_world = rot_mat @ par_world
            basis = (rest_local_inv[bname] @ rest_local[par_bname]
                     @ par_world.inverted() @ joint_world)

        pose_bone.location            = basis.to_translation()
        pose_bone.rotation_quaternion = basis.to_quaternion()

    return len(hierarchy)


# ---------------------------------------------------------------------------
# MD5Anim import entry point
# ---------------------------------------------------------------------------

def import_md5anim(filepath, arm_obj, scale=1.0, rotation='NONE',
                   action_name_prefix="", afu = True):
    anim    = parse_md5anim(filepath)
    rot_mat = ROTATION_PRESETS[rotation]
    ok, msg = check_skeleton_compatibility(anim, arm_obj)
    if not ok:
        raise ValueError(f"Skeleton mismatch:\n{msg}")

    action_name = action_name_prefix + os.path.splitext(os.path.basename(filepath))[0]
    build_action(anim, arm_obj, action_name, scale=scale, rot_mat=rot_mat, afu=afu )
    print(f"[MD5 Import] Anim '{action_name}': "
          f"{anim['num_frames']} frames @ {anim.get('frame_rate', '?')} fps")


def import_bmd5anim(filepath, arm_obj, scale=1.0, rotation='NONE',
                     action_name_prefix="", afu=True):
    """Binary (.bmd5anim) counterpart of import_md5anim() — parse_bmd5anim()
    returns the same dict shape as parse_md5anim(), so everything past
    parsing (skeleton check, action baking) is identical."""
    anim    = parse_bmd5anim(filepath)
    rot_mat = ROTATION_PRESETS[rotation]
    ok, msg = check_skeleton_compatibility(anim, arm_obj)
    if not ok:
        raise ValueError(f"Skeleton mismatch:\n{msg}")

    action_name = action_name_prefix + os.path.splitext(os.path.basename(filepath))[0]
    build_action(anim, arm_obj, action_name, scale=scale, rot_mat=rot_mat, afu=afu)
    print(f"[BMD5 Import] Anim '{action_name}': "
          f"{anim['num_frames']} frames @ {anim.get('frame_rate', '?')} fps")


# ---------------------------------------------------------------------------
# Vertex position calculation  (mesh)
# ---------------------------------------------------------------------------

def compute_vertex_positions(md5, mesh_data):
    joints    = md5['joints']
    weights   = mesh_data['weights']
    positions = []
    for vert in mesh_data['verts']:
        final = Vector((0.0, 0.0, 0.0))
        for j in range(vert['count']):
            w       = weights[vert['start'] + j]
            joint   = joints[w['joint']]
            rotated = quat_rotate_point(joint['orient'], w['pos'])
            final  += (Vector(joint['pos']) + rotated) * w['bias']
        positions.append(final)
    return positions


def coord_convert(v):
    return Vector(v)


# Rotation presets applied to all positions/directions at import time.
# Angles are around the Z-axis (vertical), matching the common need to
# reorient models that were exported facing a different axis.
ROTATION_PRESETS = {
    'NONE':   None,                                    # 0°  – no change
    'X_TO_Y': Matrix.Rotation( math.radians( 90), 4, 'Z'),  # +90° Z
    'Y_TO_X': Matrix.Rotation( math.radians(-90), 4, 'Z'),  # -90° Z
    'R180':   Matrix.Rotation( math.radians(180), 4, 'Z'),  # 180° Z
}
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

def apply_rotation_to_vec(rot_mat, v):
    """Apply a rotation Matrix to a Vector, or return the Vector unchanged."""
    if rot_mat is None:
        return Vector(v)
    return rot_mat @ Vector(v)


def apply_rotation_to_quat(rot_mat, q):
    """Rotate an orientation Quaternion by a rotation Matrix."""
    if rot_mat is None:
        return q
    return (rot_mat.to_quaternion() @ q).normalized()


# ---------------------------------------------------------------------------
# Unified scale handling — shared by every importer/exporter
# (md5mesh, md5anim, and md5camera)
# ---------------------------------------------------------------------------
# idTech4 / Doom 3 world units are inches. Blender's default unit is metres.
#   idTech4 -> Blender :  multiply by MD5_SCALE_IN_TO_M  (inches -> metres)
#   Blender -> idTech4 :  multiply by MD5_SCALE_M_TO_IN  (metres -> inches)
MD5_SCALE_IN_TO_M = 0.0254                    # 1 inch     = 0.0254 m
MD5_SCALE_M_TO_IN = 1.0 / MD5_SCALE_IN_TO_M   # 1 metre    = 39.3701 in (approx.)


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


# ─────────────────────────────────────────────────────────────────────
#  MATERIAL AUTO-GENERATION (optional — needs the companion "idTech4
#  Materials" addon). Duplicated from idTech4_map_io.py's own copy
#  (which itself duplicates idTech4_ase_lwo_io.py's copy) — each
#  idTech4 Blender addon keeps its own copy of small shared helpers
#  instead of depending on another addon's internals, since these
#  addons install independently of each other.
# ─────────────────────────────────────────────────────────────────────

_MATERIAL_IMPORT_ADDON_NAME = "idTech4_material_import"


def _find_installed_addon_module(short_name):
    """Locate an addon module among everything Blender's addon system
    knows about, matching by module basename rather than the full
    module path — a legacy (user-scripts) install is just e.g.
    "idTech4_material_import", while Blender's Extensions system
    prefixes it (e.g. "bl_ext.user_default.idTech4_material_import").
    Returns the module, or None if it isn't installed at all."""
    for mod in addon_utils.modules():
        mod_name = getattr(mod, '__name__', '')
        if mod_name == short_name or mod_name.endswith('.' + short_name):
            return mod
    return None


def get_material_import_addon():
    """Return the idTech4_material_import module (.mtr -> Blender
    material support) if it's both installed AND currently enabled, else
    None. Re-checked on every call (cheap — a scan of already-imported
    addon modules) rather than cached, so toggling the addon on/off in
    Preferences takes effect immediately without needing a Blender
    restart."""
    mod = _find_installed_addon_module(_MATERIAL_IMPORT_ADDON_NAME)
    if mod is None:
        return None
    full_name = mod.__name__
    if not addon_utils.check(full_name)[1]:
        return None
    return sys.modules.get(full_name, mod)


def resolve_materials_root(anchor_filepath, user_root=''):
    """Locate the game/mod's "materials" directory (.mtr files):
      1. A "materials" subfolder under a user-supplied game/mod root
         directory.
      2. A "materials" subfolder under anchor_filepath's own directory.
      3. A "materials" subfolder under each ancestor directory of
         anchor_filepath (covers the common .../base/materials/ layout
         when the imported file lives at .../base/models/foo.md5mesh).
    Returns the resolved directory path, or None if no candidate exists
    on disk."""
    candidates = []
    if user_root:
        candidates.append(os.path.join(user_root, 'materials'))

    file_dir = os.path.dirname(anchor_filepath)
    cur = file_dir
    for _ in range(8):
        candidates.append(os.path.join(cur, 'materials'))
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    for c in candidates:
        c = os.path.normpath(c)
        if os.path.isdir(c):
            return c
    return None


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


def _resolve_material_sources(filepath, derive_from_model, override_base_directory,
                               override_source_path, save_derived_as_default=False):
    """Return (base_directory, mod_base_directory, source_path) for
    generate_materials_for_objects.
    base_directory, in priority order:
      1. derive_from_model=True — walk up from *filepath*'s own directory
         looking for a "materials" folder (resolve_materials_root, no
         hint), base_directory = that folder's parent. If
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
      3. Otherwise the shared idTech4 config file's base_directory —
         the normal case. Blank if none of the above resolved anything.

    mod_base_directory is only ever the shared config's optional Mod
    Base, and ONLY in case 3. A stored Mod Base pairs with the stored
    Base; pairing it with a one-off gate override or a base derived from
    the imported file's own location would search a mod tree that has
    nothing to do with either, so both of those return it blank.

    source_path may be a TUPLE rather than a string. An explicitly
    configured Materials Source is used exactly as given - that is what
    the field is for - but a source DERIVED from the roots is one
    "materials" folder per root, mod first, because a mod ships a
    handful of .mtr files and inherits the rest.

    source_path is mandatory-base/optional-source: whichever of the
    above found a base_directory, source_path follows the same
    priority (override_source_path, else the shared config file's
    materials_mtr_source) but — unlike base_directory — is allowed to
    come up blank, in which case it defaults to base_directory's own
    "materials" subfolder. That folder not actually existing isn't
    checked here; generate_materials_for_objects' own "no .mtr
    materials found under X" issue message already covers a default
    that doesn't pan out, same as it would for a hand-typed wrong
    path."""
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


def generate_materials_for_objects(objects, base_directory, source_path, material_mode, context,
                                   mod_directory='', material_parameters=None):
    """Build real, fully-textured Blender materials — via the companion
    "idTech4 Materials" addon — for every material name already assigned
    somewhere on *objects* (the blank placeholder build_mesh() left
    behind). Returns (issues, built_count, report_name): issues is a list
    of human-readable issue strings (empty on full success); built_count is
    how many materials were actually built; report_name is the Text
    datablock publish_report() wrote, so the caller can NAME it in its own
    result. All three are ([], 0, None) if that addon isn't
    installed/enabled, either path is blank, or nothing ended up matching
    an .mtr source. See idTech4_ase_lwo_io.py's copy for why the third
    value is there.

    *base_directory* and *source_path* are independent — see
    idTech4_ase_lwo_io.py's copy of this function for the full reasoning
    (this file mirrors it). Callers resolve which values to pass in via
    _resolve_material_sources; this function does no resolution of its
    own.

    Mirrors idTech4_map_io.py's own material step — scoped to
    *objects*' own material_slots. Name normalisation, the
    case-insensitive fallback and the table priming all live behind the
    materials addon's public API now (API_VERSION 2). Unlike
    idTech4_ase_lwo_io.py's copy this needs no extension strip either:
    build_mesh names the material after the MD5Mesh's own "shader" line
    verbatim, which is already the bare .mtr decl name.
    """
    material_addon = get_material_import_addon()
    if material_addon is None:
        return [], 0, None

    # source_path is only ever blank here when base_directory is too
    # (_resolve_material_sources defaults source_path to base_directory's
    # own "materials" subfolder whenever base_directory is set) — a
    # default that doesn't actually exist falls through to the "no .mtr
    # materials found" branch below instead, not this one.
    roots = shared_search_roots(base_directory, mod_directory)
    if not roots:
        return ["Material import was requested, but Base Directory "
                "isn't set — no materials were created."], 0, None

    # material_targets() maps each canonical decl name to the datablocks it
    # is actually used under, so the build fills THOSE rather than a fresh
    # datablock named after the decl.
    targets = material_addon.material_targets(objects)
    if not targets:
        return [], 0, None

    if not material_addon.material_names(source_path, base_directory,
                                         mod_dir=mod_directory):
        # Name every place that was actually looked in, not just the
        # first: with a Mod Base set source_path is one folder per root,
        # and "not found under <one path>" would send the user checking
        # a directory that was never the whole story.
        searched = '", "'.join(source_path) if isinstance(
            source_path, (list, tuple)) else source_path
        return [f"Material import was requested, but no .mtr materials "
                f"were found under \"{searched}\" — no materials "
                f"were created."], 0, None

    # params=None is build_materials' own "use the Materials panel's
    # parameter_policy" fallback, which is what every caller here did
    # before the dialogs grew their own Parameters control. Callers that
    # pass one override it.
    summary = material_addon.build_materials(
        sorted(targets), mode=material_mode, params=material_parameters,
        context=context, base_dir=base_directory, source=source_path,
        targets=targets, mod_dir=mod_directory)

    # Publish to the Materials panel's Report and to the full report text,
    # exactly as that panel's own Generate Materials button does. See the
    # matching comment in idTech4_ase_lwo_io.py: without it a model import
    # built the materials and reported nothing anywhere.
    report_text = material_addon.publish_report(summary, context)

    if hasattr(context.scene, 'idtech4_settings'):
        context.scene.idtech4_settings.generation_mode = material_mode
        # The panel's shader parm sliders only do anything for materials
        # built under DYNAMIC, so its policy has to agree with what was
        # actually built here or the sliders lie.
        if material_parameters:
            context.scene.idtech4_settings.parameter_policy = material_parameters

    return (list(summary.errors), summary.built,
            getattr(report_text, 'name', None))


# The half of the two source-path tooltips below that is the same for
# both. Blender renders a text field's tooltip as the property's
# description followed by the field's own full value, which is the point
# of drawing these as locked fields rather than as labels: a path too
# long for the dialog's width is truncated on screen but shown whole on
# hover, and the sentence below says why it can't be typed over.
# Kept in step with idTech4_map_io.py's own copy - these three
# fields are meant to read identically in every import dialog of the
# toolchain, not just similarly.
_SOURCE_DISPLAY_TIP = (
    "Source directories cannot be modified from the import dialog. To "
    "change the source directories, cancel the import and use the idTech4 "
    "side (N) panel, or derive from the imported file path")


def _get_import_base_display(self):
    """Read-only Base Directory for the import dialog's locked field.

    Same effective value _resolve_material_sources would pick for this
    import, minus the derive-from-model walk (which needs a chosen file
    and is what the field greys out for anyway): the sources gate's
    one-off override if there is one, else the shared config's Base
    Directory.

    A StringProperty given a `get` but no `set` is read-only at the RNA
    level, which is what draws the field locked - the same pattern the
    Sources panel's own fields use."""
    shared_base, _mod, _ = get_shared_paths()
    return (self.override_base_directory or shared_base) or "(not set)"


def _get_import_mod_base_display(self):
    """Read-only Mod Base for the import dialog's locked field - see
    _get_import_base_display. A gate override replaces the configured
    Base outright and so replaces the Mod Base that went with it, which
    is why an override reads as no mod rather than as the stored one:
    that is exactly what _resolve_material_sources will do."""
    _base, shared_mod, _ = get_shared_paths()
    if self.override_base_directory:
        return "(none - overridden for this import)"
    return shared_mod or "(none - using Base Directory only)"


def _get_import_source_display(self):
    """Read-only Materials Source for the import dialog's locked field -
    see _get_import_base_display. Spells out the unset-but-defaulted case
    the way _resolve_material_sources resolves it, so the field shows the
    path that will actually be read rather than an empty box."""
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


class MaterialGenMixin:
    """Shared "auto-generate materials" properties/UI for the MD5 mesh
    import operator — needs the companion "idTech4 Materials" addon; see
    generate_materials_for_objects. Mirrors the .map importer's own
    Import Materials / Material Mode / Game-Mod-Root controls, except
    defaulting to MAXIMUM mode here rather than SIMPLE (a single
    imported mesh is cheap enough to build at full fidelity, unlike a
    whole level's worth of materials)."""

    import_materials: BoolProperty(
        name="Auto-Generate Materials",
        description="Build real, fully-textured Blender materials from "
                    "the .mtr source tree found under the directory "
                    "below (or auto-detected by walking up from the "
                    "imported file's own location looking for a "
                    "\"materials\" folder), for every material name this "
                    "import assigns — instead of the blank placeholder "
                    "materials it would otherwise get. Needs the "
                    "companion \"idTech4 Materials\" addon",
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
        default='MAXIMUM',
    )
    # The other half of the material addon's own build settings: what to
    # do with every `time`/parm/global/sound/table term a material's
    # expressions reach. Listed here rather than left to whatever the
    # Materials panel happens to be set to, so an import dialog fully
    # describes what it is about to build. Item ids match
    # idTech4_material_import's PARAMS_* constants, which is what gets
    # passed straight through to build_materials(params=...);
    # duplicated rather than imported for the same reason
    # material_mode's list is (the addon may not be installed when this
    # class is defined).
    #
    # Defaults to DYNAMIC, where the .map importer defaults to BAKED:
    # the two differ for the same reason their Material Mode defaults
    # do. A single imported model is a handful of materials, so the
    # per-frame cost of a driver is affordable and being able to scrub
    # an animated material - or move the Materials panel's parm
    # sliders and watch it respond - is worth more than the frame time.
    # A whole level's worth is not (219 drivers were 74ms of a 91ms
    # frame on mars_city1), which is why that dialog bakes.
    material_parameters: EnumProperty(
        name="Parameters",
        description="What the \"idTech4 Materials\" addon does with time, "
                    "parm0..11, global0..7, sound and table lookups while "
                    "building each material",
        items=[
            ('DYNAMIC', "Dynamic", "Drivers for expressions that reach "
             "`time`, and a recorded expression re-folded on slider moves "
             "for the rest. The material animates with the timeline and "
             "responds to the Materials panel's parm sliders. Costs real "
             "frame time - on a whole level that is prohibitive, on a "
             "single model it is not"),
            ('BAKED', "Baked", "Fold every expression once, now, against "
             "the current frame and the Materials panel's slider values. No "
             "drivers at all, so nothing re-runs per frame and nothing "
             "depends on the driver namespace surviving a file load. Moving "
             "a slider afterwards needs a rebuild"),
            ('SKIP', "Skip", "Refuse every parameter. Conditional stages "
             "are dropped without being evaluated and dynamic terms use "
             "their neutral defaults. The cheapest result, and the least "
             "faithful"),
        ],
        default='DYNAMIC',
    )
    derive_from_model: BoolProperty(
        name="Derive from model",
        description="Ignore Base/Materials below and instead derive them "
                    "for THIS import by walking up from the imported "
                    "file's own directory looking for a \"materials\" "
                    "folder (Base = that folder's parent)",
        default=False,
    )
    # Display only - read-only mirrors of what _resolve_material_sources
    # will use, drawn as locked text fields so a long path is truncated
    # on screen but readable in full on hover. Read-only at the RNA level
    # (a `get` with no `set`), so nothing here can be typed over or
    # saved; the editable copies live in the 3D Viewport sidebar's
    # Sources panel. Same three fields, same tooltips, as the .map
    # importer's own dialog.
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
    # operator (MaterialsSourceGateMixin) when the user picks "Select
    # now" without also checking "Set as default", so this one import
    # still uses what they typed without it being written into the
    # shared idTech4 config file. See _resolve_material_sources.
    override_base_directory: StringProperty(default='', options={'HIDDEN', 'SKIP_SAVE'})
    override_source_path: StringProperty(default='', options={'HIDDEN', 'SKIP_SAVE'})
    # Internal — set by the sources gate when the user picks "Don't set
    # up materials for this import", to force Auto-Generate Materials
    # off AND non-interactive for this one invocation (rather than just
    # defaulting it off, which the user could still re-check).
    materials_setup_skipped: BoolProperty(default=False, options={'HIDDEN', 'SKIP_SAVE'})
    # Internal — set by the sources gate's "...and save settings" derive
    # choice. Threaded through to _resolve_material_sources, which is
    # the only place the actual derived paths become known (this
    # import's own filepath isn't known until now, after the gate's
    # popup has already closed).
    save_derived_as_default: BoolProperty(default=False, options={'HIDDEN', 'SKIP_SAVE'})
    # Internal — set once by the sources gate right before it re-invokes
    # this operator with its resolved values, so _needs_sources_gate
    # knows not to fire a second time for the same import (Derive
    # Automatically can still come back empty, and Select Now doesn't
    # force retrying — either way the gate only ever runs once).
    gate_resolved: BoolProperty(default=False, options={'HIDDEN', 'SKIP_SAVE'})

    def _needs_sources_gate(self):
        """True if Auto-Generate Materials is checked, its companion
        addon is installed/enabled, and Base Directory isn't already
        resolvable — i.e. showing the sources gate popup would actually
        accomplish something. Materials Source is NOT checked here —
        it's optional, defaulting to Base Directory's own "materials"
        subfolder (see _resolve_material_sources), so its own absence
        never blocks anything on its own."""
        if self.materials_setup_skipped or self.gate_resolved:
            return False
        if not (self.import_materials and get_material_import_addon() is not None):
            return False
        base_directory, mod_directory, _ = _resolve_material_sources(
            self.filepath, self.derive_from_model,
            self.override_base_directory, self.override_source_path)
        # Roots, not base alone: a Mod Base on its own is a usable tree,
        # so prompting for a Base Directory the user has deliberately
        # left out would be a gate that can never be satisfied.
        return not shared_search_roots(base_directory, mod_directory)

    def _launch_sources_gate(self):
        """Bail out to this operator's own sources gate — named, by
        convention, this operator's own bl_idname + "_gate" — passing
        this operator's already-chosen properties through
        _pending_import_kwargs (a module global; see that global's own
        comment for why that's simpler here than mirroring properties
        onto every gate class just to receive them).

        Uses type(self).bl_idname rather than self.bl_idname: on a live
        operator instance, Blender resolves the latter through RNA to
        the "IMPORT_SCENE_OT_md5mesh"-style identifier (no dot) instead
        of the plain dotted string ("import_scene.md5mesh") set on the
        class, which breaks the split() below."""
        global _pending_import_kwargs
        _pending_import_kwargs = self.as_keywords(ignore=('filepath',))
        category, name = (type(self).bl_idname + '_gate').split('.')
        getattr(getattr(bpy.ops, category), name)('INVOKE_DEFAULT', filepath=self.filepath)

    def draw_material_gen(self, context, layout):
        layout.separator()
        addon_available = get_material_import_addon() is not None
        forced_off = addon_available and self.materials_setup_skipped
        row = layout.row()
        row.enabled = addon_available and not forced_off
        row.prop(self, "import_materials")
        if not addon_available:
            layout.label(text="Needs the \"idTech4 Materials\" addon "
                              "(not installed/enabled)", icon='ERROR')
        elif forced_off:
            layout.label(text="Materials setup was skipped for this import",
                         icon='INFO')
        col = layout.column()
        col.enabled = self.import_materials and addon_available and not forced_off
        # Label above the dropdown rather than beside it, matching the
        # .map importer's own Material Mode control: the enum names are
        # long enough that Blender's own property split left the
        # dropdown almost no width in a normal-width dialog.
        col.label(text="Material Mode:")
        col.prop(self, "material_mode", text="")
        col.label(text="Parameters:")
        col.prop(self, "material_parameters", text="")
        col.separator()
        # Source Directories in the .map importer's shape: a label above
        # a locked field rather than one "Base: <path>" line. A label
        # never wraps and never truncates - it just runs off the
        # dialog's edge, taking the tail of the path (the part that
        # identifies it) with it. The field truncates instead, and
        # hovering it shows the whole path plus why it can't be edited
        # here (see _SOURCE_DISPLAY_TIP), which is what replaced the
        # four "To change these:" lines this used to spell out inline.
        col.label(text="Source Directories")
        box = col.box()
        # Derive from model heads the box because it decides what the
        # three fields under it are even showing. It must stay OUTSIDE
        # the greyed-out part: the fields grey out when it is checked
        # (they no longer describe the import), and a checkbox that
        # greys ITSELF out the moment it is ticked cannot be unticked.
        box.prop(self, "derive_from_model")
        paths = box.column()
        paths.enabled = not self.derive_from_model
        sub = paths.column(align=True)
        sub.label(text="Base:")
        sub.prop(self, "display_base_directory", text="")
        sub = paths.column(align=True)
        sub.label(text="Mod Base:")
        sub.prop(self, "display_mod_base_directory", text="")
        sub = paths.column(align=True)
        sub.label(text="Materials:")
        sub.prop(self, "display_materials_source", text="")


class MD5_ExportScaleRotMixin:
    """Shared Scale properties/UI for every MD5 import operator
    (md5mesh, md5anim, md5camera). Scaling is OFF by default (1:1, no
    change). When enabled, choose either an arbitrary numeric factor or the
    idTech4 -> Blender (inches -> meters) preset."""

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


# ---------------------------------------------------------------------------
# Blender object builders  (mesh)
# ---------------------------------------------------------------------------

def _new_armature_object(collection_name, target_collection):
    """Create a new, bone-less Armature object linked into
    target_collection, ready for _populate_armature_bones() once the
    caller has put it into Edit Mode. Split out of build_armature() so
    a caller building MANY armatures in one go (e.g. an entire .map
    import placing dozens of MD5 models) can batch them all into a
    SINGLE Edit Mode session instead of one bpy.ops.object.mode_set()
    round trip per armature — see build_armature()'s docstring for why
    that matters."""
    arm_name = collection_name + "_MD5_Armature"
    arm_data = bpy.data.armatures.new(arm_name)
    arm_obj  = bpy.data.objects.new(arm_name, arm_data)
    target_collection.objects.link(arm_obj)
    return arm_obj


def _populate_armature_bones(arm_obj, md5, scale, rot_mat=None):
    """Build every bone of arm_obj from md5['joints']. arm_obj MUST
    already be in Edit Mode when this is called — either alone (via
    bpy.ops.object.mode_set(mode='EDIT') with just arm_obj active/
    selected, as build_armature() does), or as one of several objects
    selected together for a batched multi-object Edit Mode session
    (Blender supports editing multiple Armature objects' bones in one
    such session) — split out of build_armature() for that batching
    case; see its docstring."""
    joints      = md5['joints']
    edit_bones  = arm_obj.data.edit_bones
    BONE_LENGTH = 5.0 * scale
    bone_list   = []

    for j in joints:
        bone      = edit_bones.new(j['name'])
        orient    = apply_rotation_to_quat(rot_mat, j['orient'])
        head      = apply_rotation_to_vec(rot_mat, j['pos']) * scale

        # Point the bone Y-axis along the joint orientation's Y axis.
        # This makes Blender's bone direction match the MD5 joint Y axis.
        local_y   = quat_rotate_point(orient, Vector((0.0, BONE_LENGTH, 0.0)))
        bone.head = head
        bone.tail = head + Vector(local_y)

        # Set roll so the bone's local Z axis matches the joint's Z axis.
        # align_roll() solves for the roll angle that minimises the angle
        # between the bone's computed Z and the supplied reference vector,
        # making matrix_local match the full 3-D joint orientation quaternion.
        local_z   = quat_rotate_point(orient, Vector((0.0, 0.0, 1.0)))
        bone.align_roll(Vector(local_z))

        bone_list.append(bone)

    for i, j in enumerate(joints):
        if j['parent'] >= 0:
            bone_list[i].parent = bone_list[j['parent']]


def _finish_armature_object(arm_obj):
    """Post-Edit-Mode armature finishing (the MD5_export_bone_collection
    setup) — split out of build_armature() for the same batching reason
    as _new_armature_object() / _populate_armature_bones(); must be
    called once arm_obj is back in Object Mode."""
    bone_coll = arm_obj.data.collections.new("MD5_export_bone_collection")
    for pose_bone in arm_obj.pose.bones:
        bone_coll.assign(pose_bone)


def build_armature(md5, collection_name, scale, target_collection, rot_mat=None):
    """
    Build one complete Armature object from md5['joints'] in a single
    call — parse, create, populate bones, and finish, exactly as
    before this function was split into _new_armature_object() /
    _populate_armature_bones() / _finish_armature_object() (see those
    for why: a caller building many armatures at once, like the .map
    importer placing dozens of MD5 models, can call those three
    directly instead to batch every armature's Edit Mode work into one
    bpy.ops.object.mode_set() round trip rather than paying for one
    per armature — bpy.ops.object.mode_set() was measured directly to
    cost ~1.5ms on a near-empty scene but ~500ms+ once the scene
    already has a few thousand objects (e.g. mid-way through a large
    .map import), a cost that's paid per CALL regardless of how many
    bones are built inside that one Edit Mode session, not per bone).
    This function's own behavior — one armature, one mode_set() pair —
    is unchanged; it's just now a thin wrapper around the split-out
    pieces.
    """
    arm_obj = _new_armature_object(collection_name, target_collection)
    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)

    bpy.ops.object.mode_set(mode='EDIT')
    _populate_armature_bones(arm_obj, md5, scale, rot_mat)
    bpy.ops.object.mode_set(mode='OBJECT')

    _finish_armature_object(arm_obj)

    return arm_obj


def get_or_create_shader_material(name):
    """Return the material called *name*, creating it only if absent.

    bpy.data.materials.new() always makes a NEW datablock and lets Blender
    uniquify the name, so importing the same .md5mesh five times - which one
    .map does routinely, five monsters sharing a model - left
    "models/monsters/zombie/boney/boney" plus .001 .. .004 behind, four of
    them never reached by material generation and rendering flat white.
    Reusing by exact name means every instance shares the one datablock the
    generator will fill in.
    """
    existing = bpy.data.materials.get(name)
    if existing is not None:
        return existing
    return bpy.data.materials.new(name=name)


def build_mesh(md5, mesh_idx, mesh_data, name, scale, arm_obj, target_collection, rot_mat=None, reconstruct_normals=True, ignore_normals=False,smooth = False):
    positions = compute_vertex_positions(md5, mesh_data)
    joints    = md5['joints']

    shader    = mesh_data['shader']
    # The shader is a decl name, and idTech4 content spells one with
    # either separator: quake4's napalmgun.md5mesh writes
    # "models\weapons\napalmgun\w_flamegun", which has no forward
    # slash at all, so rsplit('/') returned the WHOLE string and named the
    # mesh object after the full path. Separators are folded the way
    # engine_canonical_decl folds them; the case is left alone, because
    # this is the object's display name and 209 shipped shaders carry
    # uppercase that the user has no reason to see lowercased.
    mesh_name = (shader.replace('\\', '/').rsplit('/', 1)[-1]
                 if shader else f"md5_mesh{mesh_idx}")
    if not mesh_name:
        mesh_name = f"md5_mesh{mesh_idx}"

    me = bpy.data.meshes.new(mesh_name)
    ob = bpy.data.objects.new(mesh_name, me)
    target_collection.objects.link(ob)

    verts_co = [apply_rotation_to_vec(rot_mat, p) * scale for p in positions]
    faces    = [(v0, v2, v1) for v0, v1, v2 in mesh_data['tris']]  # flip normals
    me.from_pydata(verts_co, [], faces)
    if smooth:
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    me.update()
    if len(me.vertices) != len(mesh_data['verts']):
        print(f"[MD5 Import] WARNING {mesh_name}: "
              f"from_pydata created {len(me.vertices)} verts "
              f"but MD5 has {len(mesh_data['verts'])} verts — index mismatch!")

    uv_layer = me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        for loop_idx, vert_idx in zip(poly.loop_indices, poly.vertices):
            s, t = mesh_data['verts'][vert_idx]['uv']
            uv_layer.data[loop_idx].uv = (s, 1.0 - t)

    # Vertex colors (v12 only) — stored as a byte-color attribute
    colors = mesh_data.get('colors')
    if colors and len(colors) == len(mesh_data['verts']):
        vc_attr = me.color_attributes.new(
            name="Color", type='FLOAT_COLOR', domain='POINT')
        for vi, (r, g, b, a) in enumerate(colors):
            vc_attr.data[vi].color = (r, g, b, a)

    mat = get_or_create_shader_material(shader or f"md5_mat_{mesh_idx}")
    ob.data.materials.append(mat)

    v12_tag = " [v12]" if mesh_data.get('verts') and mesh_data['verts'][0].get('normal') else ""
    print(f"[MD5 Import] {mesh_name}{v12_tag}: "
          f"{len(mesh_data['verts'])} md5 verts, "
          f"{len(mesh_data['weights'])} weights, "
          f"{len(mesh_data['tris'])} tris"
          + (f", {len(colors)} vertex colors" if colors else ""))

    for j in joints:
        ob.vertex_groups.new(name=j['name'])

    for vi, vert in enumerate(mesh_data['verts']):
        for k in range(vert['count']):
            w = mesh_data['weights'][vert['start'] + k]
            if w['bias'] > 0.0:
                ob.vertex_groups[w['joint']].add([vi], w['bias'], 'ADD')

    if arm_obj:
        mod        = ob.modifiers.new("Armature", 'ARMATURE')
        mod.object = arm_obj

    # V12 normal reconstruction -----------------------------------------------
    # Each vert stores a normal in dominant-bone-local space.
    # Reverse the export transform: bone_3x3 @ normal_local → world space,
    # then apply the import rotation.  Set as Blender custom split normals.
    if not ignore_normals and reconstruct_normals and arm_obj:
        has_normals = (mesh_data.get('verts') and
                       mesh_data['verts'][0].get('normal') is not None)
        if has_normals:
            arm_data  = arm_obj.data
            arm_world = arm_obj.matrix_world

            # Pre-build bone 3x3 world matrices (rotation only, no scale)
            bone_mat3 = {}
            for j in joints:
                bm = arm_world @ arm_data.bones[j['name']].matrix_local
                bone_mat3[j['name']] = bm.to_3x3()

            # For each MD5 vert, find dominant joint (highest bias)
            def dominant_joint_name(vert):
                best_joint, best_bias = 0, -1.0
                for k in range(vert['count']):
                    w = mesh_data['weights'][vert['start'] + k]
                    if w['bias'] > best_bias:
                        best_bias  = w['bias']
                        best_joint = w['joint']
                return joints[best_joint]['name']

            # Build per-vertex world-space normals
            vert_normals = []
            for vi, vert in enumerate(mesh_data['verts']):
                nx, ny, nz = vert['normal']
                n_local    = Vector((nx, ny, nz))
                jname      = dominant_joint_name(vert)
                n_world    = (bone_mat3[jname] @ n_local).normalized()
                if rot_mat is not None:
                    n_world = (rot_mat.to_3x3() @ n_world).normalized()
                vert_normals.append(n_world)

            # Apply as custom split normals (per-loop, indexed by vert index)
            loop_normals = [None] * len(me.loops)
            for poly in me.polygons:
                for loop_idx, vert_idx in zip(poly.loop_indices, poly.vertices):
                    loop_normals[loop_idx] = vert_normals[vert_idx]

            me.normals_split_custom_set(loop_normals)
            print(f"[MD5 Import] {mesh_name}: v12 normals reconstructed")

    return ob


# ---------------------------------------------------------------------------
# MD5Mesh import entry point
# ---------------------------------------------------------------------------

def _set_viewport_shading_material_preview(context):
    """Switch every 3D viewport in the current screen to Material Preview
    shading — called once real, textured materials have actually been
    auto-generated on import, so the result is immediately visible
    instead of still showing flat Solid shading. A no-op in a headless/
    background context with no screen (e.g. a script running with no
    open window)."""
    screen = getattr(context, 'screen', None)
    if screen is None:
        return
    for area in screen.areas:
        if area.type != 'VIEW_3D':
            continue
        space = area.spaces.active
        if space is not None and space.type == 'VIEW_3D':
            space.shading.type = 'MATERIAL'


def _new_import_collection(name):
    """Create a new scene-root collection named *name* and make it active
    in the outliner (recursive search so this works on all 4.x/5.x
    versions even when the collection is nested). Shared by every MD5
    mesh importer (text and binary)."""
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)

    def _find_layer_collection(layer_col, name):
        if layer_col.name == name:
            return layer_col
        for child in layer_col.children:
            found = _find_layer_collection(child, name)
            if found:
                return found
        return None
    lc = _find_layer_collection(
        bpy.context.view_layer.layer_collection, col.name)
    if lc:
        bpy.context.view_layer.active_layer_collection = lc
    return col


def import_md5mesh(filepath, scale=1.0, rotation='NONE', reconstruct_normals=True, ignore_normals=False, smooth = False,
                    import_materials=True, material_mode='GOOD',
                    material_parameters=None, derive_from_model=False,
                    override_base_directory='', override_source_path='',
                    save_derived_as_default=False):
    """*import_materials*/*material_mode*/*material_parameters*/
    *derive_from_model*/*override_base_directory*/*override_source_path*/
    *save_derived_as_default*: see _resolve_material_sources and
    generate_materials_for_objects — needs the companion "idTech4
    Materials" addon; silently does nothing (leaving each mesh's blank
    placeholder material, named after its MD5 "shader" line) if that
    addon isn't installed/enabled.

    Returns (result_set, material_issues, report_name) — material_issues
    is a list of
    human-readable strings (empty on full success / when materials
    weren't requested)."""
    name    = os.path.splitext(os.path.basename(filepath))[0]
    rot_mat = ROTATION_PRESETS[rotation]
    print(f"[MD5 Import] Parsing mesh: {filepath}")
    md5 = parse_md5mesh(filepath)

    version = md5.get('version', 10)
    if version not in (10, 12):
        print(f"[MD5 Import] WARNING: unexpected MD5Version {version} "
              f"(supported: 10, 12) — attempting import anyway")
    else:
        print(f"[MD5 Import] MD5Version {version}")

    col = _new_import_collection(name)

    arm_obj = build_armature(md5, name, scale, col, rot_mat)

    mesh_objects = []
    for i, mesh_data in enumerate(md5['meshes']):
        ob = build_mesh(md5, i, mesh_data, name, scale, arm_obj, col, rot_mat,
                   reconstruct_normals=reconstruct_normals,
                   ignore_normals=ignore_normals,smooth = smooth)
        mesh_objects.append(ob)

    material_issues = []
    material_report = None
    if import_materials and mesh_objects:
        base_directory, mod_directory, source_path = _resolve_material_sources(
            filepath, derive_from_model, override_base_directory,
            override_source_path,
            save_derived_as_default=save_derived_as_default)
        material_issues, built_count, material_report = \
            generate_materials_for_objects(
                mesh_objects, base_directory, source_path, material_mode,
                bpy.context, mod_directory=mod_directory,
                material_parameters=material_parameters)
        if built_count > 0:
            _set_viewport_shading_material_preview(bpy.context)

    print(f"[MD5 Import] Done — {len(md5['joints'])} joints, "
          f"{len(md5['meshes'])} meshes.")
    return {'FINISHED'}, material_issues, material_report


# ---------------------------------------------------------------------------
# Binary MD5 mesh (.bmd5mesh) import — mesh building + entry point
# ---------------------------------------------------------------------------

def build_mesh_binary(md5, mesh_idx, mesh_data, scale, arm_obj, target_collection,
                       rot_mat=None, smooth=False):
    """Binary-format counterpart of build_mesh(). The .bmd5mesh vertex data
    is already fully baked (final bind-pose position, up to 4 packed
    (joint index, byte weight) pairs) — no compute_vertex_positions() /
    weight-list expansion needed, unlike the plaintext path."""
    joints = md5['joints']

    shader    = mesh_data['shader']
    # The shader is a decl name, and idTech4 content spells one with
    # either separator: quake4's napalmgun.md5mesh writes
    # "models\weapons\napalmgun\w_flamegun", which has no forward
    # slash at all, so rsplit('/') returned the WHOLE string and named the
    # mesh object after the full path. Separators are folded the way
    # engine_canonical_decl folds them; the case is left alone, because
    # this is the object's display name and 209 shipped shaders carry
    # uppercase that the user has no reason to see lowercased.
    mesh_name = (shader.replace('\\', '/').rsplit('/', 1)[-1]
                 if shader else f"bmd5_mesh{mesh_idx}")
    if not mesh_name:
        mesh_name = f"bmd5_mesh{mesh_idx}"

    me = bpy.data.meshes.new(mesh_name)
    ob = bpy.data.objects.new(mesh_name, me)
    target_collection.objects.link(ob)

    verts_co = [apply_rotation_to_vec(rot_mat, p) * scale for p in mesh_data['positions']]
    faces    = [(v0, v2, v1) for v0, v1, v2 in mesh_data['tris']]  # flip normals, matches build_mesh()
    me.from_pydata(verts_co, [], faces)
    if smooth:
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    me.update()

    uv_layer = me.uv_layers.new(name="UVMap")
    for poly in me.polygons:
        for loop_idx, vert_idx in zip(poly.loop_indices, poly.vertices):
            s, t = mesh_data['uvs'][vert_idx]
            uv_layer.data[loop_idx].uv = (s, 1.0 - t)

    mat = get_or_create_shader_material(shader or f"bmd5_mat_{mesh_idx}")
    ob.data.materials.append(mat)

    print(f"[BMD5 Import] {mesh_name}: "
          f"{len(mesh_data['positions'])} verts, {len(mesh_data['tris'])} tris")

    for j in joints:
        ob.vertex_groups.new(name=j['name'])

    for vi, vg in enumerate(mesh_data['vgroups']):
        for joint_idx, weight in vg:
            if 0 <= joint_idx < len(joints):
                ob.vertex_groups[joint_idx].add([vi], weight, 'ADD')

    if arm_obj:
        mod        = ob.modifiers.new("Armature", 'ARMATURE')
        mod.object = arm_obj

    return ob


def import_bmd5mesh(filepath, scale=1.0, rotation='NONE', smooth=False,
                     import_materials=True, material_mode='GOOD',
                     material_parameters=None, derive_from_model=False,
                     override_base_directory='', override_source_path='',
                     save_derived_as_default=False):
    """Binary (.bmd5mesh) counterpart of import_md5mesh(). Returns
    (result_set, material_issues, report_name), same as import_md5mesh()."""
    name    = os.path.splitext(os.path.basename(filepath))[0]
    rot_mat = ROTATION_PRESETS[rotation]
    print(f"[BMD5 Import] Parsing binary mesh: {filepath}")
    md5 = parse_bmd5mesh(filepath)

    col     = _new_import_collection(name)
    arm_obj = build_armature(md5, name, scale, col, rot_mat)

    mesh_objects = []
    for i, mesh_data in enumerate(md5['meshes']):
        ob = build_mesh_binary(md5, i, mesh_data, scale, arm_obj, col, rot_mat, smooth=smooth)
        mesh_objects.append(ob)

    material_issues = []
    material_report = None
    if import_materials and mesh_objects:
        base_directory, mod_directory, source_path = _resolve_material_sources(
            filepath, derive_from_model, override_base_directory,
            override_source_path,
            save_derived_as_default=save_derived_as_default)
        material_issues, built_count, material_report = \
            generate_materials_for_objects(
                mesh_objects, base_directory, source_path, material_mode,
                bpy.context, mod_directory=mod_directory,
                material_parameters=material_parameters)
        if built_count > 0:
            _set_viewport_shading_material_preview(bpy.context)

    print(f"[BMD5 Import] Done — {len(md5['joints'])} joints, "
          f"{len(md5['meshes'])} meshes.")
    return {'FINISHED'}, material_issues, material_report


# ---------------------------------------------------------------------------
# Armature enum helper for anim operator
# ---------------------------------------------------------------------------

def armature_items(self, context):
    items = [(o.name, o.name, "") for o in bpy.data.objects
             if o.type == 'ARMATURE']
    return items or [('', "(no armatures in scene)", "")]


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class IMPORT_OT_md5mesh(Operator, ImportHelper, ImportFileGuardMixin,
                        MD5_ImportScaleRotMixin, MaterialGenMixin):
    """Import a Doom 3 / id Tech 4 MD5 mesh file — either the standard
    text .md5mesh, or the Doom 3 BFG compiled binary cache of one
    (.bmd5mesh). The armature is always imported alongside the mesh."""
    bl_idname  = "import_scene.md5mesh"
    bl_label   = "Import MD5 Mesh (.md5mesh, .bmd5mesh)"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".md5mesh"
    filter_glob: StringProperty(default="*.md5mesh;*.bmd5mesh;*.bMD5mesh", options={'HIDDEN'})

    smooth: BoolProperty(
        name="Set mesh shading to Smooth",
        description="Enable smooth shading - equivalent to Object->Shade Smooth or setting all faces to Smooth",
        default=True,
    )
    v12_ignore_normals: BoolProperty(
        name="V12 Don't Import Normals",
        description="When importing a Version 12 mesh, ignore the stored "
                    "normal data and let Blender derive normals from geometry. "
                    "Has no effect on Version 10 files",
        default=False,
    )
    reconstruct_normals: BoolProperty(
        name="V12 Bone Local to Blender Custom Normals",
        description="When importing a Version 12 mesh, transform the stored "
                    "bone-local normals back to world space and apply them as "
                    "Blender custom split normals. "
                    "Ignored when 'V12 Don't Import Normals' is active. "
                    "Has no effect on Version 10 files",
        default=False,
    )
    merge_verts: BoolProperty(
        name="Merge Verts by Distance",
        description="If checked, after import merge all vertices with a distance "
                    "less than the selected value. Off by default: an MD5 mesh "
                    "is stored with its UV seams and hard edges already split "
                    "into separate vertices, exactly as the engine renders it, "
                    "so welding them is a change to the model rather than a "
                    "cleanup of an import artifact",
        default=False,
    )
    merge_distance: FloatProperty(
        name="Merge Distance",
        description="Maximum distance between vertices to merge",
        default=0.0001, min=0.0, max=1.0, precision=6,
    )
    load_anims_after: BoolProperty(
        name="Load anims after",
        description="When this import finishes, open the MD5 Anim import "
                    "dialog straight away, already targeting the armature "
                    "this import just built - the usual next step, since an "
                    ".md5anim can only be loaded onto an existing armature. "
                    "Cancelling that second dialog leaves the imported mesh "
                    "in place",
        default=False,
    )

    def _is_binary(self):
        return os.path.splitext(self.filepath)[1].lower() == '.bmd5mesh'

    def draw(self, context):
        layout = self.layout
        self.draw_transforms(layout)
        layout.separator()
        layout.prop(self, "smooth")
        if not self._is_binary():
            layout.separator()
            layout.prop(self, "v12_ignore_normals")
            row = layout.row()
            row.enabled = not self.v12_ignore_normals
            row.prop(self, "reconstruct_normals")
        layout.separator()
        layout.prop(self, "merge_verts")
        dist_row = layout.row()
        dist_row.enabled = self.merge_verts
        dist_row.prop(self, "merge_distance")
        self.draw_material_gen(context, layout)
        # Last, deliberately: it is not a setting for THIS import at all
        # but what happens once it is over, so it reads as the tail of
        # the dialog rather than as another mesh option.
        layout.separator()
        layout.prop(self, "load_anims_after")

    def execute(self, context):
        # Refuse an unusable selection before anything else - notably
        # before the sources gate, which would otherwise pop a Base
        # Directory dialog for a path that was never going to open (a
        # directory derives no base directory).
        _paths, status = self.guard_input_files()
        if status:
            return status
        if self._needs_sources_gate():
            self._launch_sources_gate()
            return {'CANCELLED'}
        # Snapshot before importing so the armature this import creates can
        # be identified by difference afterwards (for "Load anims after").
        # By name is not good enough: Blender uniquifies a second
        # "<stem>_MD5_Armature" to "...001", which importing the same mesh
        # twice does immediately.
        armatures_before = {o for o in bpy.data.objects if o.type == 'ARMATURE'}
        try:
            if self._is_binary():
                result, material_issues, material_report = import_bmd5mesh(self.filepath,
                                        scale=self.get_scale(),
                                        rotation=self.get_rotation(),
                                        smooth = self.smooth,
                                        import_materials=self.import_materials,
                                        material_mode=self.material_mode,
                                        material_parameters=self.material_parameters,
                                        derive_from_model=self.derive_from_model,
                                        override_base_directory=self.override_base_directory,
                                        override_source_path=self.override_source_path,
                                        save_derived_as_default=self.save_derived_as_default)
            else:
                # When v12_ignore_normals is on, force reconstruct_normals off
                reconstruct = self.reconstruct_normals and not self.v12_ignore_normals
                result, material_issues, material_report = import_md5mesh(self.filepath,
                                        scale=self.get_scale(),
                                        rotation=self.get_rotation(),
                                        reconstruct_normals=reconstruct,
                                        ignore_normals=self.v12_ignore_normals, smooth = self.smooth,
                                        import_materials=self.import_materials,
                                        material_mode=self.material_mode,
                                        material_parameters=self.material_parameters,
                                        derive_from_model=self.derive_from_model,
                                        override_base_directory=self.override_base_directory,
                                        override_source_path=self.override_source_path,
                                        save_derived_as_default=self.save_derived_as_default)
        except Exception as e:
            self.report({'ERROR'}, f"MD5 mesh import failed: {e}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}

        for issue in material_issues:
            self.report({'WARNING'}, issue)
        if material_report:
            # The report was always written and never mentioned; the
            # Report panel is default_closed and the WARNING lines above
            # are driven by summary.errors, so a clean import said
            # nothing at all. Same sentence the Materials panel's own
            # Generate Materials button ends on.
            self.report({'INFO'}, 'Materials generated - full report in '
                        'the Text Editor as "%s"' % material_report)

        if result == {'FINISHED'} and self.merge_verts:
            # Find the collection that was just created by import_md5mesh.
            # It is named after the file stem and was added to the scene
            # root collection. Identify it as the one containing the
            # newly-imported armature (named <stem>_MD5_Armature).
            import os as _os
            stem        = _os.path.splitext(_os.path.basename(self.filepath))[0]
            new_col     = bpy.data.collections.get(stem)
            mesh_objs   = []
            if new_col:
                mesh_objs = [o for o in new_col.objects if o.type == 'MESH']
            # Fall back: collect any mesh selected after import
            if not mesh_objs:
                mesh_objs = [o for o in context.selected_objects if o.type == 'MESH']
            for obj in mesh_objs:
                context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.remove_doubles(threshold=self.merge_distance)
                bpy.ops.object.mode_set(mode='OBJECT')

        if result == {'FINISHED'} and self.load_anims_after:
            self._launch_anim_import(context, armatures_before)

        return result

    def _launch_anim_import(self, context, armatures_before):
        """Open the MD5 Anim import dialog on the armature this import
        just built ("Load anims after").

        The armature is made active rather than passed as a property:
        IMPORT_OT_md5anim.invoke() sets its own target_armature from the
        active object (and its Prepend Text from that object's
        collection), so anything handed in as a keyword would just be
        overwritten a moment later. Making it active is what actually
        takes.

        Best-effort throughout — an anim dialog that fails to open must
        not turn a mesh import that already succeeded into an error, so
        the mesh import's own result is never disturbed here."""
        if bpy.app.background:
            # No display to host a file browser (Blender running with
            # -b/--background, e.g. a batch/pipeline script) —
            # fileselect_add would take the process down rather than
            # fail cleanly, the same reason IMPORT_OT_md5anim refuses to
            # raise its own overwrite popup here. The mesh import itself
            # already succeeded; a batch script wanting anims should
            # call import_scene.md5anim directly.
            self.report({'WARNING'},
                        "\"Load anims after\" needs the UI - skipped in "
                        "background mode")
            return
        new_arms = [o for o in bpy.data.objects
                    if o.type == 'ARMATURE' and o not in armatures_before]
        try:
            if new_arms:
                arm = new_arms[0]
                bpy.ops.object.select_all(action='DESELECT')
                arm.select_set(True)
                context.view_layer.objects.active = arm
            bpy.ops.import_scene.md5anim('INVOKE_DEFAULT')
        except Exception as e:
            self.report({'WARNING'},
                        f"Mesh imported, but the MD5 Anim dialog could not "
                        f"be opened: {e}")


class IMPORT_OT_md5anim(Operator, ImportHelper, ImportFileGuardMixin,
                        MD5_ImportScaleRotMixin):
    """Import one or more MD5 anim files as Actions on an existing armature —
    either standard text .md5anim, or the Doom 3 BFG compiled binary cache
    of one (.bmd5anim); both may be selected together. Select multiple
    files in the file browser; each becomes its own Action. The first file
    imported is set as the active action."""
    bl_idname  = "import_scene.md5anim"
    bl_label   = "Import MD5 Anim (.md5anim, .bmd5anim)"
    bl_options = {'REGISTER', 'UNDO'}

    # Allow multi-file selection
    filename_ext = ".md5anim"
    filter_glob: StringProperty(default="*.md5anim;*.bmd5anim;*.bMD5anim", options={'HIDDEN'})

    # directory and files are the standard Blender multi-file properties
    directory: StringProperty(subtype='DIR_PATH', options={'HIDDEN'})
    files: bpy.props.CollectionProperty(
        type=bpy.types.OperatorFileListElement,
        options={'HIDDEN', 'SKIP_SAVE'},
    )

    target_armature: EnumProperty(
        name="Target Armature",
        description="Armature to receive the imported animation actions",
        items=armature_items,
    )
    assign_fake_user: BoolProperty(
    name="Assign fake user",
    description="Assign a fake user to each created action to ensure unassigned actions will be saved with the .blend file",
    default=True,
    )
    prepend_name: BoolProperty(
        name="Prepend Action Names",
        description="Prepend each imported action name with the text below",
        default=False,
    )
    prepend_text: StringProperty(
        name="Prepend Text",
        description="Text to prepend to each imported action name",
        default="",
    )
    confirm_overwrite: BoolProperty(
        name="Confirm Overwrite",
        description="Internal flag: user has confirmed overwriting existing actions",
        default=False,
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    overwrite_list: StringProperty(
        name="",
        description="Internal: comma-separated list of actions that would be overwritten",
        default="",
        options={'HIDDEN', 'SKIP_SAVE'},
    )

    def invoke(self, context, event):
        ao = context.active_object
        if ao and ao.type == 'ARMATURE':
            self.target_armature = ao.name
        # Initialise prepend text to collection name + underscore
        try:
            col = get_collection_of_active_object(context)
            self.prepend_text = col.name + "_"
        except Exception:
            self.prepend_text = ""
        return super().invoke(context, event)

    def draw(self, context):
        layout = self.layout
        layout.label(text="Import MD5Anim(s).")
        layout.separator()
        # Label above the dropdown, matching every other enum across
        # these dialogs — an armature name is long enough that Blender's
        # own property split left the dropdown very little width.
        layout.label(text="Target Armature:")
        layout.prop(self, "target_armature", text="")
        layout.separator()
        layout.label(text="Use same Scale/Rot as mesh import.")
        layout.separator()
        self.draw_transforms(layout)
        layout.separator()
        layout.prop(self, "assign_fake_user")
        layout.separator()
        layout.prop(self, "prepend_name")
        row = layout.row()
        row.enabled = self.prepend_name
        row.label(text="Prepend Text:")
        col = layout.column()
        col.enabled = self.prepend_name
        col.prop(self, "prepend_text", text="")
        if self.confirm_overwrite and self.overwrite_list:
            layout.separator()
            box = layout.box()
            box.label(text="Warning: these actions will be overwritten:", icon='ERROR')
            for name in self.overwrite_list.split(","):
                box.label(text=f"  • {name.strip()}")
            box.label(text="Press Import again to confirm.")

    def execute(self, context):
        # File selection first, so a mis-click in the browser reports the
        # same way here as in every other importer. Unusable entries in a
        # multi-selection are dropped with a warning; only an entirely
        # unusable selection refuses.
        filepaths, status = self.guard_input_files(multi=True)
        if status:
            return status

        if not self.target_armature:
            self.report({'ERROR'}, "No target armature selected.")
            return {'CANCELLED'}

        arm_obj = bpy.data.objects.get(self.target_armature)
        if arm_obj is None or arm_obj.type != 'ARMATURE':
            self.report({'ERROR'},
                        f"'{self.target_armature}' is not a valid armature.")
            return {'CANCELLED'}

        prefix = self.prepend_text if self.prepend_name else ""

        # Check for existing actions that would be overwritten
        conflicts = [
            prefix + os.path.splitext(os.path.basename(fp))[0]
            for fp in filepaths
            if (prefix + os.path.splitext(os.path.basename(fp))[0]) in bpy.data.actions
        ]

        if conflicts and not self.confirm_overwrite:
            if bpy.app.background:
                # No display to host a confirmation popup (Blender running
                # with -b/--background, e.g. a batch/pipeline script) —
                # invoke_props_dialog would crash the process outright even
                # though context.window still exists as a placeholder, so
                # refuse instead.
                self.report({'ERROR'},
                            f"Would overwrite existing action(s): {', '.join(conflicts)}. "
                            f"Re-run with confirm_overwrite=True to proceed.")
                return {'CANCELLED'}
            # First pass: flag conflicts and re-show the dialog for confirmation
            self.confirm_overwrite = True
            self.overwrite_list    = ", ".join(conflicts)
            return context.window_manager.invoke_props_dialog(self, width=400)

        # Reset confirmation state for next use
        self.confirm_overwrite = False
        self.overwrite_list    = ""

        first_action = None
        imported     = []
        errors       = []

        for fp in filepaths:
            try:
                loader = (import_bmd5anim
                          if os.path.splitext(fp)[1].lower() == '.bmd5anim'
                          else import_md5anim)
                loader(fp, arm_obj,
                       scale=self.get_scale(),
                       rotation=self.get_rotation(),
                       action_name_prefix=prefix, afu = self.assign_fake_user)
                action_name = prefix + os.path.splitext(os.path.basename(fp))[0]
                action      = bpy.data.actions.get(action_name)
                if first_action is None and action is not None:
                    first_action = action
                imported.append(os.path.basename(fp))
            except ValueError as e:
                errors.append(f"{os.path.basename(fp)}: {e}")
            except Exception as e:
                errors.append(f"{os.path.basename(fp)}: {e}")
                import traceback; traceback.print_exc()

        # Set the first imported action as the active action on the armature
        if first_action is not None:
            if arm_obj.animation_data is None:
                arm_obj.animation_data_create()
            arm_obj.animation_data.action = first_action
            _md5_get_or_create_slot(arm_obj, first_action)
            bpy.ops.object.select_all(action='DESELECT')
            # Make the armature the active object, this way if the user immediately
            # selects a new action in the action editor it will take effect
            bpy.context.view_layer.objects.active = arm_obj
            arm_obj.select_set(True)

        if imported:
            self.report({'INFO'},
                        f"Imported {len(imported)} action(s) onto "
                        f"'{self.target_armature}': {', '.join(imported)}")
        if errors:
            for msg in errors:
                self.report({'ERROR'}, msg)

        return {'FINISHED'} if imported else {'CANCELLED'}


# ===========================================================================
# MD5Mesh EXPORTER
# ===========================================================================

# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def invert_rotation_preset(rotation):
    """
    Return the inverse rotation key so we can un-rotate data before writing.
    Export applies the *inverse* of the import rotation so the file comes out
    in the original MD5 coordinate system regardless of what the user imported.
    """
    inv = {'NONE': 'NONE', 'X_TO_Y': 'Y_TO_X', 'Y_TO_X': 'X_TO_Y', 'R180': 'R180'}
    return inv[rotation]


def mat4_to_md5_joint(mat, scale):
    """
    Decompose a 4x4 armature-space matrix into the MD5 pos + orient(x,y,z)
    representation. Translation is multiplied by the export scale factor
    (Blender units -> MD5 units).
    """
    pos    = mat.translation * scale
    orient = mat.to_quaternion()
    # MD5 stores negative-w quaternions  (w = -sqrt(1-x²-y²-z²))
    # Ensure we store the correct sign convention by negating if w > 0
    if orient.w > 0.0:
        orient.negate()
    return pos, orient


def get_collection_of_active_object(context):
    """
    Return the first collection that contains the active object.
    Raises ValueError if there is no active object or it belongs to no
    named collection (i.e. only the scene root collection).
    """
    ao = context.active_object
    if ao is None:
        raise ValueError("No active object. Click an object to make it active "
                         "before exporting.")
    for col in bpy.data.collections:
        if ao.name in col.objects:
            return col
    raise ValueError(
        f"Active object '{ao.name}' does not belong to any named collection.")


def is_object_visible(obj, collection):
    """
    Return True if the object is not hidden at the object level
    and not hidden in the viewport within its collection.
    Checks both object hide_viewport and collection hide_viewport.
    """
    if obj.hide_viewport:
        return False
    for col in bpy.data.collections:
        if obj.name in col.objects:
            if col.hide_viewport:
                return False
    return True


def to_object_mode(context):
    """Leave Edit/Pose/Sculpt Mode so an export reads finished data.

    Blender does not write an Edit Mode session's BMesh back to the Mesh
    datablock (or its edit_bones back to the Armature) until the mode is
    exited - the bmesh.update_edit_mesh() that runs as you model only
    refreshes the viewport cage - and an MD5 export reads the datablocks
    throughout: mesh_obj.data for positions, UVs and the vertex-group
    weights behind every vert line, arm_obj.data.bones for the rest pose
    the joints are built from. Exporting without tabbing out therefore
    wrote the model as it was when Edit Mode was ENTERED: no error, no
    warning.

    export_mesh() had a sharper version of it. With use_sharp_edges or
    v12, corner normals and MikkTSpace tangents come from the
    DEPSGRAPH-evaluated mesh, which DOES see the edit cage, while
    positions and weights come from the datablock, which does not - so
    adding or removing geometry in Edit Mode put the two on different
    loop counts. Exporting in Object Mode is what keeps them agreeing.

    This is the recipe Blender's own exporters use. io_scene_fbx's save()
    records the active object's mode, switches to Object Mode and
    switches back; io_scene_gltf2's save() switches without restoring.
    The mode round trip flushes every object of a multi-object edit
    session, not just the active one.

    Two costs, accepted the same way they are there: leaving and
    re-entering Edit Mode resets that session's Edit Mode undo stack, and
    bpy.ops.object.mode_set() scales with scene size - measured at 0.046s
    in a 3000-object scene, twice per export, which is nothing next to
    writing the file.

    One place this goes further than FBX: FBX guards the switch with
    mode_set.poll() and, when that fails, exports the stale data anyway.
    Here it raises instead, so the caller reports it. A refused export is
    recoverable; a silently pre-edit one is the failure this exists to
    remove.

    Returns what restore_mode() needs, or None when nothing had to change.
    """
    obj = context.view_layer.objects.active
    if obj is None or obj.mode == 'OBJECT':
        return None
    if not bpy.ops.object.mode_set.poll():
        pretty = obj.mode.replace('_', ' ').title()
        raise RuntimeError(
            "Cannot leave %s Mode automatically. Switch to Object Mode and "
            "export again - exporting from %s Mode would write the model as "
            "it was when you entered it." % (pretty, pretty))
    mode = obj.mode
    bpy.ops.object.mode_set(mode='OBJECT')
    return obj, mode


def restore_mode(context, saved):
    """Put back what to_object_mode() left. Safe to call with None."""
    if not saved:
        return
    obj, mode = saved
    context.view_layer.objects.active = obj
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode=mode)


def exports_in_object_mode(execute):
    """Run an operator's execute() in Object Mode, then restore the mode.

    A decorator rather than a `with` block inside each execute() so the
    restore covers every return path - including the early
    return {'CANCELLED'} ones - without reindenting the bodies.
    """
    @functools.wraps(execute)
    def wrapper(self, context):
        try:
            saved = to_object_mode(context)
        except RuntimeError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        try:
            return execute(self, context)
        finally:
            restore_mode(context, saved)
    return wrapper


def collect_export_objects(context):
    """
    Return (collection_name, arm_obj, mesh_objects) sourced from the
    collection that the active object belongs to.
    Only non-hidden objects are included.
    Raises ValueError with a helpful message if requirements are not met.
    """
    col  = get_collection_of_active_object(context)
    name = col.name

    all_objs  = [o for o in col.objects if is_object_visible(o, col)]
    arm_objs  = [o for o in all_objs if o.type == 'ARMATURE']
    mesh_objs = [o for o in all_objs if o.type == 'MESH']

    if not arm_objs:
        raise ValueError(
            f"No visible armature found in collection '{name}'.")
    if len(arm_objs) > 1:
        raise ValueError(
            f"More than one armature found in collection '{name}'. "
            "The collection must contain exactly one armature.")
    if not mesh_objs:
        raise ValueError(
            f"No visible mesh objects found in collection '{name}'.")

    return name, arm_objs[0], mesh_objs


# ---------------------------------------------------------------------------
# Joint (skeleton) export
# ---------------------------------------------------------------------------

EXPORT_BONE_COLLECTION = "MD5_export_bone_collection"


def export_joints(arm_obj, scale, rot_mat_inv):
    """
    Walk the armature's bones and return only those assigned to the
    'MD5_export_bone_collection' bone collection, as a list of dicts:
        { 'name', 'parent_idx', 'pos': Vector, 'orient': Quaternion }

    Positions and orientations are taken from the REST pose (matrix_local).
    parent_idx is remapped to refer only to other exported bones; if the
    immediate parent is not exported the chain is walked upward until an
    exported ancestor is found (or -1 for root).
    The inverse rotation and scale are applied so the output matches the
    original MD5 coordinate system.
    """
    arm_data = arm_obj.data

    # Find the export bone collection
    export_coll = None
    for bc in arm_data.collections:
        if bc.name == EXPORT_BONE_COLLECTION:
            export_coll = bc
            break

    if export_coll is None:
        raise ValueError(
            f"Bone collection '{EXPORT_BONE_COLLECTION}' not found on "
            f"armature '{arm_obj.name}'. All bones will be exported as "
            "a fallback — create the collection and assign bones to filter.")

    # Set of bone names that are in the export collection
    export_bone_names = {b.name for b in arm_data.bones
                         if any(bc.name == EXPORT_BONE_COLLECTION
                                for bc in b.collections)}

    # Ordered list of bones to export (preserve armature order for stable indices)
    export_bones = [b for b in arm_data.bones if b.name in export_bone_names]

    # name -> position in the *exported* list
    export_idx = {b.name: i for i, b in enumerate(export_bones)}

    def find_exported_ancestor(bone):
        """Walk up the parent chain until we find an exported bone or run out."""
        cur = bone.parent
        while cur is not None:
            if cur.name in export_idx:
                return export_idx[cur.name]
            cur = cur.parent
        return -1

    arm_world = arm_obj.matrix_world
    joints = []
    for bone in export_bones:
        # Full world-space rest matrix: armature world transform + bone local.
        # Matches the reference exporter and is consistent with the weight
        # position calculation which also uses arm_world @ matrix_local.
        world_mat = arm_world @ bone.matrix_local

        if rot_mat_inv is not None:
            world_mat = rot_mat_inv @ world_mat

        pos, orient = mat4_to_md5_joint(world_mat, scale)

        # Parent index within the exported set only
        if bone.parent is None or bone.parent.name not in export_idx:
            parent_idx = find_exported_ancestor(bone)
        else:
            parent_idx = export_idx[bone.parent.name]

        joints.append({
            'name':       bone.name,
            'parent_idx': parent_idx,
            'pos':        pos,
            'orient':     orient,
        })

    return joints


# ---------------------------------------------------------------------------
# Mesh export
# ---------------------------------------------------------------------------

def export_mesh(mesh_obj, arm_obj, joints, scale, rot_mat_inv,
               bone_influence_limit=4, v12=False, use_sharp_edges=True):
    """
    Export a single mesh object as one md5mesh 'mesh' block.

    v12=True  : appends bone-local normal + MikkTSpace tangent to each vert line,
                and includes a vertexcolor block if the mesh has a color attribute.
    use_sharp_edges=True : vertices at sharp edges (or with differing corner
                normals) are split in the exported data (non-destructive).
                Works for both v10 and v12.
    """
    import bmesh as _bmesh
    from mathutils import Vector as _Vector

    me         = mesh_obj.data
    arm_data   = arm_obj.data
    arm_world  = arm_obj.matrix_world
    mesh_world = mesh_obj.matrix_world

    # Shader from first material
    shader = ''
    if me.materials and me.materials[0]:
        shader = me.materials[0].name

    uv_layer = me.uv_layers.active

    # Map vertex-group index -> joint index in our export list
    name_to_joint_idx = {j['name']: i for i, j in enumerate(joints)}
    vg_to_joint = {}
    for vg in mesh_obj.vertex_groups:
        if vg.name in name_to_joint_idx:
            vg_to_joint[vg.index] = name_to_joint_idx[vg.name]

    # Bone world matrices and their inverses (rotation applied, no scale)
    bone_world_mats = []
    joint_inv       = []
    for j in joints:
        bm = arm_world @ arm_data.bones[j['name']].matrix_local
        if rot_mat_inv is not None:
            bm = rot_mat_inv @ bm
        bone_world_mats.append(bm)
        joint_inv.append(bm.inverted())

    MIN_BIAS = 0.001

    def get_influences(vi):
        raw, total = [], 0.0
        for vge in me.vertices[vi].groups:
            if vge.group in vg_to_joint and vge.weight >= MIN_BIAS:
                raw.append((vg_to_joint[vge.group], vge.weight))
                total += vge.weight
        if not raw:
            return [(0, 1.0)]
        raw.sort(key=lambda x: x[1], reverse=True)
        raw = raw[:bone_influence_limit]
        total = sum(w for _, w in raw)
        return [(ji, w / total) for ji, w in raw]

    def vert_world_pos(vi):
        p = mesh_world @ me.vertices[vi].co
        if rot_mat_inv is not None:
            p = rot_mat_inv @ p
        return p

    # ── Evaluated mesh for corner normals and tangents ────────────────────────
    # Obtained non-destructively via depsgraph; freed immediately after use.
    eval_me = None
    corner_normals = None  # [loop_idx] -> Vector (world space)
    tangents       = None  # [loop_idx] -> (tx, ty, tz, tw)

    if use_sharp_edges or v12:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj  = mesh_obj.evaluated_get(depsgraph)
        eval_me   = eval_obj.to_mesh()
        # Build corner normals in world space
        corner_normals = {}
        normal_mat = mesh_world.to_3x3().inverted().transposed()
        for loop in eval_me.loops:
            n = normal_mat @ loop.normal
            if rot_mat_inv is not None:
                n = rot_mat_inv.to_3x3() @ n
            corner_normals[loop.index] = n.normalized()

        if v12:
            # MikkTSpace tangents — requires UV layer
            if eval_me.uv_layers.active:
                eval_me.calc_tangents()
                tangents = {}
                for loop in eval_me.loops:
                    t  = normal_mat @ _Vector(loop.tangent)
                    if rot_mat_inv is not None:
                        t = rot_mat_inv.to_3x3() @ t
                    tangents[loop.index] = (
                        t.normalized().x, t.normalized().y, t.normalized().z,
                        loop.bitangent_sign,
                    )
            else:
                tangents = None

        eval_obj.to_mesh_clear()
        eval_me = None

    # ── Vertex color attribute ────────────────────────────────────────────────
    # Look for a color attribute with domain CORNER or POINT
    vc_data = None
    if v12:
        for attr in me.color_attributes:
            if attr.domain in ('CORNER', 'POINT'):
                vc_data = attr.data
                vc_domain = attr.domain
                break

    # ── Build verts / weights ─────────────────────────────────────────────────
    UV_ROUND        = 6
    # Key includes loop_idx when sharp-edge splitting is active so that the
    # same geometry vertex with the same UV but different corner normals maps
    # to distinct MD5 verts.
    key_to_md5vert  = {}
    loop_to_md5vert = {}
    out_verts   = []
    out_weights = []

    for poly in me.polygons:
        for li, vi in zip(poly.loop_indices, poly.vertices):
            if uv_layer:
                u, v_raw = uv_layer.data[li].uv
                uv = (round(u, UV_ROUND), round(1.0 - v_raw, UV_ROUND))
            else:
                uv = (0.0, 0.0)

            # Split key: include loop_idx when sharp splitting is on so
            # differing corner normals yield separate MD5 verts.
            if use_sharp_edges and corner_normals:
                key = (vi, uv[0], uv[1], li)
            else:
                key = (vi, uv[0], uv[1])

            if key not in key_to_md5vert:
                infl    = get_influences(vi)
                vw      = vert_world_pos(vi)
                w_start = len(out_weights)

                for joint_idx, bias in infl:
                    result = joint_inv[joint_idx] @ vw.to_4d()
                    pos    = result.to_3d() * scale
                    out_weights.append({
                        'joint': joint_idx,
                        'bias':  bias,
                        'pos':   pos,
                    })

                # Normal: per-loop (split) or per-vertex smooth
                if corner_normals and li in corner_normals:
                    normal_world = corner_normals[li]
                else:
                    n = mesh_world.to_3x3() @ me.vertices[vi].normal
                    if rot_mat_inv is not None:
                        n = rot_mat_inv.to_3x3() @ n
                    normal_world = n.normalized()

                # Transform normal to dominant-bone-local space (v12)
                if v12:
                    dominant_ji = infl[0][0]
                    inv3 = bone_world_mats[dominant_ji].inverted().to_3x3()
                    nl   = (inv3 @ normal_world).normalized()
                    normal_local = (nl.x, nl.y, nl.z)

                    # Tangent in dominant-bone-local space
                    if tangents and li in tangents:
                        tx, ty, tz, tw = tangents[li]
                        t_world = _Vector((tx, ty, tz))
                        tl = (inv3 @ t_world).normalized()
                        tangent_local = (tl.x, tl.y, tl.z, tw)
                    else:
                        tangent_local = (1.0, 0.0, 0.0, 1.0)
                else:
                    normal_local  = None
                    tangent_local = None

                key_to_md5vert[key] = len(out_verts)
                out_verts.append({
                    'uv':     uv,
                    'start':  w_start,
                    'count':  len(infl),
                    'normal': normal_local,
                    'tangent':tangent_local,
                    'loop_vi':vi,       # original vertex index (for vert colors)
                    'loop_li':li,
                })

            loop_to_md5vert[li] = key_to_md5vert[key]

    # Triangulate (fan), reversing winding
    out_tris = []
    for poly in me.polygons:
        loops = list(poly.loop_indices)
        m0 = loop_to_md5vert[loops[0]]
        for k in range(1, len(loops) - 1):
            m1 = loop_to_md5vert[loops[k]]
            m2 = loop_to_md5vert[loops[k + 1]]
            out_tris.append((m0, m2, m1))

    # Vertex colors (v12 only)
    out_colors = None
    if v12 and vc_data is not None:
        out_colors = []
        for v in out_verts:
            idx = v['loop_li'] if vc_domain == 'CORNER' else v['loop_vi']
            c   = vc_data[idx].color
            out_colors.append((c[0], c[1], c[2], c[3]))

    print(f"[MD5 Export] {mesh_obj.name}: "
          f"{len(out_verts)} verts, {len(out_tris)} tris, {len(out_weights)} weights"
          + (f", {len(out_colors)} colors" if out_colors else ""))

    return {
        'shader':  shader,
        'verts':   out_verts,
        'tris':    out_tris,
        'weights': out_weights,
        'colors':  out_colors,
    }

def _fmt(v):
    """Format a float with 10 decimal places, collapsing any value that rounds
    to zero at that precision to the bare token '0' (no decimal, no sign)."""
    if not v:  # fast path: exact -0.0 or 0.0
        return "0"
    s = f"{v:.10f}"
    # If every digit after the decimal is zero the value is effectively zero
    if float(s) == 0.0:
        return "0"
    return s


def write_md5mesh(filepath, joints, meshes, v12=False):
    """Write joints and mesh blocks to an .md5mesh file.

    v12=True  : writes MD5Version 12 and extended vert lines with
                bone-local normals/tangents plus optional vertexcolor block.
    v12=False : writes standard MD5Version 10 format.
    """
    md5_version = 12 if v12 else 10

    # Blender and plugin version strings for the commandline tag
    blender_ver = ".".join(str(x) for x in bpy.app.version[:3])
    plugin_ver  = ".".join(str(x) for x in bl_info["version"])
    cmdline     = (f"Exported by Blender {blender_ver} "
                   f"idTech Tools {plugin_ver}")

    lines = []
    lines.append(f'MD5Version {md5_version}')
    lines.append(f'commandline "{cmdline}"')
    lines.append('')
    lines.append(f'numJoints {len(joints)}')
    lines.append(f'numMeshes {len(meshes)}')
    lines.append('')

    # joints block
    lines.append('joints {')
    for j in joints:
        p = j['pos']
        o = j['orient']
        lines.append(
            f'\t"{j["name"]}"\t{j["parent_idx"]} '
            f'( {_fmt(p.x)} {_fmt(p.y)} {_fmt(p.z)} ) '
            f'( {_fmt(o.x)} {_fmt(o.y)} {_fmt(o.z)} )'
        )
    lines.append('}')
    lines.append('')

    # mesh blocks
    for mesh in meshes:
        lines.append('mesh {')
        lines.append(f'\tshader "{mesh["shader"]}"')
        lines.append('')

        verts   = mesh['verts']
        tris    = mesh['tris']
        weights = mesh['weights']
        colors  = mesh.get('colors')

        lines.append(f'\tnumverts {len(verts)}')
        for vi, v in enumerate(verts):
            u, t = v['uv']
            base = f'\tvert {vi} ( {_fmt(u)} {_fmt(t)} ) {v["start"]} {v["count"]}'
            if v12 and v.get('normal') is not None:
                nx, ny, nz = v['normal']
                tx, ty, tz, tw = v['tangent'] if v.get('tangent') else (1.0, 0.0, 0.0, 1.0)
                base += (f' ( {_fmt(nx)} {_fmt(ny)} {_fmt(nz)} )'
                         f' ( {_fmt(tx)} {_fmt(ty)} {_fmt(tz)} {_fmt(tw)} )')
            lines.append(base)
        lines.append('')

        lines.append(f'\tnumtris {len(tris)}')
        for ti, tri in enumerate(tris):
            lines.append(f'\ttri {ti} {tri[0]} {tri[1]} {tri[2]}')
        lines.append('')

        # Sanity-check weight ordering
        expected = 0
        for vi, v in enumerate(verts):
            assert v['start'] == expected, (
                f"Weight ordering broken at vert {vi}: "
                f"start={v['start']} expected={expected}")
            expected += v['count']

        lines.append(f'\tnumweights {len(weights)}')
        for wi, w in enumerate(weights):
            wp = w['pos']
            lines.append(
                f'\tweight {wi} {w["joint"]} {_fmt(w["bias"])} '
                f'( {_fmt(wp[0])} {_fmt(wp[1])} {_fmt(wp[2])} )'
            )

        # Optional vertex color block (v12 only)
        if v12 and colors:
            lines.append('')
            lines.append(f'\tnumvertexcolors {len(colors)}')
            for ci, (r, g, b, a) in enumerate(colors):
                lines.append(
                    f'\tvertexcolor {ci} ( {_fmt(r)} {_fmt(g)} {_fmt(b)} {_fmt(a)} )'
                )

        lines.append('}')
        lines.append('')

    text = '\n'.join(lines)
    import re as _re
    text = _re.sub(r'(?<![0-9.])-?0\.0{1,10}(?![0-9])', '0', text)
    with open(filepath, 'w', encoding='utf-8') as fh:
        fh.write(text)
    print(f"[MD5 Export] Written: {filepath}")


def _poll_has_armature(context):
    """Return True if the active collection contains at least one armature."""
    try:
        col = get_collection_of_active_object(context)
    except ValueError:
        return False
    return any(o.type == 'ARMATURE' for o in col.objects)


def _poll_has_armature_with_actions(context):
    """Return True if the active collection contains an armature and at least
    one action exists in the scene."""
    return _poll_has_armature(context) and bool(bpy.data.actions)


# ---------------------------------------------------------------------------
# MD5Mesh export entry point
# ---------------------------------------------------------------------------

def export_md5mesh(filepath, context, scale=1.0, rotation='NONE',
                   bone_influence_limit=4, v12=False, use_sharp_edges=True):
    rot_mat_inv = ROTATION_PRESETS[rotation]

    col_name, arm_obj, mesh_objs = collect_export_objects(context)

    joints = export_joints(arm_obj, scale, rot_mat_inv)
    meshes = []
    for mo in mesh_objs:
        meshes.append(export_mesh(mo, arm_obj, joints, scale, rot_mat_inv,
                                  bone_influence_limit=bone_influence_limit,
                                  v12=v12, use_sharp_edges=use_sharp_edges))

    write_md5mesh(filepath, joints, meshes, v12=v12)
    print(f"[MD5 Export] Done — '{col_name}' — "
          f"{len(joints)} joints, {len(meshes)} meshes"
          + (" (MD5v12)" if v12 else "") + ".")
    return {'FINISHED'}


# ---------------------------------------------------------------------------
# Export operator
# ---------------------------------------------------------------------------

class EXPORT_OT_md5mesh(Operator, ExportHelper, MD5_ExportScaleRotMixin):
    """Export selected armature and meshes as a Doom 3 / id Tech 4 MD5 mesh file.
    Select the armature and all mesh objects you want to include, then export.
    The rest pose is used for the skeleton and vertex weights."""
    bl_idname  = "export_scene.md5mesh"
    bl_label   = "Export MD5 Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".md5mesh"
    filter_glob: StringProperty(default="*.md5mesh", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return _poll_has_armature(context)

    bone_influence_limit: IntProperty(
        name="Bone Influence Limit",
        description="Maximum number of bones that can influence each vertex",
        default=4, min=1, max=8,
    )
    use_v12: BoolProperty(
        name="MD5 Version 12",
        description="Write MD5Version 12 with per-vertex normals, MikkTSpace "
                    "tangents, and optional vertex colors. When unchecked, "
                    "writes standard v10 format",
        default=False,
    )
    use_sharp_edges: BoolProperty(
        name="Split Sharp Edges",
        description="Split vertices at edges marked sharp (or with differing "
                    "corner normals). Non-destructive — the Blender scene is "
                    "never modified. Works for both v10 and v12",
        default=False,
    )

    def invoke(self, context, event):
        # Pre-populate the filename with the active object's collection name
        try:
            col = get_collection_of_active_object(context)
            self.filepath = col.name + self.filename_ext
        except ValueError:
            pass  # Leave filepath as-is if no active object/collection
        return super().invoke(context, event)

    def draw(self, context):
        layout = self.layout
        layout.label(text="If Rotation was applied on import,", icon='INFO')
        layout.label(text="select inverse to restore MD5 space.")
        self.draw_transforms(layout)
        layout.separator()
        layout.prop(self, "bone_influence_limit")
        layout.prop(self, "use_v12")
        layout.prop(self, "use_sharp_edges")

    @exports_in_object_mode
    def execute(self, context):
        try:
            return export_md5mesh(self.filepath, context,
                                  scale=self.get_scale(),
                                  rotation=self.get_rotation(),
                                  bone_influence_limit=self.bone_influence_limit,
                                  v12=self.use_v12,
                                  use_sharp_edges=self.use_sharp_edges)
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"MD5 mesh export failed: {e}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}


# ===========================================================================
# MD5Anim EXPORTER
# ===========================================================================

def get_armature_from_collection(context):
    """
    Return the single armature from the collection the active object belongs
    to.  Raises ValueError if none or more than one armature is found.
    """
    col_name, arm_obj, _meshes = collect_export_objects(context)
    return col_name, arm_obj


def get_export_bone_names(arm_obj):
    """Return ordered list of bone names in EXPORT_BONE_COLLECTION."""
    arm_data = arm_obj.data
    if not any(bc.name == EXPORT_BONE_COLLECTION for bc in arm_data.collections):
        raise ValueError(
            f"Bone collection '{EXPORT_BONE_COLLECTION}' not found on "
            f"armature '{arm_obj.name}'.")
    return [
        b.name for b in arm_data.bones
        if any(bc.name == EXPORT_BONE_COLLECTION for bc in b.collections)
    ]


def eval_pose_matrices(arm_obj, action, frame_start, frame_end, rot_mat_inv, scale):
    """
    Convert a Blender action into per-frame MD5 (pos, orient_xyz) values.

    Mirrors the reference exporter approach:
      - Read pose_bone.matrix (armature space) each frame via scene.frame_set.
      - Root joints:  md5_mat = arm_world @ pose_bone.matrix
      - Child joints: md5_mat = parent_pose_matrix.inverted() @ pose_bone.matrix
        (expresses child relative to parent, which is what MD5 stores)
      - Then un-apply import-time rotation and scale.

    This is the exact inverse of build_frame_skeleton's child formula:
        final_pos = par_orient @ local_pos + par_pos
        final_orient = par_orient @ local_orient
    """
    export_bone_names = get_export_bone_names(arm_obj)
    arm_data  = arm_obj.data
    scene     = bpy.context.scene
    w_matrix  = arm_obj.matrix_world

    # Parent name within export set
    name_to_idx = {n: i for i, n in enumerate(export_bone_names)}

    def exported_parent(bname):
        cur = arm_data.bones[bname].parent
        while cur is not None:
            if cur.name in name_to_idx:
                return cur.name
            cur = cur.parent
        return None

    parent_of = {n: exported_parent(n) for n in export_bone_names}

    if arm_obj.animation_data is None:
        arm_obj.animation_data_create()
    prev_action = arm_obj.animation_data.action
    prev_slot   = getattr(arm_obj.animation_data, "action_slot", None)
    prev_frame  = scene.frame_current
    arm_obj.animation_data.action = action
    _md5_get_or_create_slot(arm_obj, action)

    all_frames           = []
    all_world_positions  = []   # [frame][bone] -> Vector (world space)
    for fi in range(int(frame_start), int(frame_end) + 1):
        scene.frame_set(fi)
        bpy.context.view_layer.update()

        pose = arm_obj.pose
        frame_mats        = []
        frame_world_positions = []   # world-space bone head positions for bounds

        for bname in export_bone_names:
            pb      = pose.bones[bname]
            parname = parent_of[bname]

            # World-space position for bounds (always arm_world @ pose_matrix)
            world_pos = (w_matrix @ pb.matrix).translation.copy()
            frame_world_positions.append(world_pos)

            if parname is None:
                # Root: convert from armature space to world space
                md5_mat = w_matrix @ pb.matrix
            else:
                # Child: express relative to parent pose matrix
                # (parent_pose_mat.inv @ child_pose_mat gives local-to-parent)
                md5_mat = pose.bones[parname].matrix.inverted() @ pb.matrix

            # Un-apply import rotation and scale
            pos    = md5_mat.translation.copy()
            orient = md5_mat.to_3x3().normalized().to_quaternion()

            if rot_mat_inv is not None:
                pos    = rot_mat_inv.to_3x3() @ pos
                orient = rot_mat_inv.to_quaternion() @ orient

            pos *= scale

            mat             = orient.to_matrix().to_4x4()
            mat.translation = pos
            frame_mats.append(mat)

        all_frames.append(frame_mats)
        all_world_positions.append(frame_world_positions)

    arm_obj.animation_data.action = prev_action
    if prev_action is not None and prev_slot is not None:
        try:
            arm_obj.animation_data.action_slot = prev_slot
        except Exception:
            pass
    scene.frame_set(prev_frame)
    bpy.context.view_layer.update()

    return export_bone_names, all_frames, all_world_positions

def build_hierarchy_for_export(arm_obj, export_bone_names,
                               all_frame_mats=None, compress=False,
                               delta_threshold=0.0001):
    """
    Build the md5anim hierarchy list.

    When compress=False: flags=63 for every joint (all 6 components).
    The export operator defaults compress to True.
    When compress=True: for each joint, compare every frame's tx/ty/tz/qx/qy/qz
    against frame-0 values.  A component gets a flag bit only if it moves more
    than delta_threshold across any frame.  start_index is the cumulative count
    of flagged components for all preceding joints.

    Flag bits:
        1  = Tx,  2  = Ty,  4  = Tz
        8  = Qx,  16 = Qy,  32 = Qz
    """
    arm_data    = arm_obj.data
    name_to_exp = {n: i for i, n in enumerate(export_bone_names)}

    def find_exported_ancestor(bone):
        cur = bone.parent
        while cur is not None:
            if cur.name in name_to_exp:
                return name_to_exp[cur.name]
            cur = cur.parent
        return -1

    # Pre-compute per-joint per-frame (pos, orient) tuples if compressing
    if compress and all_frame_mats:
        # all_frame_mats[frame][bone_idx] = Matrix
        joint_frames = []  # [bone_idx] -> list of (pos, (qx,qy,qz)) per frame
        for bi in range(len(export_bone_names)):
            frames = []
            for frame_mats in all_frame_mats:
                pos, orient = mat_to_pos_orient(frame_mats[bi])
                frames.append((pos, orient))
            joint_frames.append(frames)

    hierarchy  = []
    start_idx  = 0
    for i, bname in enumerate(export_bone_names):
        bone = arm_data.bones[bname]
        if bone.parent is None or bone.parent.name not in name_to_exp:
            parent_idx = find_exported_ancestor(bone)
        else:
            parent_idx = name_to_exp[bone.parent.name]

        if compress and all_frame_mats:
            base_pos, base_ori = joint_frames[i][0]
            flags = 0
            for pos, ori in joint_frames[i][1:]:
                if abs(pos.x - base_pos.x) > delta_threshold: flags |= 1
                if abs(pos.y - base_pos.y) > delta_threshold: flags |= 2
                if abs(pos.z - base_pos.z) > delta_threshold: flags |= 4
                if abs(ori[0] - base_ori[0]) > delta_threshold: flags |= 8
                if abs(ori[1] - base_ori[1]) > delta_threshold: flags |= 16
                if abs(ori[2] - base_ori[2]) > delta_threshold: flags |= 32
        else:
            flags = 63

        num_components = bin(flags).count('1')
        hierarchy.append({
            'name':        bname,
            'parent':      parent_idx,
            'flags':       flags,
            # Per spec: if no components are animated (flags==0) startIndex is 0
            'start_index': start_idx if num_components > 0 else 0,
        })
        start_idx += num_components

    return hierarchy


def mat_to_pos_orient(mat):
    """Decompose a 4x4 matrix into (pos Vector, orient xyz tuple)."""
    pos    = mat.translation.copy()
    orient = mat.to_quaternion()
    if orient.w > 0.0:
        orient.negate()                  # MD5 sign convention: w <= 0
    return pos, (orient.x, orient.y, orient.z)


def write_md5anim(filepath, action_name, frame_rate,
                  hierarchy, baseframe_mats, all_frame_mats,
                  all_world_positions=None, bounds_scale=1.0):
    """
    Write a .md5anim file.

    baseframe_mats      : list of 4x4 Matrix  (one per export bone, first frame)
    all_frame_mats      : list of lists of 4x4 Matrix  ([frame][bone])
    all_world_positions : list of lists of Vector ([frame][bone], world space)
                          used for accurate per-frame bounds computation.
    """
    num_frames = len(all_frame_mats)
    num_joints = len(hierarchy)
    num_animated_components = sum(bin(h['flags']).count('1') for h in hierarchy)

    lines = []
    blender_ver = ".".join(str(x) for x in bpy.app.version[:3])
    plugin_ver  = ".".join(str(x) for x in bl_info["version"])
    cmdline     = (f"Exported by Blender {blender_ver} "
                   f"idTech Tools {plugin_ver}")
    lines.append('MD5Version 10')
    lines.append(f'commandline "{cmdline}"')
    lines.append('')
    lines.append(f'numFrames {num_frames}')
    lines.append(f'numJoints {num_joints}')
    lines.append(f'frameRate {frame_rate}')
    lines.append(f'numAnimatedComponents {num_animated_components}')
    lines.append('')

    # hierarchy block
    lines.append('hierarchy {')
    for h in hierarchy:
        lines.append(
            f'\t"{h["name"]}"\t{h["parent"]} {h["flags"]} {h["start_index"]}')
    lines.append('}')
    lines.append('')

    # bounds block — use world-space bone positions for accurate AABBs.
    # all_world_positions[frame][bone] are arm_world @ pose_bone.matrix
    # positions, which correctly reflect the full skeleton hierarchy.
    lines.append('bounds {')
    for fi, frame_mats in enumerate(all_frame_mats):
        if all_world_positions and fi < len(all_world_positions):
            positions = all_world_positions[fi]
        else:
            # Fallback: use MD5-space translations (less accurate)
            positions = [m.translation for m in frame_mats]
        # Compute the raw AABB from bone positions.
        raw_min_x = min(p.x for p in positions)
        raw_min_y = min(p.y for p in positions)
        raw_min_z = min(p.z for p in positions)
        raw_max_x = max(p.x for p in positions)
        raw_max_y = max(p.y for p in positions)
        raw_max_z = max(p.z for p in positions)
        # Scale the bounding box by expanding/contracting each side
        # equally around the box centre, so the centre stays fixed.
        cx = (raw_min_x + raw_max_x) * 0.5
        cy = (raw_min_y + raw_max_y) * 0.5
        cz = (raw_min_z + raw_max_z) * 0.5
        hx = (raw_max_x - raw_min_x) * 0.5 * bounds_scale
        hy = (raw_max_y - raw_min_y) * 0.5 * bounds_scale
        hz = (raw_max_z - raw_min_z) * 0.5 * bounds_scale
        min_x = cx - hx;  max_x = cx + hx
        min_y = cy - hy;  max_y = cy + hy
        min_z = cz - hz;  max_z = cz + hz
        lines.append(
            f'\t( {_fmt(min_x)} {_fmt(min_y)} {_fmt(min_z)} ) ' +
             f'( {_fmt(max_x)} {_fmt(max_y)} {_fmt(max_z)} )')
    lines.append('}')
    lines.append('')

    # baseframe block  (first frame poses)
    lines.append('baseframe {')
    for mat in baseframe_mats:
        pos, orient = mat_to_pos_orient(mat)
        lines.append(
            f'\t( {_fmt(pos.x)} {_fmt(pos.y)} {_fmt(pos.z)} ) ' +
            f'( {_fmt(orient[0])} {_fmt(orient[1])} {_fmt(orient[2])} )')
    lines.append('}')
    lines.append('')

    # frame blocks — only emit components whose flag bit is set
    COMPONENT_ORDER = [
        (1,  lambda p, o: p.x),
        (2,  lambda p, o: p.y),
        (4,  lambda p, o: p.z),
        (8,  lambda p, o: o[0]),
        (16, lambda p, o: o[1]),
        (32, lambda p, o: o[2]),
    ]
    for fi, frame_mats in enumerate(all_frame_mats):
        lines.append(f'frame {fi} {{')
        for bi, mat in enumerate(frame_mats):
            pos, orient = mat_to_pos_orient(mat)
            flags = hierarchy[bi]['flags']
            vals  = [getter(pos, orient)
                     for bit, getter in COMPONENT_ORDER
                     if flags & bit]
            if vals:
                lines.append('\t' + ' '.join(_fmt(v) for v in vals))
        lines.append('}')
        lines.append('')

    text = '\n'.join(lines)
    import re as _re
    text = _re.sub(r'(?<![0-9.])-?0\.0{1,10}(?![0-9])', '0', text)
    with open(filepath, 'w', encoding='utf-8') as fh:
        fh.write(text)
    print(f"[MD5 Export] Anim written: {filepath}")


def export_single_action(context, action, directory, scale, rot_mat_inv,
                         compress=False, delta_threshold=0.0001,
                         bounds_scale=1.0, filename_stem=None):
    """Export one Blender action as a .md5anim file.
    filename_stem overrides the output filename stem (defaults to action.name)."""
    _col_name, arm_obj = get_armature_from_collection(context)

    frame_start = action.frame_range[0]
    frame_end   = action.frame_range[1]
    frame_rate  = bpy.context.scene.render.fps

    export_bone_names, all_frame_mats, all_world_positions = eval_pose_matrices(
        arm_obj, action, frame_start, frame_end, rot_mat_inv, scale)

    hierarchy      = build_hierarchy_for_export(
        arm_obj, export_bone_names,
        all_frame_mats=all_frame_mats,
        compress=compress,
        delta_threshold=delta_threshold)
    baseframe_mats = all_frame_mats[0]

    stem     = filename_stem if filename_stem is not None else action.name
    filename = stem + '.md5anim'
    filepath = os.path.join(directory, filename)
    write_md5anim(filepath, action.name, frame_rate,
                  hierarchy, baseframe_mats, all_frame_mats,
                  all_world_positions=all_world_positions,
                  bounds_scale=bounds_scale)
    return filepath


def export_md5anim(filepath, context, export_all=False,
                   scale=1.0, rotation='NONE',
                   compress=False, delta_threshold=0.0001,
                   bounds_scale=1.0,
                   action_names=None,
                   filename_stem_transform=None):
    """
    Export animation(s) from the armature in the active object's collection.

    filepath               : chosen path from the file browser (directory + filename)
    action_names           : if provided, a list/set of action names to export.
                             Overrides export_all when present.
    export_all             : if True (and action_names is None), export every action.
    filename_stem_transform: optional callable(stem) -> stem applied to each filename.
    """
    rot_mat_inv = ROTATION_PRESETS[rotation]
    directory   = os.path.dirname(filepath)

    _col_name, arm_obj = get_armature_from_collection(context)

    if action_names is not None:
        # Export exactly the named actions; empty list = nothing to do
        if len(action_names) == 0:
            raise ValueError("No actions selected for export.")
        actions_to_export = [a for a in bpy.data.actions if a.name in set(action_names)]
        if not actions_to_export:
            raise ValueError("None of the selected actions were found.")
    elif export_all:
        actions_to_export = []
        for action in bpy.data.actions:
            for fc in _md5_iter_action_fcurves(action):
                if 'pose.bones[' in fc.data_path:
                    actions_to_export.append(action)
                    break
        if not actions_to_export:
            raise ValueError("No actions found in this blend file.")
    else:
        if (arm_obj.animation_data is None or
                arm_obj.animation_data.action is None):
            raise ValueError(
                f"Armature '{arm_obj.name}' has no active action.")
        actions_to_export = [arm_obj.animation_data.action]

    exported = []
    for action in actions_to_export:
        stem = filename_stem_transform(action.name) if filename_stem_transform else None
        fp = export_single_action(context, action, directory, scale, rot_mat_inv,
                                  compress=compress, delta_threshold=delta_threshold,
                                  bounds_scale=bounds_scale, filename_stem=stem)
        exported.append(fp)

    print(f"[MD5 Export] Exported {len(exported)} anim(s).")
    return {'FINISHED'}


def _make_stem_transform(change_prepend, prepend_mode, prepend_text):
    """Return a stem->stem callable based on the operator's prepend settings,
    or None if no transform is needed."""
    if not change_prepend:
        return None
    text = prepend_text.strip()
    if not text:
        return None
    if prepend_mode == 'ADD':
        return lambda stem: text + stem
    if prepend_mode == 'REMOVE':
        def _remove(stem):
            if stem.startswith(text) and len(stem) > len(text):
                return stem[len(text):]
            return stem
        return _remove
    return None


# ---------------------------------------------------------------------------
# Anim export operator
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Action list infrastructure — PropertyGroup + UIList + All/None operators
# ---------------------------------------------------------------------------

class MD5_ActionItem(bpy.types.PropertyGroup):
    """One row in the action export checklist."""
    # name is inherited from PropertyGroup
    selected: BoolProperty(name="", default=True)


class MD5_UL_actions(bpy.types.UIList):
    """Scrollable action checklist for export dialogs."""
    bl_idname = "MD5_UL_actions"

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_propname, index):
        layout.prop(item, "selected", text=item.name)


class EXPORT_OT_md5_actions_all(bpy.types.Operator):
    """Select all actions"""
    bl_idname = "export_scene.md5_actions_all"
    bl_label  = "All"
    def execute(self, context):
        for it in context.scene.md5_action_items:
            it.selected = True
        return {'FINISHED'}


class EXPORT_OT_md5_actions_none(bpy.types.Operator):
    """Deselect all actions"""
    bl_idname = "export_scene.md5_actions_none"
    bl_label  = "None"
    def execute(self, context):
        for it in context.scene.md5_action_items:
            it.selected = False
        return {'FINISHED'}


class EXPORT_OT_md5_actions_invert(bpy.types.Operator):
    """Invert the current action selection"""
    bl_idname = "export_scene.md5_actions_invert"
    bl_label  = "Invert"
    def execute(self, context):
        for it in context.scene.md5_action_items:
            it.selected = not it.selected
        return {'FINISHED'}


def _populate_action_list(context, active_action_name="", select_all=False):
    """Rebuild scene.md5_action_items from bpy.data.actions.
    The active action is placed first. When select_all=True every item is
    selected; otherwise only the active action is pre-selected (or all if
    there is no active action)."""
    scene = context.scene
    scene.md5_action_items.clear()
    scene.md5_action_list_index = 0
    actions = list(bpy.data.actions)
    if active_action_name:
        actions.sort(key=lambda a: (0 if a.name == active_action_name else 1, a.name))
    for action in actions:
        item = scene.md5_action_items.add()
        item.name     = action.name
        item.selected = True if select_all else (action.name == active_action_name)
    # If nothing ended up selected, select all
    if not any(it.selected for it in scene.md5_action_items):
        for it in scene.md5_action_items:
            it.selected = True


def _draw_action_list(layout, context, rows=8):
    """Draw All/None/Invert buttons + scrollable UIList into layout."""
    scene = context.scene
    row = layout.row(align=True)
    row.operator("export_scene.md5_actions_all",    text="All",    icon='CHECKBOX_HLT')
    row.operator("export_scene.md5_actions_none",   text="None",   icon='CHECKBOX_DEHLT')
    row.operator("export_scene.md5_actions_invert", text="Invert", icon='ARROW_LEFTRIGHT')
    layout.template_list(
        "MD5_UL_actions", "",
        scene, "md5_action_items",
        scene, "md5_action_list_index",
        rows=rows,
    )


def _get_selected_actions(context):
    """Return list of action names that are checked."""
    return [it.name for it in context.scene.md5_action_items if it.selected]


# ---------------------------------------------------------------------------
# Anim export operator
# ---------------------------------------------------------------------------

class EXPORT_OT_md5anim(Operator, ExportHelper, MD5_ExportScaleRotMixin):
    """Export armature animation(s) as Doom 3 / id Tech 4 MD5 anim file(s)."""
    bl_idname  = "export_scene.md5anim"
    bl_label   = "Export MD5 Anim"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".md5anim"
    filter_glob: StringProperty(default="*.md5anim", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return _poll_has_armature_with_actions(context)

    compress_anim: BoolProperty(
        name="Compress Animation",
        default=True,
        description="Only export components that move beyond the delta threshold. "
                    "Static components are stored in the baseframe only, "
                    "reducing file size",
    )
    delta_threshold: FloatProperty(
        name="Delta Threshold",
        description="When compressing: minimum change in a translation or rotation component "
                    "required for it to be considered animated",
        default=0.0001, min=0.0, max=1.0, precision=6,
    )
    bounds_scale: FloatProperty(
        name="Bounds Scale",
        description="Scale the bounding box for each frame by this factor, "
                    "expanding or contracting it around its centre. "
                    "Use to correct bounds that appear too small or too large in-engine",
        default=1.0, min=0.0, max=1000.0, precision=4,
    )
    change_prepend: BoolProperty(
        name="Change Prepend",
        description="Add or remove a prefix from each exported animation filename",
        default=False,
    )
    prepend_mode: EnumProperty(
        name="Mode",
        description="Whether to add or remove the prepend text",
        items=[
            ('ADD',    "Add",    "Prepend the text to the filename stem"),
            ('REMOVE', "Remove", "Strip the text from the start of the filename stem"),
        ],
        default='ADD',
    )
    prepend_text: StringProperty(
        name="Prepend Text",
        description="Text to add to or remove from exported animation filenames",
        default="",
    )

    def invoke(self, context, event):
        ao = context.active_object
        active_name = ""
        try:
            if ao and ao.animation_data and ao.animation_data.action:
                active_name     = ao.animation_data.action.name
                self.filepath   = active_name + self.filename_ext
            else:
                col = get_collection_of_active_object(context)
                self.filepath = col.name + self.filename_ext
        except ValueError:
            pass
        try:
            col = get_collection_of_active_object(context)
            self.prepend_text = col.name + "_"
        except Exception:
            pass
        _populate_action_list(context, active_name)
        return super().invoke(context, event)

    def draw(self, context):
        layout = self.layout
        layout.label(text="Scale and Rotation should match mesh export.", icon='INFO')
        self.draw_transforms(layout)
        layout.prop(self, "compress_anim")
        row = layout.row()
        row.enabled = self.compress_anim
        row.prop(self, "delta_threshold")
        layout.prop(self, "bounds_scale")
        selected_count = sum(1 for it in context.scene.md5_action_items if it.selected)
        layout.label(text=f"Selected Export Actions: {selected_count}")
        _draw_action_list(layout, context, rows=6)
        layout.separator()
        col2 = layout.column(align=True)
        col2.prop(self, "change_prepend")
        mode_row = col2.row(align=True)
        mode_row.enabled = self.change_prepend
        mode_row.prop(self, "prepend_mode", expand=True)
        txt_row = col2.row()
        txt_row.enabled = self.change_prepend
        txt_row.prop(self, "prepend_text", text="")

    @exports_in_object_mode
    def execute(self, context):
        names = _get_selected_actions(context)
        transform = _make_stem_transform(self.change_prepend, self.prepend_mode, self.prepend_text)
        try:
            return export_md5anim(
                self.filepath, context,
                action_names=names,
                scale=self.get_scale(),
                rotation=self.get_rotation(),
                compress=self.compress_anim,
                delta_threshold=self.delta_threshold,
                bounds_scale=self.bounds_scale,
                filename_stem_transform=transform,
            )
        except ValueError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"MD5 anim export failed: {e}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}


# ---------------------------------------------------------------------------
# Combined Mesh/Anims export operator
# ---------------------------------------------------------------------------

class EXPORT_OT_md5mesh_anim(Operator, ExportHelper, MD5_ExportScaleRotMixin):
    """Export the active armature and meshes as an MD5 mesh file, then export
    animation(s) to the same directory."""
    bl_idname  = "export_scene.md5mesh_anim"
    bl_label   = "Export Mesh/Anims"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".md5mesh"
    filter_glob: StringProperty(default="*.md5mesh", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return _poll_has_armature(context)

    bone_influence_limit: IntProperty(
        name="Bone Influence Limit",
        description="Maximum number of bones that can influence each vertex",
        default=4, min=1, max=8,
    )
    use_v12: BoolProperty(
        name="MD5 Version 12",
        description="Write MD5Version 12 with per-vertex normals, MikkTSpace "
                    "tangents, and optional vertex colors",
        default=False,
    )
    use_sharp_edges: BoolProperty(
        name="Split Sharp Edges",
        description="Split vertices at edges marked sharp. Non-destructive.",
        default=False,
    )
    compress_anim: BoolProperty(
        name="Compress Animation",
        default=True,
        description="Only export components that move beyond the delta threshold",
    )
    delta_threshold: FloatProperty(
        name="Delta Threshold",
        description="When compressing: minimum change required for a component to be considered animated",
        default=0.0001, min=0.0, max=1.0, precision=6,
    )
    bounds_scale: FloatProperty(
        name="Bounds Scale",
        description="Scale the bounding box for each frame by this factor, "
                    "expanding or contracting it around its centre. "
                    "Use to correct bounds that appear too small or too large in-engine",
        default=1.0, min=0.0, max=1000.0, precision=4,
    )
    change_prepend: BoolProperty(
        name="Change Prepend",
        description="Add or remove a prefix from each exported animation filename",
        default=False,
    )
    prepend_mode: EnumProperty(
        name="Mode",
        description="Whether to add or remove the prepend text",
        items=[
            ('ADD',    "Add",    "Prepend the text to the filename stem"),
            ('REMOVE', "Remove", "Strip the text from the start of the filename stem"),
        ],
        default='ADD',
    )
    prepend_text: StringProperty(
        name="Prepend Text",
        description="Text to add to or remove from exported animation filenames",
        default="",
    )

    def invoke(self, context, event):
        ao = context.active_object
        active_name = ""
        try:
            col = get_collection_of_active_object(context)
            self.filepath     = col.name + self.filename_ext
            self.prepend_text = col.name + "_"
        except ValueError:
            pass
        if ao and ao.animation_data and ao.animation_data.action:
            active_name = ao.animation_data.action.name
        _populate_action_list(context, active_name, select_all=True)
        return super().invoke(context, event)

    def draw(self, context):
        layout = self.layout
        #box = layout.box()
        #box.label(text="Mesh Export", icon='MESH_DATA')
        #self.draw_transforms(box)
        self.draw_transforms(layout)
        col = layout.column(align=True)
        col.prop(self, "bone_influence_limit")
        col.prop(self, "use_v12")
        col.prop(self, "use_sharp_edges")
        anim_col = layout.column(align=True)
        anim_col.prop(self, "compress_anim")
        delta_row = anim_col.row(align=True)
        delta_row.enabled = self.compress_anim
        delta_row.prop(self, "delta_threshold")
        anim_col.prop(self, "bounds_scale")
        selected_count = sum(1 for it in context.scene.md5_action_items if it.selected)
        anim_col.label(text=f"Selected Export Actions: {selected_count}")
        _draw_action_list(anim_col, context, rows=6)
        anim_col.separator()
        col2 = anim_col.column(align=True)
        col2.prop(self, "change_prepend")
        mode_row = col2.row(align=True)
        mode_row.enabled = self.change_prepend
        mode_row.prop(self, "prepend_mode", expand=True)
        txt_row = col2.row()
        txt_row.enabled = self.change_prepend
        txt_row.prop(self, "prepend_text", text="")

    @exports_in_object_mode
    def execute(self, context):
        names = _get_selected_actions(context)
        transform = _make_stem_transform(self.change_prepend, self.prepend_mode, self.prepend_text)
        # Export mesh — always
        try:
            result = export_md5mesh(
                self.filepath, context,
                scale=self.get_scale(),
                rotation=self.get_rotation(),
                bone_influence_limit=self.bone_influence_limit,
                v12=self.use_v12,
                use_sharp_edges=self.use_sharp_edges,
            )
            if result != {'FINISHED'}:
                return result
        except Exception as e:
            self.report({'ERROR'}, f"MD5 mesh export failed: {e}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}
        # Export anims — skip gracefully if nothing selected
        if not names:
            self.report({'INFO'}, "Mesh exported. No actions selected — skipping anim export.")
            return {'FINISHED'}
        try:
            export_md5anim(
                self.filepath, context,
                action_names=names,
                scale=self.get_scale(),
                rotation=self.rotation,
                compress=self.compress_anim,
                delta_threshold=self.delta_threshold,
                filename_stem_transform=transform,
            )
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"MD5 anim export failed: {e}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}

# ===========================================================================
# MD5 Camera Import / Export
# ===========================================================================

import math as _math
import re   as _re

def _normalize180(angle):
    """Clamp an angle in radians to the range [-π, π]."""
    while angle < -_math.pi:
        angle += 2 * _math.pi
    while angle >  _math.pi:
        angle -= 2 * _math.pi
    return angle


# ---------------------------------------------------------------------------
# Parse / write helpers
# ---------------------------------------------------------------------------

def parse_md5camera(filepath):
    """Parse a .md5camera file and return a dict:
      {numFrames, frameRate, numCuts, cuts:[int], frames:[{pos,quat,fov}]}
    Handles Windows (CRLF) and Unix line endings, // comments, and
    flexible whitespace around block braces.
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
        raw = fh.read()

    # Normalise line endings and strip // comments
    raw = raw.replace('\r\n', '\n').replace('\r', '\n')
    raw = _re.sub(r'//[^\n]*', '', raw)

    # Helper: find "keyword <integer>" anywhere in the file
    def _get_int(keyword):
        m = _re.search(keyword + r'[ \t]+(\d+)', raw)
        if not m:
            raise ValueError(f"Could not find '{keyword}' in file")
        return int(m.group(1))

    num_frames = _get_int('numFrames')
    frame_rate = _get_int('frameRate')
    num_cuts   = _get_int('numCuts')

    # Extract cuts block — content between "cuts {" ... "}"
    cuts = []
    m = _re.search(r'cuts\s*\{([^}]*)\}', raw)
    if m:
        for tok in m.group(1).split():
            try:
                cuts.append(int(tok))
            except ValueError:
                pass

    # Extract camera block — content between "camera {" ... "}"
    frames = []
    m = _re.search(r'camera\s*\{([^}]*)\}', raw)
    if m:
        block = m.group(1).replace('(', ' ').replace(')', ' ')
        for line in block.splitlines():
            vals = line.split()
            if len(vals) >= 7:
                try:
                    px, py, pz = float(vals[0]), float(vals[1]), float(vals[2])
                    rx, ry, rz = float(vals[3]), float(vals[4]), float(vals[5])
                    fov        = float(vals[6])
                    frames.append({'pos': (px, py, pz),
                                   'quat': (rx, ry, rz),
                                   'fov': fov})
                except ValueError:
                    pass

    return {'numFrames': num_frames, 'frameRate': frame_rate,
            'numCuts': num_cuts, 'cuts': cuts, 'frames': frames}
def _camera_pos_to_blender(px, py, pz, reorient_deg, offset, scale=1.0):
    """Convert MD5 camera position to Blender space.
    1. This is not needed : Negate Y (MD5 +Y left -> Blender +Y forward) 
    2. Apply uniform scale (MD5 units -> Blender units)
    3. Apply reorient rotation in Blender space
    4. Add XYZ offsets in Blender space (offsets are never scaled)"""
    from mathutils import Vector, Matrix
    #v = Vector((px, -py, pz)) * scale  # flip Y: MD5 -> Blender, then scale
    v = Vector((px, py, pz)) * scale  # scale
    if reorient_deg:
        v.rotate(Matrix.Rotation(_math.radians(reorient_deg), 4, 'Z'))
    # Offsets are in Blender space — applied after conversion, never scaled
    return v.x + offset[0], v.y + offset[1], v.z + offset[2]


def _camera_quat_to_blender(rx, ry, rz, reorient_deg):
    """Convert MD5 camera orientation (packed XYZ of unit quaternion) to
    a Blender Euler XYZ, applying the coordinate-system correction."""
    from mathutils import Quaternion, Euler
    # MD5 stores (qx, qy, qz); recover qw
    qy = -rx;  qx = ry;  qz = -rz
    qw_sq = max(0.0, 1.0 - qx*qx - qy*qy - qz*qz)
    qw = _math.sqrt(qw_sq)
    q = Quaternion((qw, qx, qy, qz))
    euler = q.to_euler('XYZ')
    # Undo the coordinate-system correction applied on export
    euler.x = _normalize180(euler.x + _math.radians(90.0))
    euler.z = _normalize180(euler.z - _math.radians(90.0) + _math.radians(reorient_deg))
    return euler


def _blender_pos_to_md5(obj, reorient_deg, offset, scale=1.0):
    """Convert Blender camera location to MD5 space.
    1. Apply uniform scale to the location only (Blender units -> MD5 units)
    2. Add XYZ offsets in Blender space (offsets are never scaled)
    3. Apply reorient rotation in Blender space
    4. This is not needed: Negate Y (Blender +Y forward -> MD5 +Y left)"""
    from mathutils import Vector, Matrix
    # Offsets are in Blender space — applied after scaling, before rotation
    v = Vector((obj.location.x * scale + offset[0],
                obj.location.y * scale + offset[1],
                obj.location.z * scale + offset[2]))
    if reorient_deg:
        v.rotate(Matrix.Rotation(_math.radians(reorient_deg), 4, 'Z'))
    #return (v.x, -v.y, v.z)
    return (v.x, v.y, v.z)


def _blender_rot_to_md5(obj, reorient_deg):
    """Convert Blender camera rotation to MD5 quaternion XYZ components."""
    # Save and restore rotation mode
    orig_mode = obj.rotation_mode
    obj.rotation_mode = 'XYZ'
    euler = obj.rotation_euler.copy()
    obj.rotation_mode = orig_mode

    euler.x = _normalize180(euler.x - _math.radians(90.0))
    euler.z = _normalize180(euler.z + _math.radians(90.0) + _math.radians(reorient_deg))
    q = euler.to_quaternion()
    # Pack as MD5 XYZ (negating y and z components)
    return (-q.y, q.x, -q.z)


def _fmt10(v):
    return f"{v:.10f}"


def write_md5camera(filepath, frames, cuts, frame_rate):
    """Write a .md5camera file from pre-built frame and cut data."""
    lines = [
        'MD5Version 10',
        'commandline ""',
        '',
        f'numFrames {len(frames)}',
        f'frameRate {frame_rate}',
        f'numCuts {len(cuts)}',
        '',
        'cuts {',
    ]
    for c in cuts:
        lines.append(f'	{c}')
    lines.append('}')
    lines.append('')
    lines.append('camera {')
    for pos, ori, fov in frames:
        px, py, pz = pos
        ox, oy, oz = ori
        lines.append(
            f'	( {_fmt10(px)} {_fmt10(py)} {_fmt10(pz)} )'
            f' ( {_fmt10(ox)} {_fmt10(oy)} {_fmt10(oz)} )'
            f' {fov:.6f}'
        )
    lines.append('}')
    lines.append('')
    with open(filepath, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))


# ---------------------------------------------------------------------------
# Import operator
# ---------------------------------------------------------------------------

class IMPORT_OT_md5camera(Operator, ImportHelper, ImportFileGuardMixin,
                          MD5_ImportScaleRotMixin):
    """Import one or more Doom 3 / id Tech 4 .md5camera files.
    Each file creates its own camera object named MD5_Cam_<stem>
    with keyframed location, rotation, and FOV stored as one Action.
    Cuts are stored as timeline markers named after the action."""
    bl_idname  = "import_scene.md5camera"
    bl_label   = "Import MD5 Camera"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".md5camera"
    filter_glob: StringProperty(default="*.md5camera", options={'HIDDEN'})

    # Multi-file selection support
    files: bpy.props.CollectionProperty(
        type=bpy.types.OperatorFileListElement,
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    directory: StringProperty(subtype='DIR_PATH', options={'HIDDEN', 'SKIP_SAVE'})

 #   reorient: EnumProperty(
 #       name="Reorient",
 #       description="Rotate camera position and orientation around Z during import",
 #       items=[
 #           ('0',   '0°',             'No reorientation'),
 #           ('90',  '90° (X → Y)',    'Rotate 90° CCW from above, idTech4 → Blender'),
 #           ('-90', '-90° (Y → X)',   'Rotate 90° CW from above, Blender → idTech4'),
 #           ('180', '180°',           'Rotate 180°'),
 #       ],
 #       default='0',
 #   )
 
    offset_x: IntProperty(name="X Offset",
        description="Blender space offset applied after camera import, rotation, "
                    "and scale. Not affected by the Scale option", default=0)
    offset_y: IntProperty(name="Y Offset",
        description="Blender space offset applied after camera import, rotation, "
                    "and scale. Not affected by the Scale option", default=0)
    offset_z: IntProperty(name="Z Offset",
        description="Blender space offset applied after camera import, rotation, "
                    "and scale. Not affected by the Scale option", default=0)
    clear_markers: BoolProperty(
        name="Clear Timeline Markers",
        description="Remove existing timeline markers before importing the first file",
        default=True,
    )

    def draw(self, context):
        layout = self.layout
        self.draw_transforms(layout)
        col = layout.column(align=True)
        #col.prop(self, "reorient")
        col.prop(self, "offset_x")
        col.prop(self, "offset_y")
        col.prop(self, "offset_z")
        layout.separator()
        layout.prop(self, "clear_markers")
        layout.separator()
        layout.label(text="Expected scene orientation:", icon='INFO')
        layout.label(text="  +X Forward   +Y Left   +Z Up")
        layout.label(text="Cuts are imported as timeline markers named after the action.")

    def _import_one(self, context, filepath, reorient_deg, offset, scale, first_file):
        """Import a single .md5camera file into its own camera object.
        The camera is named MD5_Cam_<stem> and carries one action.
        Returns (frames, cuts, numFrames, stem)."""
        data = parse_md5camera(filepath)
        stem = os.path.splitext(os.path.basename(filepath))[0]
        obj_name = 'MD5_Cam_' + stem

        # Create a fresh camera data block and object for this file.
        # Reuse an existing one with the same name if it is already a camera.
        cam_obj = bpy.data.objects.get(obj_name)
        if cam_obj is not None and cam_obj.type != 'CAMERA':
            cam_obj = None
        if cam_obj is None:
            cam_data = bpy.data.cameras.new(obj_name)
            cam_obj  = bpy.data.objects.new(obj_name, cam_data)
            context.collection.objects.link(cam_obj)

        cam_obj.data.lens_unit = 'FOV'
        context.view_layer.objects.active = cam_obj

        # Each camera gets its own action.
        if cam_obj.animation_data is None:
            cam_obj.animation_data_create()
        action = bpy.data.actions.new(name=stem)
        cam_obj.animation_data.action = action

        for i, f in enumerate(data['frames']):
            frame_num = i + 1
            px, py, pz = _camera_pos_to_blender(*f['pos'], reorient_deg, offset, scale)
            euler       = _camera_quat_to_blender(*f['quat'], reorient_deg)

            cam_obj.location        = (px, py, pz)
            cam_obj.rotation_mode   = 'XYZ'
            cam_obj.rotation_euler  = euler
            cam_obj.data.angle      = _math.radians(f['fov'])

            cam_obj.keyframe_insert(data_path='location',       frame=frame_num)
            cam_obj.keyframe_insert(data_path='rotation_euler', frame=frame_num)
            cam_obj.data.keyframe_insert(data_path='lens',      frame=frame_num)

        # Set this camera as the scene camera on the first file.
        if first_file:
            context.scene.camera = cam_obj

        # Clear markers only on the first file if requested.
        if first_file and self.clear_markers:
            context.scene.timeline_markers.clear()

        # Add cuts as timeline markers named after the action.
        for cut in data['cuts']:
            m = context.scene.timeline_markers.new(name=stem)
            m.frame = cut

        return len(data['frames']), len(data['cuts']), data['numFrames'], stem

    def execute(self, context):
        # Resolve the file list first. `files` is always populated - with
        # a single empty-named placeholder entry when nothing is selected
        # - so testing it for truth (as this did) silently produced one
        # path that was really just self.directory.
        filepaths, status = self.guard_input_files(multi=True)
        if status:
            return status

        #reorient_deg = int(self.reorient)
        reorient_deg = int(ROTATION_DEGREE_PRESETS[self.get_rotation()])
        print('.md5camera input reorient_deg = ',reorient_deg)
        offset       = (self.offset_x, self.offset_y, self.offset_z)
        scale        = self.get_scale()
        print('.md5camera input scale = ',scale)

        total_frames = 0
        total_cuts   = 0
        last_num_frames = 0
        errors = []

        for idx, fp in enumerate(filepaths):
            try:
                frames, cuts, num_frames, stem = self._import_one(
                    context, fp, reorient_deg, offset, scale,
                    first_file=(idx == 0)
                )
                total_frames += frames
                total_cuts   += cuts
                last_num_frames = max(last_num_frames, num_frames)
            except Exception as e:
                import traceback; traceback.print_exc()
                errors.append(f"{os.path.basename(fp)}: {e}")

        imported = len(filepaths) - len(errors)
        # Only touch the scene's frame range if something was actually
        # imported - otherwise a wholly failed import would leave
        # frame_end at 0 on the way out.
        if imported:
            context.scene.frame_start = 1
            context.scene.frame_end   = last_num_frames

        if errors:
            for err in errors:
                self.report({'WARNING'}, f"MD5 camera import error — {err}")
        # An import where every file failed is a failure, not a
        # {'FINISHED'} with an "Imported 0 file(s)" INFO next to it: that
        # pushed an undo step for a scene nothing was added to, and left a
        # scripted caller no way to tell the difference. Partial success
        # still finishes - the files that did load are really in the scene.
        if not imported:
            self.report({'ERROR'}, "No MD5 camera file could be imported.")
            return {'CANCELLED'}
        self.report({'INFO'},
            f"Imported {imported} file(s), "
            f"{total_frames} frames, {total_cuts} cuts")
        return {'FINISHED'}



# ---------------------------------------------------------------------------
# Camera action checklist infrastructure
# ---------------------------------------------------------------------------

class MD5_CameraActionItem(bpy.types.PropertyGroup):
    """One row in the camera export checklist."""
    selected: BoolProperty(name="", default=True)


class MD5_UL_camera_actions(bpy.types.UIList):
    bl_idname = "MD5_UL_camera_actions"
    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_propname, index):
        layout.prop(item, "selected", text=item.name)


class EXPORT_OT_md5_cam_actions_all(bpy.types.Operator):
    """Select all camera actions"""
    bl_idname = "export_scene.md5_cam_actions_all"
    bl_label  = "All"
    def execute(self, context):
        for it in context.scene.md5_cam_action_items:
            it.selected = True
        return {'FINISHED'}


class EXPORT_OT_md5_cam_actions_none(bpy.types.Operator):
    """Deselect all camera actions"""
    bl_idname = "export_scene.md5_cam_actions_none"
    bl_label  = "None"
    def execute(self, context):
        for it in context.scene.md5_cam_action_items:
            it.selected = False
        return {'FINISHED'}


class EXPORT_OT_md5_cam_actions_invert(bpy.types.Operator):
    """Invert camera action selection"""
    bl_idname = "export_scene.md5_cam_actions_invert"
    bl_label  = "Invert"
    def execute(self, context):
        for it in context.scene.md5_cam_action_items:
            it.selected = not it.selected
        return {'FINISHED'}


def _populate_camera_action_list(context, active_action_name=""):
    """Rebuild scene.md5_cam_action_items from all actions in the scene.
    The active camera action is listed first and pre-selected."""
    scene = context.scene
    scene.md5_cam_action_items.clear()
    scene.md5_cam_action_list_index = 0

    # List all actions in the scene; the user selects which to export.
    actions = sorted(bpy.data.actions, key=lambda a: a.name)
    if active_action_name:
        actions.sort(key=lambda a: (0 if a.name == active_action_name else 1, a.name))

    for action in actions:
        item = scene.md5_cam_action_items.add()
        item.name     = action.name
        item.selected = (action.name == active_action_name)

    # If nothing selected, pre-select all
    if not any(it.selected for it in scene.md5_cam_action_items):
        for it in scene.md5_cam_action_items:
            it.selected = True


def _draw_camera_action_list(layout, context):
    scene = context.scene
    row = layout.row(align=True)
    row.operator("export_scene.md5_cam_actions_all",    text="All",    icon='CHECKBOX_HLT')
    row.operator("export_scene.md5_cam_actions_none",   text="None",   icon='CHECKBOX_DEHLT')
    row.operator("export_scene.md5_cam_actions_invert", text="Invert", icon='ARROW_LEFTRIGHT')
    layout.template_list(
        "MD5_UL_camera_actions", "",
        scene, "md5_cam_action_items",
        scene, "md5_cam_action_list_index",
        rows=8,
    )


def _get_selected_camera_actions(context):
    return [it.name for it in context.scene.md5_cam_action_items if it.selected]


# ---------------------------------------------------------------------------
# Export operator
# ---------------------------------------------------------------------------

class EXPORT_OT_md5camera(Operator, ExportHelper, MD5_ExportScaleRotMixin):
    """Export one or more MD5 camera actions as Doom 3 / id Tech 4 .md5camera files.
    Each selected action is evaluated frame-by-frame and written to its own file
    in the chosen directory. Only timeline markers whose name matches the action
    are written as cuts."""
    bl_idname  = "export_scene.md5camera"
    bl_label   = "Export MD5 Camera"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".md5camera"
    filter_glob: StringProperty(default="*.md5camera", options={'HIDDEN'})

    fps: IntProperty(
        name="Frame Rate",
        description="Frame rate written into the .md5camera header",
        default=24, min=1, max=240,
    )
    #reorient: EnumProperty(
    #    name="Reorient",
    #    description="Rotate camera position and orientation around Z during export",
    #    items=[
    #        ('0',   '0°',             'No reorientation'),
    #        ('90',  '90° (X → Y)',    'Rotate 90° CCW from above, idTech4 → Blender'),
    #        ('-90', '-90° (Y → X)',   'Rotate 90° CW from above, Blender → idTech4'),
    #        ('180', '180°',           'Rotate 180°'),
    #    ],
    #    default='0',
    #)
    
    offset_x: IntProperty(name="X Offset",
        description="Blender space offset applied to camera positions after scale, "
                    "before export rotation. Not affected by the Scale option", default=0)
    offset_y: IntProperty(name="Y Offset",
        description="Blender space offset applied to camera positions after scale, "
                    "before export rotation. Not affected by the Scale option", default=0)
    offset_z: IntProperty(name="Z Offset",
        description="Blender space offset applied to camera positions after scale, "
                    "before export rotation. Not affected by the Scale option", default=0)
    export_all_cuts: BoolProperty(
        name="Export All Cuts",
        description="By default only cuts in the timeline with the same name as the "
                    "action are exported. Check this to export all cuts on the "
                    "timeline regardless of their name. "
                    "This probably wont work well if exporting multiple actions.",
        default=False,
    )

    @classmethod
    def poll(cls, context):
        return context.scene.camera is not None or any(
            o.type == 'CAMERA' for o in context.scene.objects)

    def invoke(self, context, event):
        # If the active object is a camera with an action, use that.
        # Otherwise fall back to the scene camera.
        active_name = ""
        ao = context.active_object
        if ao and ao.type == 'CAMERA' and ao.animation_data and ao.animation_data.action:
            active_name = ao.animation_data.action.name
        elif context.scene.camera and context.scene.camera.animation_data and                 context.scene.camera.animation_data.action:
            active_name = context.scene.camera.animation_data.action.name
        if active_name:
            self.filepath = active_name + self.filename_ext
        _populate_camera_action_list(context, active_name)
        return super().invoke(context, event)

    def draw(self, context):
        layout = self.layout
        #self.draw_scale(layout)
        self.draw_transforms(layout)
        col = layout.column(align=True)
        col.prop(self, "fps")
        #col.prop(self, "reorient")
        col.prop(self, "offset_x")
        col.prop(self, "offset_y")
        col.prop(self, "offset_z")
        layout.separator()
        selected_count = sum(1 for it in context.scene.md5_cam_action_items if it.selected)
        layout.label(text=f"Selected Camera Actions: {selected_count}")
        _draw_camera_action_list(layout, context)
        layout.separator()
        layout.label(text="Expected scene orientation:", icon='INFO')
        layout.label(text="  +X Forward   +Y Left   +Z Up")
        layout.prop(self, "export_all_cuts")

    def _export_one(self, context, cam_obj, action_name, directory,
                    reorient_deg, offset, scale, filepath):
        """Switch to action_name, bake frames, write file. Returns filepath."""
        scene = context.scene
        action = bpy.data.actions.get(action_name)
        if action is None:
            raise ValueError(f"Action '{action_name}' not found")

        # Assign the action to the camera object so frame_set evaluates
        # location and rotation correctly.
        if cam_obj.animation_data is None:
            cam_obj.animation_data_create()
        cam_obj.animation_data.action = action
        _md5_get_or_create_slot(cam_obj, action)
        if cam_obj.data.animation_data is None:
            cam_obj.data.animation_data_create()
        cam_obj.data.animation_data.action = action
        _md5_get_or_create_slot(cam_obj.data, action)

        # Determine frame range from the action's keyframe range
        frame_range = action.frame_range
        frame_start = int(frame_range[0])
        frame_end   = int(frame_range[1])

        orig_frame = scene.frame_current
        frames = []
        try:
            for fn in range(frame_start, frame_end + 1):
                scene.frame_set(fn)
                pos = _blender_pos_to_md5(cam_obj, reorient_deg, offset, scale)
                ori = _blender_rot_to_md5(cam_obj, reorient_deg)
                fov = _math.degrees(cam_obj.data.angle)
                frames.append((pos, ori, fov))
        finally:
            scene.frame_set(orig_frame)

        # Cuts: all markers, or only those named after this action
        cuts = sorted({
            int(m.frame) for m in scene.timeline_markers
            if self.export_all_cuts or m.name == action_name
        })

        #filepath = os.path.join(directory, action_name + self.filename_ext)
        write_md5camera(filepath, frames, cuts, self.fps)
        return filepath, len(frames), len(cuts)

    def _find_camera_for_action(self, context, action_name):
        """Return the camera object whose animation_data references this action.
        Falls back to the scene camera or first camera in the scene."""
        # Prefer a camera whose current action matches
        for obj in context.scene.objects:
            if obj.type != 'CAMERA':
                continue
            if (obj.animation_data and obj.animation_data.action and
                    obj.animation_data.action.name == action_name):
                return obj
        # Also check if any camera has this action on its data block
        for obj in context.scene.objects:
            if obj.type != 'CAMERA':
                continue
            if (obj.data.animation_data and obj.data.animation_data.action and
                    obj.data.animation_data.action.name == action_name):
                return obj
        # Fall back to scene camera or first camera
        cam = context.scene.camera
        if cam and cam.type == 'CAMERA':
            return cam
        return next((o for o in context.scene.objects if o.type == 'CAMERA'), None)

    def execute(self, context):
        scene = context.scene

        selected_names = _get_selected_camera_actions(context)
        if not selected_names:
            self.report({'WARNING'}, "No camera actions selected for export.")
            return {'CANCELLED'}

        #reorient_deg = int(self.reorient)
        reorient_deg = ROTATION_DEGREE_PRESETS[self.get_rotation()]
        offset       = (self.offset_x, self.offset_y, self.offset_z)
        scale        = self.get_scale()
        directory    = os.path.dirname(self.filepath)

        errors = []
        exported = 0
        for name in selected_names:
            # Find the camera object that owns this action
            cam_obj = self._find_camera_for_action(context, name)
            if cam_obj is None:
                errors.append(f"{name}: no camera object found in scene")
                continue

            # Remember original actions so we can restore after export
            orig_obj_action  = (cam_obj.animation_data.action
                                if cam_obj.animation_data else None)
            orig_data_action = (cam_obj.data.animation_data.action
                                if cam_obj.data.animation_data else None)
            try:
                fp, nframes, ncuts = self._export_one(
                    context, cam_obj, name, directory, reorient_deg, offset, scale, self.filepath)
                exported += 1
            except Exception as e:
                import traceback; traceback.print_exc()
                errors.append(f"{name}: {e}")
            finally:
                # Restore original actions
                if cam_obj.animation_data:
                    cam_obj.animation_data.action = orig_obj_action
                    if orig_obj_action is not None:
                        _md5_get_or_create_slot(cam_obj, orig_obj_action)
                if cam_obj.data.animation_data:
                    cam_obj.data.animation_data.action = orig_data_action
                    if orig_data_action is not None:
                        _md5_get_or_create_slot(cam_obj.data, orig_data_action)

        for err in errors:
            self.report({'WARNING'}, f"MD5 camera export error — {err}")
        self.report({'INFO'}, f"Exported {exported} camera file(s) to '{directory}'")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Sources gate
# ---------------------------------------------------------------------------
# File > Import points straight at IMPORT_OT_md5mesh/bmd5mesh now —
# unlike the old design, neither gate below runs in front of the menu
# entry. See idTech4_ase_lwo_io.py's copy of MaterialsSourceGateMixin
# for the full reasoning — this mirrors it exactly. Each gate only fires
# from INSIDE its own operator's execute(), and only once the file AND
# every checkbox are already chosen: MaterialGenMixin._needs_sources_
# gate() checks whether Auto-Generate Materials is on, its companion
# addon is installed, and Base Directory/Materials Source aren't already
# resolvable — only then does execute() launch the matching gate (via
# _launch_sources_gate). That means the popup below no longer needs its
# own "is it even needed" check — by the time it's shown, that's already
# been decided.
#
# self.filepath plus every one of the real operator's own already-set
# properties are handed off via _pending_import_kwargs (a plain module
# global — simpler than mirroring properties onto each gate class just
# to receive them). _launch_target merges in whatever this popup
# resolved and re-invokes the real operator via EXEC_DEFAULT (filepath
# is already known, so there's no reason to reopen the file browser),
# with gate_resolved=True so _needs_sources_gate never fires the same
# gate twice for one import.

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
    """Shared popup + dispatch logic for every gate operator below."""

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


class IMPORT_OT_md5mesh_gate(Operator, MaterialsSourceGateMixin):
    """Resolve Base Directory / Materials Source for an MD5 mesh import
    already in progress (file and options already chosen), then
    re-launch it"""
    bl_idname  = "import_scene.md5mesh_gate"
    bl_label   = "MD5 Import : Base Directory / Materials Source Setup"
    bl_options = {'INTERNAL'}

    filepath: StringProperty(subtype='FILE_PATH', options={'HIDDEN', 'SKIP_SAVE'})

    def _target_op(self):
        return bpy.ops.import_scene.md5mesh


def menu_func_import_camera(self, context):
    self.layout.operator(IMPORT_OT_md5camera.bl_idname,
                         text="idTech4 MD5 Camera (.md5camera)")

def menu_func_export_camera(self, context):
    self.layout.operator(EXPORT_OT_md5camera.bl_idname,
                         text="idTech4 MD5 Camera (.md5camera)")


# ---------------------------------------------------------------------------
# Menu hooks
# ---------------------------------------------------------------------------

def menu_func_import_mesh(self, context):
    self.layout.operator(IMPORT_OT_md5mesh.bl_idname,
                         text="idTech4 MD5 Mesh (.md5mesh, .bmd5mesh)")

def menu_func_import_anim(self, context):
    self.layout.operator(IMPORT_OT_md5anim.bl_idname,
                         text="idTech4 MD5 Anim (.md5anim, .bmd5anim)")

def menu_func_export_mesh(self, context):
    self.layout.operator(EXPORT_OT_md5mesh.bl_idname,
                         text="idTech4 MD5 Mesh (.md5mesh)")

def menu_func_export_anim(self, context):
    self.layout.operator(EXPORT_OT_md5anim.bl_idname,
                         text="idTech4 MD5 Anim (.md5anim)")

def menu_func_export_mesh_anim(self, context):
    self.layout.operator(EXPORT_OT_md5mesh_anim.bl_idname,
                         text="idTech4 MD5 Mesh+Anim(s) (.md5mesh,.md5anim)")


# ===========================================================================
# MD5 Bone Collection Management — Data Properties Panel
# ===========================================================================

MD5_BONE_COLLECTION_NAME = "MD5_export_bone_collection"


def _get_or_create_md5_collection(arm_data):
    """Return the MD5 bone collection, creating it if it doesn't exist."""
    if MD5_BONE_COLLECTION_NAME not in arm_data.collections:
        arm_data.collections.new(MD5_BONE_COLLECTION_NAME)
    return arm_data.collections[MD5_BONE_COLLECTION_NAME]


def _get_active_armature(context):
    """Return the armature data from the active object, or None."""
    obj = context.active_object
    if obj is None:
        return None
    if obj.type == 'ARMATURE':
        return obj.data
    if obj.type == 'MESH':
        for mod in obj.modifiers:
            if mod.type == 'ARMATURE' and mod.object:
                return mod.object.data
    return None


class MD5BONES_OT_add_selected(bpy.types.Operator):
    """Add selected bones to the MD5_export_bone_collection"""
    bl_idname = "md5bones.add_selected"
    bl_label  = "Add Selected"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arm = _get_active_armature(context)
        if arm is None:
            self.report({'ERROR'}, "No active armature")
            return {'CANCELLED'}
        coll = _get_or_create_md5_collection(arm)
        count = 0
        for bone in context.selected_bones or []:
            coll.assign(bone)
            count += 1
        self.report({'INFO'}, f"Added {count} bone(s) to {MD5_BONE_COLLECTION_NAME}")
        return {'FINISHED'}


class MD5BONES_OT_remove_selected(bpy.types.Operator):
    """Remove selected bones from the MD5_export_bone_collection"""
    bl_idname = "md5bones.remove_selected"
    bl_label  = "Remove Selected"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arm = _get_active_armature(context)
        if arm is None:
            self.report({'ERROR'}, "No active armature")
            return {'CANCELLED'}
        if MD5_BONE_COLLECTION_NAME not in arm.collections:
            self.report({'WARNING'}, f"{MD5_BONE_COLLECTION_NAME} does not exist")
            return {'CANCELLED'}
        coll = arm.collections[MD5_BONE_COLLECTION_NAME]
        count = 0
        for bone in context.selected_bones or []:
            coll.unassign(bone)
            count += 1
        self.report({'INFO'}, f"Removed {count} bone(s) from {MD5_BONE_COLLECTION_NAME}")
        return {'FINISHED'}


class MD5BONES_OT_replace_with_selected(bpy.types.Operator):
    """Clear MD5_export_bone_collection and add selected bones"""
    bl_idname = "md5bones.replace_with_selected"
    bl_label  = "Replace with Selected"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arm = _get_active_armature(context)
        if arm is None:
            self.report({'ERROR'}, "No active armature")
            return {'CANCELLED'}
        coll = _get_or_create_md5_collection(arm)
        # Unassign all current members
        for bone in arm.bones:
            coll.unassign(bone)
        # Assign selected
        count = 0
        for bone in context.selected_bones or []:
            coll.assign(bone)
            count += 1
        self.report({'INFO'}, f"Replaced — {count} bone(s) now in {MD5_BONE_COLLECTION_NAME}")
        return {'FINISHED'}


class MD5BONES_OT_clear_all(bpy.types.Operator):
    """Remove all bones from MD5_export_bone_collection"""
    bl_idname = "md5bones.clear_all"
    bl_label  = "Clear All"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        arm = _get_active_armature(context)
        if arm is None:
            self.report({'ERROR'}, "No active armature")
            return {'CANCELLED'}
        if MD5_BONE_COLLECTION_NAME not in arm.collections:
            self.report({'WARNING'}, f"{MD5_BONE_COLLECTION_NAME} does not exist")
            return {'CANCELLED'}
        coll = arm.collections[MD5_BONE_COLLECTION_NAME]
        for bone in arm.bones:
            coll.unassign(bone)
        self.report({'INFO'}, f"Cleared {MD5_BONE_COLLECTION_NAME}")
        return {'FINISHED'}


class MD5BONES_OT_select_collection(bpy.types.Operator):
    """Deselect all bones then select all bones in MD5_export_bone_collection"""
    bl_idname = "md5bones.select_collection"
    bl_label  = "Select Collection"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if obj is None or obj.type != 'ARMATURE':
            self.report({'ERROR'}, "No active armature object")
            return {'CANCELLED'}
        arm = obj.data
        if MD5_BONE_COLLECTION_NAME not in arm.collections:
            self.report({'WARNING'}, f"{MD5_BONE_COLLECTION_NAME} does not exist")
            return {'CANCELLED'}
        coll = arm.collections[MD5_BONE_COLLECTION_NAME]

        # Switch to Pose Mode so bone.select works reliably
        prev_mode = obj.mode
        if prev_mode != 'POSE':
            bpy.ops.object.mode_set(mode='POSE')

        # Deselect all bones first
        bpy.ops.pose.select_all(action='DESELECT')

        # Select members via pose_bone.bone.select
        count = 0
        for bone in arm.bones:
            in_coll = coll.name in [c.name for c in bone.collections]
            bone.select = in_coll
            bone.select_head = in_coll
            bone.select_tail = in_coll
            if in_coll:
                count += 1

        # Restore previous mode
        if prev_mode != 'POSE':
            bpy.ops.object.mode_set(mode=prev_mode)

        self.report({'INFO'}, f"Selected {count} bone(s) from {MD5_BONE_COLLECTION_NAME}")
        return {'FINISHED'}


class MD5BONES_PT_panel(bpy.types.Panel):
    """Manage which bones are included in MD5 exports"""
    bl_label       = "Manage MD5 Export Bone Collection"
    bl_idname      = "DATA_PT_md5_bone_collection"
    bl_space_type  = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context     = "data"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'ARMATURE'

    def draw(self, context):
        layout = self.layout
        arm    = context.active_object.data

        has_selected = bool(context.selected_bones)
        has_coll     = MD5_BONE_COLLECTION_NAME in arm.collections

        col = layout.column(align=True)

        # Add: always enabled (creates collection if needed)
        col.operator(MD5BONES_OT_add_selected.bl_idname,
                     text="Add Selected (Create collection if needed)",
                     icon='ADD')

        # Remove / Replace: need selected bones AND existing collection
        row = col.row(align=True)
        row.enabled = has_selected and has_coll
        row.operator(MD5BONES_OT_remove_selected.bl_idname,       icon='REMOVE')

        row2 = col.row(align=True)
        row2.enabled = has_selected and has_coll
        row2.operator(MD5BONES_OT_replace_with_selected.bl_idname, icon='FILE_REFRESH')

        col.separator()

        # Clear all / Select collection: need existing collection
        row3 = col.row(align=True)
        row3.enabled = has_coll
        row3.operator(MD5BONES_OT_clear_all.bl_idname,             icon='TRASH')

        row4 = col.row(align=True)
        row4.enabled = has_coll
        row4.operator(MD5BONES_OT_select_collection.bl_idname,     text="Select All Bones in Collection", icon='RESTRICT_SELECT_OFF')



# ===========================================================================
# MD5 Animation Retargeter
# ===========================================================================
# Corrects an action's rotation keyframes after bone orientations have been
# changed in Edit Mode on the same armature.
#
# WORKFLOW:
#   Step 1 — Select armature → "Snapshot Rest Pose" (run BEFORE editing bones)
#   Step 2 — Enter Edit Mode, rotate/realign bones, return to Object Mode
#   Step 3 — Select armature → "Retarget Animation(s)"
#
# HOW IT WORKS:
#   tgt_basis_rot = C @ src_basis_rot @ inv(C)
#   where C = inv(new_bone.matrix_local) @ old_bone.matrix_local  (3x3 only)
# ===========================================================================

import json as _json
from collections import defaultdict as _defaultdict

ART_SNAPSHOT_PROP = "art_rest_snapshot"   # JSON: {bone_name: [9 floats]}
ART_BACKUP_PROP   = "art_backup_action"   # name of backup action


def _art_get_snapshot(obj):
    """Return {bone_name: Matrix3x3} from stored JSON, or None."""
    raw = obj.get(ART_SNAPSHOT_PROP)
    if not raw:
        return None
    try:
        data = _json.loads(raw)
        return {name: Matrix((m[0:3], m[3:6], m[6:9]))
                for name, m in data.items()}
    except Exception:
        return None


def _art_set_snapshot(obj, snapshot):
    """Store {bone_name: Matrix3x3} as JSON on the object."""
    data = {name: list(m[0]) + list(m[1]) + list(m[2])
            for name, m in snapshot.items()}
    obj[ART_SNAPSHOT_PROP] = _json.dumps(data)


# ---------------------------------------------------------------------------
#  Scene property group
# ---------------------------------------------------------------------------

class ART_Props(bpy.types.PropertyGroup):
    status: StringProperty(default="")
    adjust_lengths: bpy.props.BoolProperty(
        name        = "Adjust Bone Lengths",
        description = "Resize each bone so its tail reaches the child's head",
        default     = True,
    )
    only_selected: bpy.props.BoolProperty(
        name        = "Only Realign Selected",
        description = "Only modify bones that are currently selected in the armature",
        default     = False,
    )
    retarget_all: bpy.props.BoolProperty(
        name        = "Retarget All Actions",
        description = "Correct every action in the file, not just the active one",
        default     = True,
    )


# ---------------------------------------------------------------------------
#  Operator: Snapshot Rest Pose
# ---------------------------------------------------------------------------

class ART_OT_Snapshot(bpy.types.Operator):
    """Record current bone rest matrices before editing orientations in Edit Mode"""
    bl_idname  = "art.snapshot"
    bl_label   = "Snapshot Rest Pose"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        o = ctx.active_object
        return o and o.type == 'ARMATURE' and ctx.mode == 'OBJECT'

    def execute(self, ctx):
        obj      = ctx.active_object
        snapshot = {b.name: b.matrix_local.to_3x3() for b in obj.data.bones}
        _art_set_snapshot(obj, snapshot)

        bpy.ops.object.mode_set(mode='EDIT')
        ctx.scene.tool_settings.transform_pivot_point = 'ACTIVE_ELEMENT'

        msg = f"Snapshot saved: {len(snapshot)} bones. Now in Edit Mode (pivot = Active Element)."
        self.report({'INFO'}, msg)
        ctx.scene.art.status = msg
        return {'FINISHED'}


# ---------------------------------------------------------------------------
#  Operator: Auto Realign Bones
# ---------------------------------------------------------------------------

def _art_count_descendants(bone):
    total = 0
    for child in bone.children:
        total += 1 + _art_count_descendants(child)
    return total


class ART_OT_RealignBones(bpy.types.Operator):
    """Automatically point each bone's tail toward the dominant child.
    If one child has more descendants than all others, the tail points at
    that child's head. If multiple children tie, the tail points at the
    median of those children's head positions. Optionally adjusts bone
    length so the tail reaches the target. Bones sharing a head location
    get staggered lengths."""
    bl_idname  = "art.realign_bones"
    bl_label   = "Auto Realign Bones"
    bl_options = {'REGISTER', 'UNDO'}

    adjust_lengths: bpy.props.BoolProperty(
        name="Adjust Bone Lengths",
        description="Resize each bone so its tail reaches the child's head",
        default=True, options={'HIDDEN'},
    )
    only_selected: bpy.props.BoolProperty(
        name="Only Realign Selected",
        description="Only modify bones that are currently selected",
        default=False, options={'HIDDEN'},
    )

    @classmethod
    def poll(cls, ctx):
        o = ctx.active_object
        return (o and o.type == 'ARMATURE'
                and ctx.mode == 'EDIT_ARMATURE'
                and o.get(ART_SNAPSHOT_PROP))

    def execute(self, ctx):
        obj   = ctx.active_object
        arm   = obj.data
        bones = arm.edit_bones

        desc_count = {b.name: _art_count_descendants(b) for b in bones}

        # For each bone, determine the target position for its tail.
        # If one child has more descendants than all others, point at that
        # child's head. If multiple children tie for the highest descendant
        # count, compute the median of their heads and point there instead.
        target_pos = {}   # bone.name -> Vector or None
        for bone in bones:
            if not bone.children:
                target_pos[bone.name] = None
                continue
            max_desc = max(desc_count[c.name] for c in bone.children)
            tied = [c for c in bone.children if desc_count[c.name] == max_desc]
            if len(tied) == 1:
                # Single dominant child — point directly at its head
                target_pos[bone.name] = tied[0].head.copy()
            else:
                # Multiple children tied — use the median of their head positions
                xs = sorted(c.head.x for c in tied)
                ys = sorted(c.head.y for c in tied)
                zs = sorted(c.head.z for c in tied)
                n   = len(tied)
                mid = n // 2
                if n % 2 == 1:
                    median = Vector((xs[mid], ys[mid], zs[mid]))
                else:
                    median = Vector((
                        (xs[mid - 1] + xs[mid]) / 2.0,
                        (ys[mid - 1] + ys[mid]) / 2.0,
                        (zs[mid - 1] + zs[mid]) / 2.0,
                    ))
                target_pos[bone.name] = median

        head_groups = _defaultdict(list)
        for bone in bones:
            key = (round(bone.head.x, 4), round(bone.head.y, 4), round(bone.head.z, 4))
            head_groups[key].append(bone.name)

        MIN_LEN   = 0.01
        realigned = 0

        for bone in bones:
            if bone.name == 'origin':
                continue
            if self.only_selected and not bone.select:
                continue

            tgt = target_pos[bone.name]
            if tgt is not None:
                direction = tgt - bone.head
                dist      = direction.length
                if dist > 1e-6:
                    if self.adjust_lengths:
                        bone.tail   = bone.head + direction
                        bone.length = dist
                    else:
                        bone.tail = bone.head + direction.normalized() * bone.length
                    realigned += 1
            else:
                if bone.parent is not None:
                    parent_dir = bone.parent.tail - bone.parent.head
                    if parent_dir.length > 1e-6:
                        length    = max(bone.length, MIN_LEN)
                        bone.tail = bone.head + parent_dir.normalized() * length
                        realigned += 1

            if self.adjust_lengths:
                head_key     = (round(bone.head.x, 4), round(bone.head.y, 4), round(bone.head.z, 4))
                group        = head_groups[head_key]
                if len(group) > 1:
                    group_sorted = sorted(group)
                    idx          = group_sorted.index(bone.name)
                    if idx > 0:
                        bone.length = max(bone.length, MIN_LEN) * (1.0 + idx * 0.15)

        msg = f"Realigned {realigned} bone(s)."
        self.report({'INFO'}, msg)
        ctx.scene.art.status = msg
        return {'FINISHED'}


# ---------------------------------------------------------------------------
#  Operator: Adjust Lengths Only
# ---------------------------------------------------------------------------

class ART_OT_AdjustLengths(bpy.types.Operator):
    """Update each bone's length to reach its dominant child's head,
    without changing bone directions. Stagger lengths for bones sharing
    the same head position. Respects 'Only Realign Selected'."""
    bl_idname  = "art.adjust_lengths"
    bl_label   = "Adjust Lengths Only"
    bl_options = {'REGISTER', 'UNDO'}

    only_selected: bpy.props.BoolProperty(
        name="Only Adjust Selected",
        default=False, options={'HIDDEN'},
    )

    def execute(self, ctx):
        bones = ctx.active_object.data.edit_bones

        desc_count   = {b.name: _art_count_descendants(b) for b in bones}
        target_child = {}
        for bone in bones:
            if not bone.children:
                target_child[bone.name] = None
                continue
            best = max(bone.children, key=lambda c: (desc_count[c.name], c.name))
            target_child[bone.name] = best

        # Group bones by head position for stagger detection
        head_groups = _defaultdict(list)
        for bone in bones:
            key = (round(bone.head.x, 4), round(bone.head.y, 4), round(bone.head.z, 4))
            head_groups[key].append(bone.name)

        MIN_LEN  = 0.01
        adjusted = 0

        for bone in bones:
            if bone.name == 'origin':
                continue
            if self.only_selected and not bone.select:
                continue

            child = target_child[bone.name]
            if child is not None:
                dist = (child.head - bone.head).length
                if dist > 1e-6:
                    bone.length = dist
                    adjusted += 1
            else:
                # Leaf bone: keep existing length, enforce minimum
                if bone.length < MIN_LEN:
                    bone.length = MIN_LEN
                    adjusted += 1

            # Stagger bones sharing the same head location
            head_key     = (round(bone.head.x, 4), round(bone.head.y, 4), round(bone.head.z, 4))
            group        = head_groups[head_key]
            if len(group) > 1:
                group_sorted = sorted(group)
                idx          = group_sorted.index(bone.name)
                if idx > 0:
                    bone.length = max(bone.length, MIN_LEN) * (1.0 + idx * 0.15)

        msg = f"Adjusted {adjusted} bone length(s)."
        self.report({'INFO'}, msg)
        ctx.scene.art.status = msg
        return {'FINISHED'}


# ---------------------------------------------------------------------------
#  Operator: Retarget Animation
# ---------------------------------------------------------------------------

class ART_OT_Retarget(bpy.types.Operator):
    """Correct active action rotation keyframes for the new bone orientations"""
    bl_idname  = "art.retarget"
    bl_label   = "Retarget Animation(s)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        o = ctx.active_object
        return (o and o.type == 'ARMATURE'
                and ctx.mode == 'OBJECT'
                and o.get(ART_SNAPSHOT_PROP)
                and o.animation_data
                and o.animation_data.action)

    def _retarget_action(self, action, corrections):
        rot_curves = _defaultdict(dict)
        loc_curves = _defaultdict(dict)
        for fc in _md5_iter_action_fcurves(action):
            try:
                bone_name = fc.data_path.split('"')[1]
            except IndexError:
                continue
            if bone_name not in corrections:
                continue
            if 'rotation_quaternion' in fc.data_path:
                rot_curves[bone_name][fc.array_index] = fc
            elif 'location' in fc.data_path:
                loc_curves[bone_name][fc.array_index] = fc

        corrected_bones = 0
        for bone_name in set(rot_curves.keys()) | set(loc_curves.keys()):
            C, C_inv = corrections[bone_name]
            rcurves  = rot_curves.get(bone_name, {})
            lcurves  = loc_curves.get(bone_name, {})

            bone_frames = sorted({int(kp.co.x)
                                  for curves in (rcurves, lcurves)
                                  for fc in curves.values()
                                  for kp in fc.keyframe_points})

            from mathutils import Quaternion as _Quat, Vector as _Vec
            last_quat = _Quat((1, 0, 0, 0))

            for frame in bone_frames:
                if set(rcurves.keys()) == {0, 1, 2, 3}:
                    src_rot = _Quat((
                        rcurves[0].evaluate(frame),
                        rcurves[1].evaluate(frame),
                        rcurves[2].evaluate(frame),
                        rcurves[3].evaluate(frame),
                    ))
                    src_rot.normalize()
                    rot = (C @ src_rot.to_matrix() @ C_inv).to_quaternion()
                    rot.normalize()
                    if rot.dot(last_quat) < 0.0:
                        rot = -rot
                    last_quat = rot.copy()
                    for idx, fc in rcurves.items():
                        for kp in fc.keyframe_points:
                            if int(kp.co.x) == frame:
                                kp.co.y           = rot[idx]
                                kp.handle_left.y  = rot[idx]
                                kp.handle_right.y = rot[idx]
                                break

                if set(lcurves.keys()) == {0, 1, 2}:
                    src_loc = _Vec((
                        lcurves[0].evaluate(frame),
                        lcurves[1].evaluate(frame),
                        lcurves[2].evaluate(frame),
                    ))
                    tgt_loc = C @ src_loc
                    for idx, fc in lcurves.items():
                        for kp in fc.keyframe_points:
                            if int(kp.co.x) == frame:
                                kp.co.y           = tgt_loc[idx]
                                kp.handle_left.y  = tgt_loc[idx]
                                kp.handle_right.y = tgt_loc[idx]
                                break

            for fc in list(rcurves.values()) + list(lcurves.values()):
                fc.update()

            corrected_bones += 1
        return corrected_bones

    def execute(self, ctx):
        obj      = ctx.active_object
        snapshot = _art_get_snapshot(obj)
        if not snapshot:
            self.report({'ERROR'}, "No snapshot found. Run 'Snapshot Rest Pose' first.")
            return {'CANCELLED'}
        if not obj.animation_data or not obj.animation_data.action:
            self.report({'ERROR'}, "Armature has no active action.")
            return {'CANCELLED'}

        corrections = {}
        for bone in obj.data.bones:
            name = bone.name
            if name not in snapshot:
                continue
            old_ml = snapshot[name]
            new_ml = bone.matrix_local.to_3x3()
            C      = new_ml.inverted() @ old_ml
            C_inv  = old_ml.inverted() @ new_ml
            identity = Matrix.Identity(3)
            diff = sum(abs(C[r][c] - identity[r][c]) for r in range(3) for c in range(3))
            if diff > 1e-5:
                corrections[name] = (C, C_inv)

        if not corrections:
            self.report({'WARNING'}, "No bone orientation changes detected vs snapshot.")
            return {'CANCELLED'}

        retarget_all = ctx.scene.art.retarget_all
        if retarget_all:
            bone_names = {b.name for b in obj.data.bones}
            actions_to_process = []
            for action in bpy.data.actions:
                for fc in _md5_iter_action_fcurves(action):
                    try:
                        bn = fc.data_path.split('"')[1]
                    except IndexError:
                        continue
                    if bn in bone_names:
                        actions_to_process.append(action)
                        break
        else:
            actions_to_process = [obj.animation_data.action]

        total_actions = len(actions_to_process)
        total_bones   = 0
        backed_up     = []

        wm = ctx.window_manager
        wm.progress_begin(0, total_actions)

        for i, action in enumerate(actions_to_process):
            wm.progress_update(i)
            backup_name = action.name + "_pre_retarget"
            old_backup  = bpy.data.actions.get(backup_name)
            if old_backup:
                bpy.data.actions.remove(old_backup)
            backup      = action.copy()
            backup.name = backup_name
            backed_up.append(backup_name)
            total_bones += self._retarget_action(action, corrections)

        wm.progress_end()
        obj[ART_BACKUP_PROP] = backed_up[-1] if backed_up else ""

        ctx.view_layer.update()
        for area in ctx.screen.areas:
            area.tag_redraw()

        n   = len(actions_to_process)
        msg = (f"Retargeted {n} action(s), {total_bones} bone correction(s). "
               f"{n} backup(s) saved.")
        self.report({'INFO'}, msg)
        ctx.scene.art.status = msg
        return {'FINISHED'}


# ---------------------------------------------------------------------------
#  Operator: Restore Backup
# ---------------------------------------------------------------------------

class ART_OT_RestoreBackup(bpy.types.Operator):
    """Restore the pre-retarget backup action"""
    bl_idname  = "art.restore_backup"
    bl_label   = "Restore Backup"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        o = ctx.active_object
        return (o and o.type == 'ARMATURE'
                and bpy.data.actions.get(o.get(ART_BACKUP_PROP, "")))

    def execute(self, ctx):
        obj         = ctx.active_object
        backup_name = obj.get(ART_BACKUP_PROP, "")
        backup      = bpy.data.actions.get(backup_name)
        if not backup:
            self.report({'ERROR'}, f"Backup action '{backup_name}' not found.")
            return {'CANCELLED'}
        if not obj.animation_data:
            obj.animation_data_create()
        obj.animation_data.action = backup
        _md5_get_or_create_slot(obj, backup)
        msg = f"Restored '{backup_name}'."
        self.report({'INFO'}, msg)
        ctx.scene.art.status = msg
        return {'FINISHED'}


# ---------------------------------------------------------------------------
#  Operator: Delete Backups
# ---------------------------------------------------------------------------

class ART_OT_DeleteBackups(bpy.types.Operator):
    """Delete all _pre_retarget backup actions from this file"""
    bl_idname  = "art.delete_backups"
    bl_label   = "Delete Backups"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return any(a.name.endswith("_pre_retarget") for a in bpy.data.actions)

    def execute(self, ctx):
        to_delete = [a for a in bpy.data.actions if a.name.endswith("_pre_retarget")]
        count = len(to_delete)
        for action in to_delete:
            bpy.data.actions.remove(action)
        obj = ctx.active_object
        if obj and ART_BACKUP_PROP in obj:
            del obj[ART_BACKUP_PROP]
        msg = f"Deleted {count} backup action(s)."
        self.report({'INFO'}, msg)
        ctx.scene.art.status = msg
        return {'FINISHED'}


# ---------------------------------------------------------------------------
#  Operator: Clear Snapshot
# ---------------------------------------------------------------------------

class ART_OT_ClearSnapshot(bpy.types.Operator):
    """Remove the stored rest pose snapshot from this armature"""
    bl_idname  = "art.clear_snapshot"
    bl_label   = "Clear Snapshot"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        o = ctx.active_object
        return o and o.type == 'ARMATURE' and o.get(ART_SNAPSHOT_PROP)

    def execute(self, ctx):
        obj = ctx.active_object
        if ART_SNAPSHOT_PROP in obj:
            del obj[ART_SNAPSHOT_PROP]
        msg = "Snapshot cleared."
        self.report({'INFO'}, msg)
        ctx.scene.art.status = msg
        return {'FINISHED'}


# ---------------------------------------------------------------------------
#  N-Panel  (3D View > N-Panel > MD5 Tools)
# ---------------------------------------------------------------------------

class ART_PT_Panel(bpy.types.Panel):
    bl_label       = "MD5 Animation Retargeter"
    bl_idname      = "ART_PT_main"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = "MD5"

    @classmethod
    def poll(cls, ctx):
        o = ctx.active_object
        return o and o.type == 'ARMATURE'

    def draw(self, ctx):
        layout   = self.layout
        obj      = ctx.active_object
        sp       = ctx.scene.art
        has_snap = bool(obj.get(ART_SNAPSHOT_PROP))
        has_bkup = bool(bpy.data.actions.get(obj.get(ART_BACKUP_PROP, "")))

        # Panel.bl_description only tooltips a popover *button* that
        # opens this panel elsewhere — it does nothing for the sidebar's
        # own category tab strip (no public API covers that), so the
        # explanation lives here instead, as plain text. Split into
        # short lines since layout.label() never wraps on its own.
        info = layout.box()
        info.label(text="MD5 animation retargeting:", icon='INFO')
        info.label(text="snapshot rest pose, realign")
        info.label(text="bones after orientation edits,")
        info.label(text="retarget affected actions.")
        layout.separator(factor=0.3)

        # Step 1
        b1 = layout.box()
        b1.label(text="Step 1 — Snapshot Rest Pose", icon='ARMATURE_DATA')
        c1 = b1.column(align=True); c1.scale_y = 0.75
        c1.label(text="Run After importing animations.")
        c1.label(text="Run BEFORE editing bones.")
        c1.label(text="Records current rest matrices.")
        b1.operator("art.snapshot", icon='BOOKMARKS')
        if has_snap:
            snap = _art_get_snapshot(obj)
            b1.label(text=f"  Snapshot: {len(snap)} bones stored", icon='CHECKMARK')

        layout.separator(factor=0.3)

        # Step 2
        b2 = layout.box()
        b2.label(text="Step 2 — Edit Bone Orientations", icon='EDITMODE_HLT')
        c2 = b2.column(align=True); c2.scale_y = 0.75
        c2.label(text="Use the auto align functions below or")
        c2.label(text="to manually edit enter Edit Mode,")
        c2.label(text="rotate bones with R (pivot = active")
        c2.label(text="element).")
        c2.label(text="Do NOT change bone location.")
        c2.label(text="ONLY change orientation/length.")
        c2.label(text="Return to Object Mode when done.")
        if has_snap:
            b2.separator(factor=0.3)
            b2.prop(sp, "adjust_lengths")
            b2.prop(sp, "only_selected")
            in_edit = (ctx.mode == 'EDIT_ARMATURE')
            r2b = b2.row(); r2b.scale_y = 1.3
            r2b.enabled = in_edit
            op = r2b.operator("art.realign_bones", icon='BONE_DATA')
            op.adjust_lengths = sp.adjust_lengths
            op.only_selected  = sp.only_selected
            r2c = b2.row(); r2c.scale_y = 1.3
            r2c.enabled = in_edit
            op2 = r2c.operator("art.adjust_lengths", icon='FULLSCREEN_ENTER')
            op2.only_selected = sp.only_selected
            if not in_edit:
                b2.label(text="  (enter Edit Mode to enable)", icon='BLANK1')
        else:
            # Snapshot not yet taken — show buttons grayed out so user can see them
            b2.separator(factor=0.3)
            b2.prop(sp, "adjust_lengths")
            b2.prop(sp, "only_selected")
            col_dis = b2.column(align=True)
            col_dis.enabled = False
            col_dis.scale_y = 1.3
            col_dis.operator("art.realign_bones", icon='BONE_DATA')
            col_dis.operator("art.adjust_lengths", icon='FULLSCREEN_ENTER')
            b2.label(text="  (take snapshot first)", icon='BLANK1')

        layout.separator(factor=0.3)

        # Step 3
        b3 = layout.box()
        b3.label(text="Step 3 — Retarget Animation", icon='ACTION')
        c3 = b3.column(align=True); c3.scale_y = 0.75
        c3.label(text="(Must be in object mode.)")
        c3.label(text="Corrects keyframes in action(s)")
        c3.label(text="for new bone orientations.")
        c3.label(text="Original action(s) are backed up.")
        b3.prop(sp, "retarget_all")
        r3 = b3.row(); r3.scale_y = 1.6
        r3.enabled = has_snap
        r3.operator("art.retarget", icon='FILE_REFRESH')

        layout.separator(factor=0.3)

        # Utilities
        b4 = layout.box()
        b4.label(text="Utilities", icon='TOOL_SETTINGS')
        col4 = b4.column(align=True)
        row_bkup = col4.row()
        row_bkup.enabled = has_bkup
        row_bkup.operator("art.restore_backup", icon='LOOP_BACK')
        has_any_backup = any(a.name.endswith("_pre_retarget") for a in bpy.data.actions)
        row_del = col4.row()
        row_del.enabled = has_any_backup
        row_del.operator("art.delete_backups", icon='TRASH')
        row_snap = col4.row()
        row_snap.enabled = has_snap
        row_snap.operator("art.clear_snapshot", icon='X')

        if sp.status:
            layout.separator(factor=0.3)
            box = layout.box(); box.scale_y = 0.8
            msg   = sp.status
            first = True
            while msg:
                cut   = 44
                split = msg[:cut].rfind(' ') if len(msg) > cut else len(msg)
                split = split if split > 20 else cut
                box.label(text=msg[:split], icon='INFO' if first else 'BLANK1')
                msg   = msg[split:].lstrip()
                first = False


def register():
    # Action list classes must come before any CollectionProperty that uses them
    bpy.utils.register_class(MD5_ActionItem)
    bpy.utils.register_class(MD5_UL_actions)
    bpy.utils.register_class(EXPORT_OT_md5_actions_all)
    bpy.utils.register_class(EXPORT_OT_md5_actions_none)
    bpy.utils.register_class(EXPORT_OT_md5_actions_invert)
    bpy.types.Scene.md5_action_items = bpy.props.CollectionProperty(type=MD5_ActionItem)
    bpy.types.Scene.md5_action_list_index = bpy.props.IntProperty(default=0)
    bpy.utils.register_class(ART_Props)
    bpy.utils.register_class(ART_OT_Snapshot)
    bpy.utils.register_class(ART_OT_RealignBones)
    bpy.utils.register_class(ART_OT_AdjustLengths)
    bpy.utils.register_class(ART_OT_Retarget)
    bpy.utils.register_class(ART_OT_RestoreBackup)
    bpy.utils.register_class(ART_OT_DeleteBackups)
    bpy.utils.register_class(ART_OT_ClearSnapshot)
    bpy.utils.register_class(ART_PT_Panel)
    bpy.types.Scene.art = bpy.props.PointerProperty(type=ART_Props)
    bpy.utils.register_class(MD5BONES_OT_add_selected)
    bpy.utils.register_class(MD5BONES_OT_remove_selected)
    bpy.utils.register_class(MD5BONES_OT_replace_with_selected)
    bpy.utils.register_class(MD5BONES_OT_clear_all)
    bpy.utils.register_class(MD5BONES_OT_select_collection)
    bpy.utils.register_class(MD5BONES_PT_panel)
    bpy.utils.register_class(MD5_CameraActionItem)
    bpy.utils.register_class(MD5_UL_camera_actions)
    bpy.utils.register_class(EXPORT_OT_md5_cam_actions_all)
    bpy.utils.register_class(EXPORT_OT_md5_cam_actions_none)
    bpy.utils.register_class(EXPORT_OT_md5_cam_actions_invert)
    bpy.types.Scene.md5_cam_action_items = bpy.props.CollectionProperty(type=MD5_CameraActionItem)
    bpy.types.Scene.md5_cam_action_list_index = bpy.props.IntProperty(default=0)
    bpy.utils.register_class(IMPORT_OT_md5camera)
    bpy.utils.register_class(EXPORT_OT_md5camera)
    bpy.utils.register_class(IMPORT_OT_md5mesh)
    bpy.utils.register_class(IMPORT_OT_md5mesh_gate)
    bpy.utils.register_class(IMPORT_OT_md5anim)
    bpy.utils.register_class(EXPORT_OT_md5mesh)
    bpy.utils.register_class(EXPORT_OT_md5anim)
    bpy.utils.register_class(EXPORT_OT_md5mesh_anim)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import_camera)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import_mesh)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import_anim)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export_camera)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export_mesh)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export_anim)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export_mesh_anim)
    _register_shared_ui()


def unregister():
    _unregister_shared_ui()
    bpy.utils.unregister_class(ART_PT_Panel)
    bpy.utils.unregister_class(ART_OT_ClearSnapshot)
    bpy.utils.unregister_class(ART_OT_DeleteBackups)
    bpy.utils.unregister_class(ART_OT_RestoreBackup)
    bpy.utils.unregister_class(ART_OT_Retarget)
    bpy.utils.unregister_class(ART_OT_AdjustLengths)
    bpy.utils.unregister_class(ART_OT_RealignBones)
    bpy.utils.unregister_class(ART_OT_Snapshot)
    bpy.utils.unregister_class(ART_Props)
    try:
        del bpy.types.Scene.art
    except Exception:
        pass
    for _p in ('md5_action_items', 'md5_action_list_index'):
        try: delattr(bpy.types.Scene, _p)
        except Exception: pass
    bpy.utils.unregister_class(EXPORT_OT_md5_actions_none)
    bpy.utils.unregister_class(EXPORT_OT_md5_actions_invert)
    bpy.utils.unregister_class(EXPORT_OT_md5_actions_all)
    bpy.utils.unregister_class(MD5_UL_actions)
    bpy.utils.unregister_class(MD5_ActionItem)
    bpy.utils.unregister_class(MD5BONES_PT_panel)
    bpy.utils.unregister_class(MD5BONES_OT_select_collection)
    bpy.utils.unregister_class(MD5BONES_OT_clear_all)
    bpy.utils.unregister_class(MD5BONES_OT_replace_with_selected)
    bpy.utils.unregister_class(MD5BONES_OT_remove_selected)
    bpy.utils.unregister_class(MD5BONES_OT_add_selected)
    bpy.utils.unregister_class(EXPORT_OT_md5camera)
    bpy.utils.unregister_class(IMPORT_OT_md5camera)
    bpy.utils.unregister_class(EXPORT_OT_md5_cam_actions_invert)
    bpy.utils.unregister_class(EXPORT_OT_md5_cam_actions_none)
    bpy.utils.unregister_class(EXPORT_OT_md5_cam_actions_all)
    bpy.utils.unregister_class(MD5_UL_camera_actions)
    bpy.utils.unregister_class(MD5_CameraActionItem)
    for _p in ('md5_cam_action_items', 'md5_cam_action_list_index'):
        try: delattr(bpy.types.Scene, _p)
        except Exception: pass
    bpy.utils.unregister_class(IMPORT_OT_md5mesh_gate)
    bpy.utils.unregister_class(IMPORT_OT_md5mesh)
    bpy.utils.unregister_class(IMPORT_OT_md5anim)
    bpy.utils.unregister_class(EXPORT_OT_md5mesh)
    bpy.utils.unregister_class(EXPORT_OT_md5mesh_anim)
    bpy.utils.unregister_class(EXPORT_OT_md5anim)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import_camera)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import_mesh)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import_anim)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export_camera)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export_mesh)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export_mesh_anim)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export_anim)


if __name__ == "__main__":
    register()
