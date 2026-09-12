# =====================================================================
#  idTech4 .ase / .lwo Static Mesh Importer/Exporter for Blender 4.5
#  Standalone import AND export of idTech4 .ase (ASCII Scene Export) and
#  .lwo (Lightwave Object, LWO2) static meshes.
#
#  This addon is designed to install independently, but it also acts as
#  a small library for other idTech4 Blender addons — most notably
#  idTech4_map_io.py, which resolves "model" keys on .map
#  entities to .ase/.lwo files and needs to load them the exact same
#  way a standalone import would. Rather than duplicate the file-format
#  parsing in both addons, the .map importer detects at runtime whether
#  this addon is installed AND enabled and, only if so, calls this
#  module's load_model_meshes() directly. If this addon isn't present,
#  the .map importer simply disables its own static-model import
#  option instead of failing.
#
#  Version:  1.0.0
# =====================================================================

bl_info = {
    "name":        "idTech4 .ase / .lwo Importer/Exporter",
    "author":      "Samson & Claude Sonnet",
    "version":     (1, 0, 0),
    "blender":     (4, 5, 0),
    "location":    "File > Import/Export > idTech4 .ase / .lwo",
    "description": "Import/export idTech4 .ase / .lwo static meshes. Also "
                    "usable as a library by other idTech4 addons (e.g. the "
                    ".map importer) to resolve \"model\" references without "
                    "duplicating this format code.",
    "category":    "Import-Export",
}

import addon_utils
import base64
import bpy
import bmesh
import functools
import json
import math
import os
import re
import struct
import sys
import traceback
from mathutils import Vector, Matrix
from bpy.props import (StringProperty, BoolProperty, FloatProperty,
                       EnumProperty, PointerProperty)
from bpy.types import Operator, Panel, PropertyGroup
from bpy_extras.io_utils import ImportHelper, ExportHelper


# ─────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────
# idTech4 / Doom 3 world units are inches. Blender's default unit is
# meters. Duplicated from idTech4_map_io.py's own copy rather
# than imported from it — see the module docstring above for why each
# idTech4 Blender addon keeps its own copy of small shared constants
# instead of depending on another addon's internals.
MD5_SCALE_IN_TO_M = 0.0254                    # 1 inch     = 0.0254 m
SCALE             = 0.0254   # idTech4 game units (inches) -> metres

# Rotation presets applied to all positions/directions at import time.
# Angles are around the Z-axis (vertical), matching the common need to
# reorient models that were exported facing a different axis.
ROTATION_PRESETS = {
    'NONE':   None,                                          # 0 deg
    'X_TO_Y': Matrix.Rotation( math.radians( 90), 4, 'Z'),    # +90 deg Z
    'Y_TO_X': Matrix.Rotation( math.radians(-90), 4, 'Z'),    # -90 deg Z
    'R180':   Matrix.Rotation( math.radians(180), 4, 'Z'),    # 180 deg Z
}

IMPORT_ROTATION_ITEMS = [
    ('NONE',   '0°  : No rotation',      'No rotation applied'),
    ('X_TO_Y', '90°   : X to Y',          'Rotate 90° around Z (X-forward → Y-forward, idTech4 → Blender )'),
    ('Y_TO_X', '-90° : Y to X',          'Rotate -90° around Z (Y-forward → X-forward, Blender → idTech4 )'),
    ('R180',   '180° : Flip',            'Rotate 180° around Z'),
]

# Numeric-token regex reused across the ASE/LWO parsers below.
_NUM = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'


def _f(s):
    """Parse one numeric token; blank/garbage -> 0.0 rather than raising,
    matching how a malformed field should be tolerated, not fatal."""
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def _face_flat_normal(v0, v1, v2):
    """Standard CCW-winding triangle normal, used as a per-corner fallback
    wherever an explicit file-authored normal is missing for one corner
    of an otherwise normals-bearing face."""
    n = (v1 - v0).cross(v2 - v0)
    if n.length_squared > 0.0:
        n.normalize()
    return n


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
    """Shared Scale properties/UI for every import operator in this
    addon. Scaling is OFF by default (1:1, no change). When enabled,
    choose either an arbitrary numeric factor or the idTech4 -> Blender
    (inches -> meters) preset. Duplicated from idTech4_map_io.py's
    own copy — see the module docstring above."""

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

    def _draw_vertex_color_note(self, layout):
        """Say where a model's vertex colours end up.

        Worth stating in the dialog because the destination is not the
        only option Blender offers, and the choice is not arbitrary: .ase
        stores colour per face corner by construction (*MESH_CFACE
        indexes three entries per face) and .lwo VMAD is keyed by
        (polygon, point), so Face Corner is the only Blender domain that
        can hold either losslessly. The Vertex domain cannot express a
        colour that breaks at an edge, which both formats routinely
        carry."""
        layout.separator()
        box = layout.box()
        box.label(text="Vertex colors are mapped to Blender's", icon='INFO')
        box.label(text="Face Corner color attribute.")

    def get_scale(self):
        """Return the actual multiplier to apply to imported positions."""
        if self.scale_mode == 'NONE':
            return 1.0
        if self.scale_mode == 'FACTOR':
            return self.scale_factor
        return MD5_SCALE_IN_TO_M

    def get_rotation(self):
        return self.rotation


def get_or_create_material(name, cache):
    """Duplicated from idTech4_map_io.py's own copy — see the
    module docstring above."""
    if name in cache:
        return cache[name]
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
    cache[name] = mat
    return mat


# ─────────────────────────────────────────────────────────────────────
#  MATERIAL AUTO-GENERATION (optional — needs the companion "idTech4
#  Materials" addon). Duplicated from idTech4_map_io.py's own copy
#  — see the module docstring above for why each idTech4 Blender addon
#  keeps its own copy of small shared helpers instead of depending on
#  another addon's internals (this one is a one-way dependency in the
#  OTHER direction from the map importer's own use of THIS addon, so it
#  can't be imported from there either).
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
         when the imported file lives at .../base/models/foo.lwo).
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
    somewhere on *objects* (the blank placeholders get_or_create_material
    left behind). Returns (issues, built_count, report_name): issues is a
    list of human-readable issue strings (empty on full success);
    built_count is how many materials were actually built; report_name is
    the Text datablock publish_report() wrote, so the caller can NAME it in
    its own result. All three are ([], 0, None) if that addon isn't
    installed/enabled, either path is blank, or nothing ended up matching
    an .mtr source.

    That third value exists because the report was being written and never
    mentioned: the Report panel is default_closed, and the importer's own
    NOTICE lines come from summary.errors and so say nothing at all when
    every material builds cleanly - which is the normal case.

    *base_directory* and *source_path* are independent — base_directory
    is used only to resolve relative bitmap paths inside each .mtr decl;
    source_path is a .mtr file OR a directory searched recursively for
    .mtr files (material_addon.collect_mtr_files/parse_all_sources
    already handle either case — verified directly in that module, no
    file-vs-directory branching needed here). Callers resolve which
    values to pass in — either the shared idTech4 config file
    (get_shared_paths, edited from the Sources panel) or freshly derived
    from an imported file's location via resolve_materials_root — this
    function does no resolution/derivation of its own.

    Mirrors idTech4_map_io.py's own material step — scoped to
    *objects*' own material_slots rather than the whole Base directory's
    .mtr tree. Name normalisation, the case-insensitive fallback and the
    table priming all live behind the materials addon's public API now
    (API_VERSION 2); this only decides WHICH names to build.
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
    # is actually used under, and building through it fills THOSE rather than
    # a fresh datablock named after the decl. .ase-sourced material names keep
    # their source bitmap's file extension and .mtr decl names never have one,
    # so the two are routinely spelled differently for the same material.
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
    # exactly as that panel's own Generate Materials button does. This step
    # was simply missing: a model import built the materials and then dropped
    # the summary, so the panel sat on "No materials generated yet." beside a
    # scene full of freshly generated materials, and the only thing said
    # anywhere was the NOTICE lines below - which come from summary.errors and
    # so say nothing at all when every material builds cleanly.
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
    """Shared "auto-generate materials" properties/UI for every import
    operator in this addon — needs the companion "idTech4 Materials"
    addon; see generate_materials_for_objects. Mirrors the .map
    importer's own Import Materials / Material Mode / Game-Mod-Root
    controls, except defaulting to MAXIMUM mode here rather than SIMPLE
    (a single imported model is cheap enough to build at full fidelity,
    unlike a whole level's worth of materials)."""

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
        comment for why that's simpler here than mirroring ~10
        properties onto every gate class just to receive them).

        Uses type(self).bl_idname rather than self.bl_idname: on a live
        operator instance, Blender resolves the latter through RNA to
        the "IMPORT_SCENE_OT_..."-style identifier (no dot) instead of
        the plain dotted string set on the class, which breaks the
        split() below."""
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


def mark_smooth_for_custom_normals(mesh):
    """Put every face on a mesh into smooth fans, BEFORE its custom split
    normals are written. Both halves of that sentence matter.

    WHY AT ALL: a custom split normal fixes the SHADING normal, but not the
    TANGENT. Blender derives tangents with mikktspace over the mesh's smooth
    fans, and a face flagged sharp is a fan of one -- so a mesh left fully
    flat (which is what a builder produces by default, and what
    apply_mesh_smoothing then declines to touch) gets a per-FACE tangent
    basis. Normals stay perfectly smooth and the model still lights
    correctly bare, but the moment a bumpmap is applied the perturbation is
    expressed in a frame that jumps at every face boundary, and the surface
    reads as faceted. Measured on models/mapobjects/filler/cone.ASE: 100
    distinct tangents across 168 corners flat, 36 smooth.

    The engine has no equivalent split. R_DeriveTangents accumulates the
    tangent at each vertex and then R_CreateDupVerts/R_DeriveTangents sum it
    across every vertex sharing an exact position, so an idTech4 tangent
    frame is welded on position and never breaks at a face, which is why
    these models are smooth in-game and were faceted here.

    WHY BEFORE: Blender does not store a custom normal as a direction. It
    stores a compressed offset RELATIVE to the fan frame it was written in,
    so changing use_smooth afterwards re-decodes the same stored bytes
    against a different frame and silently yields different normals -- 43.9
    degrees mean, 71.3 worst on cone.ASE, and that is what the older
    "setting use_smooth DISCARDS custom normals" note was really observing.
    Written in this order the normals survive intact: cone.ASE lands 0.036
    degrees from an independent model of R_DeriveTangents either way.
    """
    if not mesh.polygons:
        return
    smooth = [True] * len(mesh.polygons)
    mesh.polygons.foreach_set('use_smooth', smooth)


#  A face's flat/smooth flag is not the same question as whether it SHADES
#  flat, and on an imported mesh it is not even an answer. Every mesh built
#  with Engine or File Shading now carries custom split normals and is
#  smooth-flagged throughout (mark_smooth_for_custom_normals), so the flag is
#  a constant there and the shading lives entirely in the normals. Ask the
#  normals instead: a face all of whose corner normals sit on its own
#  geometric normal is shading flat, whatever its flag says.
#
#  This is what the .lwo exporter's synthesised-SMGP path needs, since SMGP
#  islands can express exactly a per-face flat/smooth split and nothing finer.
#  Reading use_smooth there used to export EVERY Engine/File import fully
#  faceted -- correct only by accident for a faceted source, and wrong for
#  every smooth one.
#
#  The tolerance is angular and generous. Blender stores a custom normal
#  compressed, so a genuinely flat face's corners decode a hair off its face
#  normal (0.04 degrees mean, measured on low_rack.lwo, whose SMAN 0 makes it
#  fully faceted); real smoothing is orders of magnitude past that.
_SHADES_FLAT_COS = math.cos(math.radians(1.0))


def _faces_that_shade_smooth(mesh):
    """Per-polygon: does this face actually shade smooth? See above."""
    n = len(mesh.polygons)
    if not mesh.has_custom_normals:
        flags = [False] * n
        mesh.polygons.foreach_get('use_smooth', flags)
        return flags
    try:
        corner_normals = mesh.corner_normals
    except Exception:
        return [True] * n
    out = []
    for poly in mesh.polygons:
        fn = poly.normal
        smooth = False
        for li in range(poly.loop_start, poly.loop_start + poly.loop_total):
            if fn.dot(Vector(corner_normals[li].vector)) < _SHADES_FLAT_COS:
                smooth = True
                break
        out.append(smooth)
    return out


def mark_primary_color_attribute(mesh, name="Col"):
    """Name the bmesh-built colour layer as the mesh's active AND render one.

    bmesh and the mesh API disagree about this, and the difference is silent.
    mesh.color_attributes.new() sets active_color_name and default_color_name
    for you; a layer created as bm.loops.layers.float_color and carried
    through bm.to_mesh() arrives with BOTH empty and active_color_index at
    -1. The colours themselves are intact either way — the only thing missing
    is the pair of names that say which attribute to read.

    That pair is what a Vertex Color node with an empty layer_name resolves
    through, so with neither set it samples nothing and returns solid black,
    and every idTech4 `vertexColor` stage then multiplies its diffuse by
    zero. models/mapobjects/strogg/airdefense/terrain1_3.lwo carries a real
    822-point RGBA VMAP — 475 white and 299 black painted terrain weights —
    and still rendered black under textures/rock/sand01_to_skysand2.

    Left deliberately unconditional. A model with no RGBA VMAP at all gets
    this layer filled from its SURF chunk's own colour, which is what
    ConvertLWOToModelSurfaces seeds idDrawVert::color with before any vmap
    overrides it (Model.cpp), so marking it active is right there too — it
    is the engine's own default, not an invention.
    """
    if mesh.color_attributes.get(name) is None:
        return
    try:
        mesh.attributes.active_color_name = name
        mesh.attributes.default_color_name = name
    except (AttributeError, TypeError):        # older/newer API spelling
        pass


def apply_mesh_smoothing(mesh, smooth):
    """Set (or clear) the per-polygon smooth-shading flag on a built mesh
    — the same effect as Blender's Object > Shade Smooth.

    Skipped entirely for a mesh that already has custom split normals
    (from Engine/File Shading finding or deriving real normal data) —
    setting use_smooth on such a mesh AFTER normals_split_custom_set()
    changes the normals, because Blender stores a custom normal as an
    offset relative to the smooth-fan frame it was written in and this
    re-decodes it against a different one (43.9° mean on cone.ASE). The
    flag those meshes need is not this one's business anyway: the
    builders already set it themselves, in the only order that works —
    see mark_smooth_for_custom_normals, which is also where the reason
    they must be smooth at all (tangent continuity, not shading) is
    written down."""
    if mesh.has_custom_normals:
        return
    for poly in mesh.polygons:
        poly.use_smooth = smooth


# ─────────────────────────────────────────────────────────────────────
#  MATERIAL IDENTITY
#
#  "Which idTech4 material is this?" is stored in a different place by each
#  format, and neither is the one a DCC tool would think of:
#
#    .lwo  the SURF chunk's own NAME is the material name —
#          declManager->FindMaterial( lwoSurf->name )   (Model.cpp)
#    .ase  the *MATERIAL_NAME is IGNORED. The name comes from
#          *MAP_DIFFUSE -> *BITMAP, run through OSPathToRelativePath
#          (Model_ase.cpp ASE_KeyMAP_DIFFUSE)
#
#  Both then go through the same decl canonicaliser. Ports of all three
#  engine rules follow.
# ─────────────────────────────────────────────────────────────────────

BASE_GAMEDIR = 'base'

# POLS chunk types this importer turns into real Blender faces, in the order
# their index is stored in the "idtech4_poly_type" face attribute. FACE is
# ordinary geometry; PTCH (LightWave subpatch/Metaform) and SUBD
# (Catmull-Clark) store a control CAGE whose limit surface LightWave computes
# at display time and the engine never does.
LWO_POLY_TYPES = (b'FACE', b'PTCH', b'SUBD')
LWO_POLY_TYPE_ATTR = 'idtech4_poly_type'
LWO_SMGP_ATTR = 'idtech4_smoothing_group'
LWO_PART_ATTR = 'idtech4_part'
LWO_SMAN_PROP = 'idtech4_sman'
ASE_SMOOTH_ATTR = 'idtech4_ase_smoothing'
ASE_EDGE_VIS_ATTR = 'idtech4_ase_edge_visible'
ASE_NODE_TM_PROP = 'idtech4_node_tm'
# Verbatim bytes of everything the importer does not model, so an export can
# put it back. See build_lwo_export_data for the safety gate on re-emission.
LWO_ESCROW_PROP = 'idtech4_lwo_escrow'
LWO_ESCROW_SHAPE_PROP = 'idtech4_lwo_escrow_shape'
LWO_SURF_EXTRA_PROP = 'idtech4_surf_extra'
LWO_COLR_PROP = 'idtech4_colr'
# Polygons Blender structurally cannot represent (a second face on the same
# vertex set). Kept verbatim so a faithful export can put them back.
# Per-vertex: the file point a split copy came from, or -1 for an ordinary
# vertex. See the archived changelog under docs/changelog-archive/.
LWO_SOURCE_POINT_ATTR = 'idtech4_source_point'


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


def engine_os_path_to_relative(os_path, mod_dirs=()):
    """idFileSystemLocal::OSPathToRelativePath (framework/FileSystem.cpp).

    "takes a full OS path, as might be found in data from a media creation
    program, and converts it to a relativePath by stripping off directories"

    Looks for the first path segment exactly equal to "base" — delimited by a
    slash on BOTH sides, which is why a bare substring search is wrong — then
    falls back to each mod dir (the engine tries fs_game then fs_game_base).
    Returns (relative_path, anchor_segment), or (None, None) when the path is
    not anchored anywhere, which is exactly the signal that a model was not
    authored for idTech4.
    """
    p = os_path.replace('\\', '/')
    for anchor in (BASE_GAMEDIR,) + tuple(mod_dirs):
        if not anchor:
            continue
        start = 0
        while True:
            idx = p.lower().find(anchor.lower(), start)
            if idx == -1:
                break
            before_ok = idx == 0 or p[idx - 1] == '/'
            after = idx + len(anchor)
            after_ok = after < len(p) and p[after] == '/'
            if before_ok and after_ok:
                return p[after + 1:], anchor
            start = idx + 1
    return None, None


# The top-level folders an idTech4 asset tree is built from. Used ONLY to
# recover a path the engine's own anchor rule cannot reach -- see below.
IDTECH4_ASSET_ROOTS = frozenset((
    'models', 'textures', 'materials', 'guis', 'maps', 'def', 'skins',
    'particles', 'fx', 'sound', 'video', 'music', 'env', 'script', 'af',
    'dds', 'gui', 'sounds'))


def asset_root_relative(os_path):
    """Recover a qpath from an authoring path the engine's rule cannot anchor.

    OSPathToRelativePath only knows "base" plus whatever fs_game/fs_game_base
    happen to be at runtime. A Quake 4 .ase says
    "C:\\Ritual\\Q4Ritual\\game\\q4base\\textures\\..."; Prey and Dark Mod assets
    name their own mod folders. An importer cannot know which -- the model file
    carries no record of the game it belongs to, and a local copy of the tree
    is frequently not even named after it (this repo keeps Quake 4 under
    "quake4/base" while its assets all say "q4base").

    So when the engine rule fails, fall back on the shape of the tree itself:
    an idTech4 qpath always starts at one of a small set of top-level folders.
    Take everything from the FIRST such component. Across the five game corpora
    that recovers 70 of the 100 .ase bitmaps "base" cannot anchor (52 under
    textures/, 18 under models/) and leaves the other 30 -- which have no asset
    root anywhere in them -- correctly unresolved.

    Deliberately NOT folded into engine_os_path_to_relative. That function
    models the engine exactly and is used to PREDICT what the engine will do
    with a string we are about to write, so it has to stay honest. This is a
    best-effort recovery of authoring intent, which is a different question and
    is only ever asked on import.
    """
    parts = [p for p in (os_path or '').replace('\\', '/').split('/') if p]
    for i, part in enumerate(parts):
        if part.lower() in IDTECH4_ASSET_ROOTS:
            return '/'.join(parts[i:])
    return None


def engine_ase_bitmap_decl(bitmap, mod_dirs=()):
    """The decl the engine ends up looking up for an .ase *MATERIAL.

    ASE_KeyMAP_DIFFUSE (renderer/Model_ase.cpp) strips the quotes, runs
    BackSlashesToSlashes, then OSPathToRelativePath -- so a *BITMAP is an OS
    path that gets cut down to a qpath BEFORE FindMaterial canonicalises it.
    Returns '' for a path anchored nowhere, which is what the engine resolves
    (and then substitutes "_emptyName" for).
    """
    rel, _anchor = engine_os_path_to_relative(bitmap or '', mod_dirs)
    if rel is None:
        return ''
    return engine_canonical_decl(rel)


def engine_lwo_surf_decl(surf_name):
    """The decl the engine ends up looking up for an .lwo SURF.

    ConvertLWOToModelSurfaces (renderer/Model.cpp) hands the surface name
    straight to FindMaterial -- there is NO OSPathToRelativePath pass. That
    asymmetry with .ase is the whole reason both exporters have to check what
    they are about to write instead of copying a string across: an absolute
    authoring path is a perfectly good *BITMAP and a broken SURF name.
    """
    return engine_canonical_decl(surf_name or '')


# Blender appends ".001", ".002" ... to a duplicate datablock name. Strip it
# before treating a material name as a decl: MakeNameCanonical's
# truncate-at-the-last-dot happens to remove a bare ".001", but NOT one on a
# name that already carries an extension ("thing.tga.001" -> "thing.tga"), so
# leaning on the engine for this would be luck rather than a rule.
_BLENDER_UNIQUIFIER_RE = re.compile(r'\.\d{3}$')


def strip_blender_uniquifier(name):
    return _BLENDER_UNIQUIFIER_RE.sub('', name or '')


# How confident we are that a material names a real engine decl.
#   ENGINE   positively identified -- an .ase *BITMAP anchored under base/, or
#            an .lwo surface named with a decl-shaped path
#   FOREIGN  positively ruled out -- an .ase whose *BITMAP is not anchored
#            anywhere, so the engine would resolve nonsense from it
#   UNKNOWN  no signal either way. An .lwo surface name IS the decl as far as
#            the engine is concerned (FindMaterial is handed it verbatim), so
#            a bare name like "gizmo1" may well be a real short decl -- the
#            renderbump test asset's own material is exactly that. Nothing in
#            the file can settle it; only a .mtr lookup can.
#
# Material auto-generation should skip FOREIGN and attempt ENGINE and UNKNOWN,
# letting a failed .mtr lookup be the thing that settles UNKNOWN.
MAT_ENGINE = 'ENGINE'
MAT_FOREIGN = 'FOREIGN'
MAT_UNKNOWN = 'UNKNOWN'

_COMMON_DCC_SURFACE_NAMES = frozenset((
    'default', 'material', 'none', 'untitled', 'surface', 'lambert1',
    'standardsurface1', 'defaultmat', 'initialshadinggroup',
))


def _lwo_surface_kind(name):
    """Classify a bare .lwo SURF name. Path-shaped is a positive signal; a
    DCC tool's stock name is a negative one; anything else is genuinely
    undecidable from the file alone."""
    n = name.replace('\\', '/').strip()
    if '/' in n:
        return MAT_ENGINE
    base = n.lower().split('.')[0]
    if not n or base in _COMMON_DCC_SURFACE_NAMES:
        return MAT_FOREIGN
    return MAT_UNKNOWN


# ─────────────────────────────────────────────────────────────────────
#  SHADING MODES
#
#  Four ways to shade an imported model, three of which are distinct for a
#  .lwo and all four for some .ase files:
#
#    ENGINE  what idTech4 will actually render
#    FILE    what the file / its DCC intends
#    SMOOTH  no custom normals, Blender's own averaging
#    FLAT    no custom normals, faceted
#
#  ENGINE and FILE only diverge where the engine ignores something the file
#  says. That happens in exactly three places, all verified against the GPL
#  source:
#
#    .ase with no *MESH_NORMALS   the engine regenerates from geometry
#                                 (normalsParsed false -> R_DeriveTangents)
#                                 and never reads *MESH_SMOOTHING at all --
#                                 there is no reference to it anywhere in
#                                 Model_ase.cpp. FILE honours those groups.
#    renderBump material          "completely ignore any explict normals on
#                                 surfaces with a renderbump command"
#    unsmoothedTangents material  R_DeriveUnsmoothedTangents, which takes each
#                                 vertex normal from ONE dominant triangle
#
#  For .lwo the two are always identical: LWO stores no normals, so SMAN plus
#  SMGP IS the file's shading data and also what the engine computes from.
# ─────────────────────────────────────────────────────────────────────

SHADING_ENGINE = 'ENGINE'
SHADING_FILE = 'FILE'
SHADING_SMOOTH = 'SMOOTH'
SHADING_FLAT = 'FLAT'

SHADING_ITEMS = [
    (SHADING_FILE, "File Shading",
     "Shade the way the file intends, which is what a non-idTech4 model "
     "wants. Identical to Engine Shading for every .lwo -- LWO stores no "
     "normals, so its smoothing angle is both -- and for any .ase that has "
     "*MESH_NORMALS. Differs only for an .ase without them, where this "
     "honours the *MESH_SMOOTHING groups the engine throws away"),
    (SHADING_ENGINE, "Engine Shading",
     "Shade the way idTech4 will actually render it. For a .lwo that is the "
     "file's SURF SMAN angle and PTAG SMGP groups (lwGetVertNormals); for an "
     ".ase it is *MESH_NORMALS, or -- when the file has none -- the normals "
     "the engine regenerates from geometry, which ignore *MESH_SMOOTHING "
     "entirely. With material generation on, a renderBump or "
     "unsmoothedTangents material is honoured too"),
    (SHADING_SMOOTH, "Smooth",
     "No custom normals: every face smooth-shaded with Blender's own "
     "averaging. Matches no idTech4 asset -- Blender weights by corner angle "
     "and crosses material boundaries, the engine does neither -- but it is "
     "the only way to get an unconstrained mesh, since the two modes above "
     "write custom split normals that override Blender's shading controls"),
    (SHADING_FLAT, "Flat",
     "No custom normals, every face faceted. Same reasoning as Smooth: with "
     "custom normals applied you would otherwise have to clear them by hand "
     "to get here"),
]


def resolve_shading_mode(shading):
    """Normalise a shading mode. Every caller names one; an omitted or
    empty value means Engine Shading, which is what the engine itself
    does and what this importer has always defaulted to."""
    return shading or SHADING_ENGINE


# ---------------------------------------------------------------- .mtr seam

MTR_RENDERBUMP = 'renderbump'
MTR_UNSMOOTHED = 'unsmoothedtangents'
_MTR_SHADING_KEYWORDS = (MTR_RENDERBUMP, MTR_UNSMOOTHED)


def _mtr_strip_comments(text):
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == '/' and i + 1 < n and text[i + 1] == '/':
            j = text.find('\n', i)
            i = n if j == -1 else j
        elif c == '/' and i + 1 < n and text[i + 1] == '*':
            j = text.find('*/', i + 2)
            i = n if j == -1 else j + 2
        else:
            out.append(c)
            i += 1
    return ''.join(out)


# Keyed on the absolute source path. Without it a whole-.map import would
# re-read every .mtr once per placed model; with it, once per import. The
# trade is that editing a .mtr mid-session needs a Blender restart (or a
# reload of this addon) before the change is seen here - acceptable for two
# keywords that decide vertex normals.
_MTR_FLAG_CACHE = {}


def _mtr_shading_flags_via_parser(path):
    """Ask idTech4_material_import, or None if it cannot answer.

    One call to the materials addon's public scan_keywords() (API_VERSION 2),
    which makes a single pass over the whole tree and caches it per source.
    """
    addon = get_material_import_addon()
    if addon is None:
        return None
    scan = getattr(addon, 'scan_keywords', None)
    if scan is None:
        return None
    try:
        hits = scan((MTR_RENDERBUMP, MTR_UNSMOOTHED), path, None)
    except Exception:
        return None
    if not hits:
        return None
    found = {}
    for name, keywords in hits.items():
        found.setdefault(engine_canonical_decl(name),
                         set()).update(keywords)
    return found


def scan_mtr_shading_flags(source_path):
    """Which materials carry a keyword that changes how the ENGINE shades.

    ---------------------------------------------------------------------
    THIS IS THE SEAM: the only place this importer asks the material system
    anything. It used to be a purpose-built two-keyword scan because the
    shared parser could not answer correctly - unsmoothedTangents was not in
    its valueless-flag list so it swallowed the following token, and
    renderbump takes the whole rest of the line (ParseRestOfLine), which a
    key/value parser does not reproduce.

    That parser has since been rewritten and handles both: verified against
    Doom 3's tree, 759 materials with renderbump and 178 with
    unsmoothedTangents, including the ones carrying both. So the body now
    calls the materials addon's scan_keywords() - one pass over the tree,
    cached per source - and the hand-rolled scan below survives only as the
    fallback for when that addon is absent or disabled, since the standalone
    model importer can still be used without it.
    ---------------------------------------------------------------------

    Returns {canonical decl name: set of keywords}, empty when the path holds
    no .mtr files or cannot be read.

    *source_path* may be several paths rather than one - a Mod Base makes the
    derived materials source one folder per root. They are scanned in priority
    order and the FIRST root to declare a name wins outright, rather than the
    keyword sets being unioned: a mod that redeclares a material without
    renderbump means it, and merging the base's copy back in would shade the
    mod's model by a keyword its own decl dropped.
    """
    if isinstance(source_path, (list, tuple)):
        merged = {}
        for one in source_path:
            for name, flags in scan_mtr_shading_flags(one).items():
                merged.setdefault(name, flags)
        return merged

    found = {}
    path = bpy.path.abspath(source_path) if source_path else ''
    if not path or not os.path.exists(path):
        return found
    cache_key = os.path.normpath(path).lower()
    cached = _MTR_FLAG_CACHE.get(cache_key)
    if cached is not None:
        return cached
    parsed = _mtr_shading_flags_via_parser(path)
    if parsed is not None:
        _MTR_FLAG_CACHE[cache_key] = parsed
        return parsed
    files = []
    if os.path.isfile(path):
        if path.lower().endswith('.mtr'):
            files = [path]
    else:
        for root, _dirs, names in os.walk(path):
            for nm in names:
                if nm.lower().endswith('.mtr'):
                    files.append(os.path.join(root, nm))

    for mtr in files:
        try:
            with open(mtr, 'r', errors='replace') as fh:
                text = _mtr_strip_comments(fh.read())
        except OSError:
            continue
        # brace-delimited blocks; only depth 1 keywords belong to the material
        toks = text.replace('{', ' { ').replace('}', ' } ').split()
        depth, name, flags = 0, None, set()
        i, n = 0, len(toks)
        while i < n:
            t = toks[i]
            if t == '{':
                depth += 1
            elif t == '}':
                depth -= 1
                if depth == 0 and name:
                    if flags:
                        found.setdefault(engine_canonical_decl(name),
                                         set()).update(flags)
                    name, flags = None, set()
            elif depth == 0:
                if t.lower() != 'table':
                    name = t
                else:
                    i += 1          # skip the table's name
            elif depth == 1 and t.lower() in _MTR_SHADING_KEYWORDS:
                flags.add(t.lower())
            i += 1
    _MTR_FLAG_CACHE[cache_key] = found
    return found


def material_shading_overrides(decl_names, source_path):
    """(renderbump, unsmoothedTangents) sets, for the decls actually used.

    Both make the engine disregard the model's own normals, so Engine Shading
    has to follow suit. Returns empty sets when no .mtr source is available,
    in which case Engine Shading simply cannot know -- the model file carries
    no hint of either keyword.
    """
    flags = scan_mtr_shading_flags(source_path)
    if not flags:
        return set(), set()
    want = {engine_canonical_decl(d) for d in decl_names}
    rb = {d for d in want if MTR_RENDERBUMP in flags.get(d, ())}
    un = {d for d in want if MTR_UNSMOOTHED in flags.get(d, ())}
    return rb, un


# ------------------------------------------------- engine normal generation

def engine_regenerated_normals(positions, tris):
    """R_DeriveTangents' normals: what the engine computes when it has none.

    Sums the UNIT normal of every triangle at each vertex -- unweighted, which
    is not what Blender does (it weights by corner angle) -- then welds across
    vertices at an identical position, which is the dupVerts pass. Callers
    pass one material's triangles at a time, because a srfTriangles_t is one
    material and the sum never crosses that boundary.

    *tris* is a list of (i0, i1, i2) into *positions*. Returns one normal per
    corner, in the same order.
    """
    acc = {}
    face_n = []
    for (i0, i1, i2) in tris:
        a, b, c = positions[i0], positions[i1], positions[i2]
        # the engine's winding: n = (c - a) x (b - a)
        n = (c - a).cross(b - a)
        if n.length_squared > 0.0:
            n = n.normalized()
        face_n.append(n)
        for idx in (i0, i1, i2):
            key = _position_key(positions[idx])
            if key in acc:
                acc[key] = acc[key] + n
            else:
                acc[key] = n.copy()
    out = []
    for t, (i0, i1, i2) in enumerate(tris):
        for idx in (i0, i1, i2):
            v = acc[_position_key(positions[idx])]
            out.append(v.normalized() if v.length_squared > 0.0 else face_n[t])
    return out


def engine_dominant_normals(positions, tris):
    """R_DeriveUnsmoothedTangents' normals, for an unsmoothedTangents material.

    R_BuildDominantTris picks, for each vertex, the incident triangle with the
    LARGEST area, and R_DeriveUnsmoothedTangents then takes that one
    triangle's normal as the vertex normal -- so nothing is averaged at all.
    """
    best = {}
    face_n = []
    for t, (i0, i1, i2) in enumerate(tris):
        a, b, c = positions[i0], positions[i1], positions[i2]
        raw = (c - a).cross(b - a)
        area = raw.length            # 2x the triangle area, as the engine uses
        n = raw.normalized() if area > 0.0 else raw.copy()
        face_n.append(n)
        for idx in (i0, i1, i2):
            key = _position_key(positions[idx])
            if key not in best or area > best[key][0]:
                best[key] = (area, n)
    out = []
    for t, (i0, i1, i2) in enumerate(tris):
        for idx in (i0, i1, i2):
            out.append(best[_position_key(positions[idx])][1])
    return out


def _position_key(v):
    """Exact-position weld key, matching R_CreateSilRemap's exact xyz compare."""
    return (v.x, v.y, v.z)


def disambiguate_surface_names(identities):
    """Keep two SURF chunks apart when they canonicalise to one decl name.

    MakeNameCanonical truncates at the last dot, so "wood1" and "wood1.tga"
    are one decl -- and the engine really does resolve both to one material.
    But they remain two SURF chunks with their own SMAN, and lwGetVertNormals
    shades from the SURF, not from the material. Collapsing them onto one
    Blender datablock therefore throws away one of the two smoothing angles.

    Only the colliding names are changed, and only to their own raw form, so
    the ordinary case still gets a material named exactly after its decl.

    Two tags with the SAME raw name are not this case and must not be split.
    TAGS is a plain string list and a file may simply list a name twice --
    5 of the 717 shipped .lwo assets do, including mapobjects/filler/laptop --
    but there is only ONE SURF chunk behind them, so there is no second
    smoothing angle to preserve and nothing to keep apart. Splitting them
    invented `models/mapobjects/filler/laptop.002`, a name no .mtr declares,
    which then could not be built: the material generator canonicalised it
    back to `...laptop`, built THAT as a separate datablock, and left the
    mesh pointing at the unbuilt one.
    """
    groups = {}
    for ident in identities:
        groups.setdefault(ident['display'], []).append(ident)
    for disp, group in groups.items():
        distinct_raw = {ident.get('raw') for ident in group}
        if len(group) < 2 or len(distinct_raw) < 2:
            continue
        used = set()
        for ident in group:
            base = ident.get('raw') or disp
            name, n = base, 2
            while name in used:
                name = '%s.%03d' % (base, n)
                n += 1
            used.add(name)
            ident['display'] = name
    return identities


def material_identity_from_lwo(surf_name):
    """Identity for one .lwo SURF.

    The engine hands the surface name straight to FindMaterial, so the name IS
    the decl -- there is no second place to look and therefore no way to prove
    a bare name is not one. See _lwo_surface_kind for how that is graded.

    That is also why there is NO OSPathToRelativePath pass here, unlike the
    .ase path. ConvertLWOToModelSurfaces (renderer/Model.cpp:1076) does not
    strip a SURF name, so neither can we: running the strip anyway truncated
    any decl containing its own complete "base" component, which is real --
    "models/mapobjects/base/chairs/chair1" became "chairs/chair1", a decl that
    does not exist, and chair1.lwo re-exported to an invisible model. Across
    all five game corpora that call had no upside to weigh against it: of
    22201 SURF names, none is an OS path and exactly one contains a "base"
    component -- the one it broke.
    """
    raw = (surf_name or '').strip()
    body = raw
    kind = _lwo_surface_kind(body)
    decl = engine_canonical_decl(body) if kind != MAT_FOREIGN else ''
    return dict(raw=raw, decl=decl,
                origin='LWO_SURF' if kind != MAT_FOREIGN else 'NONE',
                kind=kind, engine=(kind == MAT_ENGINE),
                display=decl if kind != MAT_FOREIGN else (raw or 'unknown'))


def material_identity_from_ase(bitmap, material_name, mod_dirs=()):
    """Identity for one .ase *MATERIAL.

    The engine reads *BITMAP and ignores *MATERIAL_NAME entirely, so a bitmap
    anchored under base/ (or a mod dir) is authoritative. A bitmap that is not
    anchored means the file came from somewhere other than an idTech4
    pipeline — then *MATERIAL_NAME is the only human-meaningful name, and the
    material is flagged foreign so nothing tries to generate a shader for it.
    """
    raw = (bitmap or '').strip().strip('"')
    if raw:
        rel, _anchor = engine_os_path_to_relative(raw, mod_dirs)
        if rel is None:
            # Not anchored on "base" or any mod dir we were told about. That is
            # the normal case for every game other than Doom 3 -- a Quake 4
            # bitmap is rooted at "q4base", Prey's at "preybase" -- and none of
            # those names reach us. Recover from the asset tree's own shape
            # instead of grading a perfectly good material path FOREIGN and
            # falling back to the 3ds Max material name ("rock", "07"), which
            # is what used to be written back out and could never resolve.
            rel = asset_root_relative(raw)
        if rel is not None:
            decl = engine_canonical_decl(rel)
            return dict(raw=raw, decl=decl, origin='ASE_BITMAP',
                        kind=MAT_ENGINE, engine=True, display=decl)
    # An .ase bitmap that is not anchored anywhere is a positive signal the
    # other way: the engine would canonicalise a stray authoring path into a
    # decl name that cannot exist.
    name = (material_name or '').strip() or (raw or 'unknown')
    return dict(raw=raw, decl='', origin='NONE', kind=MAT_FOREIGN,
                engine=False, display=name)


_IDENTITY_KEYS = ('idtech4_decl', 'idtech4_origin', 'idtech4_raw',
                  'idtech4_material_kind', 'idtech4_is_engine_material')


MAT_IDENTITY_KEY = 'idtech4_mat_identity'


def stash_material_identities(mesh, idents):
    """Park per-slot identity on the mesh, in material-slot order.

    load_model_meshes' contract is [(name, mesh, [material_name, ...])] and
    idTech4_map_io.py unpacks exactly that, so identity rides along on the
    mesh datablock rather than widening the tuple. import_model_file lifts it
    off again once it has created the real materials.
    """
    if idents:
        mesh[MAT_IDENTITY_KEY] = json.dumps(idents)


def take_material_identities(mesh):
    """Pop what stash_material_identities parked, if anything."""
    raw = mesh.get(MAT_IDENTITY_KEY)
    if not raw:
        return []
    try:
        out = json.loads(raw)
    except Exception:
        out = []
    try:
        del mesh[MAT_IDENTITY_KEY]
    except Exception:
        pass
    return out


def apply_material_identity(mat, ident):
    """Record identity on the Blender material.

    The datablock NAME is the decl for an engine material (so it matches the
    .mtr declaration, which is what makes it findable), but the name is not
    the identity: Blender appends ".001" to a duplicate, and two models can
    legitimately share a decl. Export reads these properties instead.
    """
    if mat is None or not ident:
        return
    mat['idtech4_decl'] = ident.get('decl', '')
    mat['idtech4_origin'] = ident.get('origin', 'NONE')
    mat['idtech4_raw'] = ident.get('raw', '')
    mat['idtech4_material_kind'] = ident.get('kind', MAT_UNKNOWN)
    if ident.get('surf_extra'):
        mat[LWO_SURF_EXTRA_PROP] = ident['surf_extra']
    if ident.get('colr') is not None:
        mat[LWO_COLR_PROP] = list(ident['colr'])
    if ident.get('sman') is not None:
        # The surface's own LightWave smoothing angle, in radians. The engine
        # recomputes all shading from this plus PTAG SMGP, so it is the only
        # shading input a .lwo actually carries.
        mat[LWO_SMAN_PROP] = float(ident['sman'])
    # Kept as a convenience for callers that only want a yes/no; FOREIGN is
    # the only value that should stop material generation outright.
    mat['idtech4_is_engine_material'] = (
        ident.get('kind', MAT_UNKNOWN) != MAT_FOREIGN)


def material_decl_for_export(mat, fallback):
    """The decl this Blender material DENOTES -- the .mtr declaration the
    engine has to end up looking up when a model carrying it is loaded.

    This is an intent, not a string to write. What actually goes in the file is
    chosen per format by material_source_string(), because the two formats
    reach FindMaterial by different routes and the same characters do not mean
    the same thing in both.

    The datablock NAME wins. Assigning a material is how a user says which
    declaration a surface should use, so a rename has to be honoured rather
    than silently discarded; an import left alone already carries the decl as
    its name, so this reproduces it unchanged. Blender's ".001" duplicate
    suffix is stripped first -- it is an artifact of the datablock namespace,
    never part of what the user meant.

    idtech4_decl is the fallback for a material whose name canonicalises to
    nothing at all.

    What this deliberately no longer does is return idtech4_raw. Raw is the
    verbatim source string, and for an .ase that is an absolute authoring path
    from the modelling package ("\\\\purgatory\\...\\phone.tga"), not a decl.
    Returning it made every caller treat an OS path as though it were already
    relative: the .ase exporter prefixed its own "\\base\\" onto it, giving a
    doubly-rooted path that OSPathToRelativePath truncated at OUR marker and
    left the original absolute path behind, and the .lwo exporter wrote it as a
    SURF name where nothing strips it at all. Both resolved to a decl that
    cannot exist, and an unresolvable decl is invisible in game rather than
    obviously broken (see engine_ase_bitmap_decl and the implicit-material path
    in idMaterial::SetDefaultText).
    """
    if mat is not None:
        decl = engine_canonical_decl(strip_blender_uniquifier(mat.name or ''))
        if decl:
            return decl
        stored = mat.get('idtech4_decl') or ''
        if stored:
            return stored
    return engine_canonical_decl(fallback) or fallback


def material_source_string(mat, decl, resolver, wrap=None):
    """The exact string to write into a file so the engine resolves *decl*.

    Candidates are tried in order and each one is CHECKED, by running it back
    through *resolver* -- the same function that models what the engine will do
    with that field in that format. The first candidate that actually resolves
    to *decl* wins; nothing is written on the assumption that it round-trips.

      1. idtech4_raw, the verbatim source string. Keeping it makes a re-export
         of a shipped asset byte-identical, which is what lets a diff against
         the original show only real changes. It is only ever used when it
         demonstrably still means the right thing -- so it survives an
         untouched round trip and is dropped the moment the user renames the
         material or converts to the other format.
      2. the datablock name, which preserves the author's capitalisation and
         extension where canonicalisation would have flattened them.
      3. *wrap* applied to the decl, or the bare decl. This is the answer that
         is correct by construction, so it is the floor rather than the goal.

    *wrap* exists for .ase, where a relative decl needs the synthetic "\\base\\"
    root to survive OSPathToRelativePath. It is applied ONLY to candidate 3: a
    raw OS path already carries its own anchor, and prefixing one that does is
    precisely the bug this function exists to make impossible.
    """
    if mat is not None and decl:
        for cand in (mat.get('idtech4_raw') or '',
                     strip_blender_uniquifier(mat.name or '')):
            if cand and resolver(cand) == decl:
                return cand
    return wrap(decl) if wrap else decl


# ── ASE (ASCII Scene Export) ────────────────────────────────────────

_ASE_RE_GEOM      = re.compile(r'^\*GEOMOBJECT\s*\{')
_ASE_RE_MATLIST   = re.compile(r'^\*MATERIAL_LIST\b')
# The index token after *MATERIAL / *SUBMATERIAL is matched but NOT trusted:
# both are keyed by encounter order, the way ASE_KeyMATERIAL_LIST Append()s
# them. It is not always a number. Dark Mod's pot_ceramic_lid.ase ships
# "*MATERIAL o {" -- the letter o -- which a \d+ pattern skips entirely,
# dropping the material and shifting every later *MATERIAL_REF onto the wrong
# one. The engine never looks at the token at all, so neither do we.
_ASE_RE_MATERIAL  = re.compile(r'^\*MATERIAL\s+(\S+)\s*\{')
_ASE_RE_SUBMAT    = re.compile(r'^\*SUBMATERIAL\s+(\S+)\s*\{')
_ASE_RE_MATNAME   = re.compile(r'^\*MATERIAL_NAME\s+"([^"]*)"')
_ASE_RE_BITMAP    = re.compile(r'^\*BITMAP\s+"([^"]*)"')
_ASE_RE_NODENAME  = re.compile(r'^\*NODE_NAME\s+"([^"]*)"')
_ASE_RE_MATREF    = re.compile(r'^\*MATERIAL_REF\s+(\d+)')
_ASE_RE_VERT      = re.compile(r'^\*MESH_VERTEX\s+(\d+)\s+(' + _NUM + r')\s+(' + _NUM + r')\s+(' + _NUM + r')')
_ASE_RE_FACE      = re.compile(r'^\*MESH_FACE\s+(\d+):\s*A:\s*(\d+)\s+B:\s*(\d+)\s+C:\s*(\d+)')
_ASE_RE_MTLID     = re.compile(r'\*MESH_MTLID\s+(\d+)')
_ASE_RE_SMOOTHING = re.compile(r'\*MESH_SMOOTHING\s+([0-9,\s]*)')
_ASE_RE_EDGEVIS   = re.compile(r'\bAB:\s*(\d+)\s+BC:\s*(\d+)\s+CA:\s*(\d+)')
_ASE_RE_TMROW     = re.compile(r'^\*TM_ROW(\d)\s+(' + _NUM + r')\s+(' + _NUM +
                               r')\s+(' + _NUM + r')')
_ASE_RE_TVERT     = re.compile(r'^\*MESH_TVERT\s+(\d+)\s+(' + _NUM + r')\s+(' + _NUM + r')')
_ASE_RE_TFACE     = re.compile(r'^\*MESH_TFACE\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)')
_ASE_RE_CVERT     = re.compile(r'^\*MESH_VERTCOL\s+(\d+)\s+(' + _NUM + r')\s+(' + _NUM + r')\s+(' + _NUM + r')')
_ASE_RE_CFACE     = re.compile(r'^\*MESH_CFACE\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)')
_ASE_RE_FACENRM   = re.compile(r'^\*MESH_FACENORMAL\s+(\d+)\s+(' + _NUM + r')\s+(' + _NUM + r')\s+(' + _NUM + r')')
_ASE_RE_VERTNRM   = re.compile(r'^\*MESH_VERTEXNORMAL\s+(\d+)\s+(' + _NUM + r')\s+(' + _NUM + r')\s+(' + _NUM + r')')
_ASE_RE_UOFFSET   = re.compile(r'^\*UVW_U_OFFSET\s+(' + _NUM + r')')
_ASE_RE_VOFFSET   = re.compile(r'^\*UVW_V_OFFSET\s+(' + _NUM + r')')
_ASE_RE_UTILING   = re.compile(r'^\*UVW_U_TILING\s+(' + _NUM + r')')
_ASE_RE_VTILING   = re.compile(r'^\*UVW_V_TILING\s+(' + _NUM + r')')
_ASE_RE_UANGLE    = re.compile(r'^\*UVW_ANGLE\s+(' + _NUM + r')')


def _ase_bitmap_to_matname(path):
    """
    Derive a Blender material name from an ASE *BITMAP path — e.g.
    "textures\\base_wall\\wall01.tga" becomes "textures/base_wall/wall01.tga".
    Backslashes are normalized; the file extension is kept as-is (Blender
    material names should match the source texture filename verbatim).
    """
    return path.replace('\\', '/').strip().strip('"')


def _ase_material_identity(entry):
    """Identity for one parsed *MATERIAL / *SUBMATERIAL entry.

    The engine reads *MAP_DIFFUSE -> *BITMAP and ignores *MATERIAL_NAME
    entirely (Model_ase.cpp, ASE_KeyMAP_DIFFUSE), so that is the order tried
    here -- see material_identity_from_ase."""
    if not entry:
        return dict(raw='', decl='', origin='NONE', engine=False,
                    display='unknown')
    return material_identity_from_ase(entry.get('bitmap') or '',
                                      entry.get('name') or '')


def _ase_material_name(entry):
    """The Blender material name for one *MATERIAL entry: the canonical decl
    name for an idTech4-authored file, the raw *MATERIAL_NAME otherwise."""
    return _ase_material_identity(entry)['display']


def parse_ase_file(filepath):
    """
    Minimal line-based ASE parser. Returns (objects, materials) where:
      objects   = [ {name, verts:[Vector,...], faces:[(a,b,c),...],
                     face_mtl:[int,...], tverts:[(u,v),...],
                     tfaces:[(a,b,c),...], cverts:[(r,g,b),...],
                     cfaces:[(a,b,c),...],
                     vertex_normals:{face_idx: {orig_vert_idx: (x,y,z)}},
                     mat_ref:int}, ... ]
      materials = { top_index: {'name': str|None, 'bitmap': str|None,
                                 'uOffset':.., 'vOffset':.., 'uTiling':..,
                                 'vTiling':.., 'angle':..,
                                 'submats': {sub_index: {'name':.., 'bitmap':..}}} }

    Each material/submaterial entry captures both its literal
    *MATERIAL_NAME and the *BITMAP path from its *MAP_DIFFUSE block (idTech4
    tooling puts the real shader/material path in the diffuse bitmap, not
    the Max material name — see _ase_material_name). uOffset/vOffset/
    uTiling/vTiling/angle come from that same *MAP_DIFFUSE block's
    *UVW_U_OFFSET / *UVW_V_OFFSET / *UVW_U_TILING / *UVW_V_TILING /
    *UVW_ANGLE keys — verified against Model_ase.cpp (ASE_KeyMAP_DIFFUSE)
    and applied in build_ase_meshes exactly as ConvertASEToModelSurfaces
    does. Only ever captured on the TOP-LEVEL *MATERIAL (never a
    *SUBMATERIAL) — the real engine looks these up once per GEOMOBJECT,
    via the object's own *MATERIAL_REF, regardless of which submaterial
    ID any individual face uses.

    faces/tfaces/cfaces corner order: 3ds Max's *MESH_FACE (and
    *MESH_TFACE/*MESH_CFACE) list corners A, B, C in file order, and that
    order is kept AS-IS here (no B/C swap). 1.1.0 swapped each face's B
    and C on the theory that the engine's own internal correction
    (renderer/Model_ase.cpp, ASE_KeyMESH_FACE_LIST/ASE_KeyCFACE_LIST:
    "we are flipping the order here to change the front/back facing from
    3DS to our standard") needed mirroring for Blender too — same mistake
    as the *MESH_TVERT V-flip above, and same fix: confirmed against real
    game assets (models/mapobjects/filler/phone.ase and binder1-4.ase)
    that raw file order already yields correctly outward-facing normals
    in Blender, while the swap flips ~80-100% of faces inward. No axis
    conversion is needed for positions either (3ds Max and idTech4 are
    both Z-up, right-handed).

    *MESH_TVERT's raw V is used AS-IS here, same as .lwo's — see
    parse_lwo_file's docstring for the general reasoning. 1.1.0 briefly
    flipped it (`1.0 - v`) on the theory that the engine's own internal
    `1.0f - vm->val[k][1]`-style correction (ASE_KeyMESH_TVERTLIST,
    comment: "our OpenGL second texture axis is inverted from MAX's
    sense") meant Blender needed the same flip. In practice that made
    every imported .ase material appear upside-down — confirmed against
    real game assets — so the flip was reverted: raw ASE V already
    matches what Blender wants, the same as raw LWO V.
    """
    objects   = []
    materials = {}
    stack     = []           # generic block-context stack, matched on '}'
    cur_obj   = None
    cur_mat_idx    = None
    cur_submat_idx = None
    cur_normal_face = None   # tracks *MESH_FACENORMAL's face index for the *MESH_VERTEXNORMAL lines that follow it

    with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue

            if line == '}':
                if stack:
                    stack.pop()
                continue

            top = stack[-1] if stack else None

            if _ASE_RE_GEOM.match(line):
                cur_obj = {
                    'name': None, 'verts': [], 'faces': [], 'face_mtl': [],
                    'face_smoothing': [], 'face_edge_vis': [], 'node_tm': None,
                    'tverts': [], 'tfaces': [], 'cverts': [], 'cfaces': [],
                    'vertex_normals': {}, 'mat_ref': 0,
                }
                objects.append(cur_obj)
                stack.append('GEOMOBJECT')
                cur_normal_face = None
                continue

            if _ASE_RE_MATLIST.match(line):
                stack.append('MATERIAL_LIST')
                continue

            m = _ASE_RE_MATERIAL.match(line)
            if m and top == 'MATERIAL_LIST':
                # Position in the list, NOT the index the file writes after
                # *MATERIAL. ASE_KeyMATERIAL_LIST (renderer/Model_ase.cpp) never
                # reads that token -- it just Append()s -- so *MATERIAL_REF is an
                # index into encounter order. Real files disagree with their own
                # numbering: prey's scan_machine_lo.ASE declares one material as
                # "*MATERIAL 1" and its object then asks for *MATERIAL_REF 0.
                # Keying on the written number left ref 0 pointing at nothing,
                # and the material fell back to the 3ds Max name.
                cur_mat_idx = len(materials)
                materials[cur_mat_idx] = {
                    'name': None, 'bitmap': None, 'submats': {},
                    'uOffset': 0.0, 'vOffset': 0.0, 'uTiling': 1.0, 'vTiling': 1.0, 'angle': 0.0,
                }
                stack.append('MATERIAL')
                continue

            m = _ASE_RE_SUBMAT.match(line)
            if m and top == 'MATERIAL':
                # Encounter order again, for the same reason -- and here it is
                # also what *MESH_MTLID indexes into.
                cur_submat_idx = len(materials[cur_mat_idx]['submats'])
                materials[cur_mat_idx]['submats'][cur_submat_idx] = {'name': None, 'bitmap': None}
                stack.append('SUBMATERIAL')
                continue

            m = _ASE_RE_MATNAME.match(line)
            if m:
                if top == 'SUBMATERIAL' and cur_mat_idx is not None:
                    materials[cur_mat_idx]['submats'][cur_submat_idx]['name'] = m.group(1)
                elif top == 'MATERIAL' and cur_mat_idx is not None:
                    materials[cur_mat_idx]['name'] = m.group(1)
                continue

            m = _ASE_RE_BITMAP.match(line)
            if m and top == 'MAP_DIFFUSE' and len(stack) >= 2:
                owner = stack[-2]
                if owner == 'SUBMATERIAL' and cur_mat_idx is not None and cur_submat_idx in materials[cur_mat_idx]['submats']:
                    entry = materials[cur_mat_idx]['submats'][cur_submat_idx]
                    if not entry['bitmap']:
                        entry['bitmap'] = m.group(1)
                elif owner == 'MATERIAL' and cur_mat_idx is not None:
                    entry = materials[cur_mat_idx]
                    if not entry['bitmap']:
                        entry['bitmap'] = m.group(1)
                continue

            # UVW transform keys — only meaningful on the TOP-LEVEL
            # *MATERIAL's *MAP_DIFFUSE (see docstring above); a
            # *SUBMATERIAL's own copy is never consulted by the real
            # engine, so it's deliberately not captured here.
            if top == 'MAP_DIFFUSE' and len(stack) >= 2 and stack[-2] == 'MATERIAL' and cur_mat_idx is not None:
                m = _ASE_RE_UOFFSET.match(line)
                if m:
                    materials[cur_mat_idx]['uOffset'] = _f(m.group(1))
                    continue
                m = _ASE_RE_VOFFSET.match(line)
                if m:
                    materials[cur_mat_idx]['vOffset'] = _f(m.group(1))
                    continue
                m = _ASE_RE_UTILING.match(line)
                if m:
                    materials[cur_mat_idx]['uTiling'] = _f(m.group(1))
                    continue
                m = _ASE_RE_VTILING.match(line)
                if m:
                    materials[cur_mat_idx]['vTiling'] = _f(m.group(1))
                    continue
                m = _ASE_RE_UANGLE.match(line)
                if m:
                    materials[cur_mat_idx]['angle'] = _f(m.group(1))
                    continue

            if top == 'GEOMOBJECT' and cur_obj is not None:
                m = _ASE_RE_NODENAME.match(line)
                if m:
                    cur_obj['name'] = m.group(1)
                    continue
                m = _ASE_RE_MATREF.match(line)
                if m:
                    cur_obj['mat_ref'] = int(m.group(1))
                    continue

            if line.endswith('{'):
                stack.append(line.split()[0].lstrip('*'))
                continue

            if cur_obj is not None:
                m = _ASE_RE_VERT.match(line)
                if m:
                    idx = int(m.group(1))
                    v = Vector((_f(m.group(2)), _f(m.group(3)), _f(m.group(4))))
                    verts = cur_obj['verts']
                    while len(verts) <= idx:
                        verts.append(Vector((0.0, 0.0, 0.0)))
                    verts[idx] = v
                    continue

                m = _ASE_RE_FACE.match(line)
                if m:
                    a, b, c = int(m.group(2)), int(m.group(3)), int(m.group(4))
                    mm = _ASE_RE_MTLID.search(line)
                    mtl = int(mm.group(1)) if mm else 0
                    # Kept in raw file order (A, B, C) — see docstring above.
                    cur_obj['faces'].append((a, b, c))
                    cur_obj['face_mtl'].append(mtl)
                    # 3ds Max smoothing groups: a 32-bit mask, sometimes
                    # written as a comma-separated list of group numbers, and
                    # sometimes absent entirely (a fully faceted face). The
                    # engine never reads this -- it shades from *MESH_NORMALS.
                    ms = _ASE_RE_SMOOTHING.search(line)
                    mask = 0
                    if ms:
                        for tok in ms.group(1).replace(',', ' ').split():
                            try:
                                g = int(tok)
                            except ValueError:
                                continue
                            if 1 <= g <= 32:
                                mask |= 1 << (g - 1)
                    cur_obj['face_smoothing'].append(mask)
                    ev = _ASE_RE_EDGEVIS.search(line)
                    cur_obj['face_edge_vis'].append(
                        ((int(ev.group(1)) and 1) | (int(ev.group(2)) and 2)
                         | (int(ev.group(3)) and 4)) if ev else 7)
                    continue

                m = _ASE_RE_TMROW.match(line)
                if m:
                    row = int(m.group(1))
                    vals = (_f(m.group(2)), _f(m.group(3)), _f(m.group(4)))
                    tm = cur_obj.get('node_tm')
                    if tm is None:
                        tm = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                              (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)]
                    tm[row] = vals
                    cur_obj['node_tm'] = tm
                    continue

                m = _ASE_RE_TVERT.match(line)
                if m:
                    idx = int(m.group(1))
                    # V used raw (not flipped) — see docstring above.
                    uv = (_f(m.group(2)), _f(m.group(3)))
                    tverts = cur_obj['tverts']
                    while len(tverts) <= idx:
                        tverts.append((0.0, 0.0))
                    tverts[idx] = uv
                    continue

                m = _ASE_RE_TFACE.match(line)
                if m:
                    # Kept in raw file order too, same as *MESH_FACE, so
                    # corner k still lines up between faces[k] and tfaces[k].
                    cur_obj['tfaces'].append((int(m.group(2)), int(m.group(3)), int(m.group(4))))
                    continue

                m = _ASE_RE_CVERT.match(line)
                if m:
                    idx = int(m.group(1))
                    col = (_f(m.group(2)), _f(m.group(3)), _f(m.group(4)))
                    cverts = cur_obj['cverts']
                    while len(cverts) <= idx:
                        cverts.append((1.0, 1.0, 1.0))
                    cverts[idx] = col
                    continue

                m = _ASE_RE_CFACE.match(line)
                if m:
                    # Kept in raw file order too, same as *MESH_FACE.
                    cur_obj['cfaces'].append((int(m.group(2)), int(m.group(3)), int(m.group(4))))
                    continue

                m = _ASE_RE_FACENRM.match(line)
                if m:
                    cur_normal_face = int(m.group(1))
                    cur_obj['vertex_normals'].setdefault(cur_normal_face, {})
                    continue

                m = _ASE_RE_VERTNRM.match(line)
                if m and cur_normal_face is not None:
                    vidx = int(m.group(1))
                    nrm = (_f(m.group(2)), _f(m.group(3)), _f(m.group(4)))
                    cur_obj['vertex_normals'][cur_normal_face][vidx] = nrm
                    continue

    return objects, materials


def _ase_node_tm_normal(tm):
    """The 3x3 the engine multiplies each stored normal by.

    Model_ase.cpp does, per component:
        faceNormal[0] = n[0]*transform[0][0] + n[1]*transform[1][0] + n[2]*transform[2][0]
    i.e. the transpose of the upper 3x3 applied on the left -- and it does
    this to normals ONLY. Nothing in the engine transforms *MESH_VERTEX,
    because .ase vertex coordinates are already baked into world space.
    """
    if not tm:
        return None
    m = Matrix(((tm[0][0], tm[1][0], tm[2][0]),
                (tm[0][1], tm[1][1], tm[2][1]),
                (tm[0][2], tm[1][2], tm[2][2])))
    return None if m == Matrix.Identity(3) else m


def ase_smoothing_mask(value):
    """The unsigned 32-bit Max smoothing mask behind a stored signed int.

    Bit n set means the face is in smoothing group n+1. Stored two's
    complement because a Blender INT attribute is signed and group 32 sets
    bit 31.
    """
    return int(value) & 0xFFFFFFFF


def _ase_signed32(value):
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value > 0x7FFFFFFF else value


def _ase_synth_normals(mesh, obj_data, mode, rb_mats, un_mats, mat_of_poly):
    """Normals for an .ase that carries none, computed rather than read.

    ENGINE reproduces R_DeriveTangents -- unweighted sum of unit face normals,
    welded on exact position, per material -- because that is what the engine
    does when normalsParsed is false, and it never consults *MESH_SMOOTHING.

    FILE honours those smoothing groups instead: a corner sums only the faces
    that share a group bit with its own, which is what 3ds Max means by them
    and what a non-idTech4 model wants. 21 of the 169 shipped .ase assets have
    groups and no normals, so this is not a hypothetical branch.

    Returns per-loop normals, or None when nothing could be computed.
    """
    if not mesh.polygons:
        return None
    pos = [v.co.copy() for v in mesh.vertices]
    tris, loop_of = [], []
    for p in mesh.polygons:
        if p.loop_total != 3:
            return None
        tris.append(tuple(p.vertices))
        loop_of.append(tuple(range(p.loop_start, p.loop_start + 3)))

    out = [None] * len(mesh.loops)
    if mode == SHADING_ENGINE:
        groups = {}
        for t, tri in enumerate(tris):
            groups.setdefault(mat_of_poly(t), []).append(t)
        for _decl, idxs in groups.items():
            sub = [tris[t] for t in idxs]
            fn = (engine_dominant_normals(pos, sub)
                  if _decl in un_mats else engine_regenerated_normals(pos, sub))
            for k, t in enumerate(idxs):
                for j in range(3):
                    # NEGATED: engine_regenerated_normals/engine_dominant_
                    # normals take corners in ENGINE winding and use the
                    # engine's own (c-a)x(b-a), which is the opposite of
                    # Blender's (b-a)x(c-a). The .lwo path feeds them raw
                    # FILE order and reverses the winding when it builds the
                    # loops, so the two cancel there. .ase does neither:
                    # parse_ase_file deliberately keeps 3ds Max's raw A,B,C
                    # (the engine's own ASE_KeyMESH_FACE_LIST B/C flip is
                    # what makes id's convention agree with Blender's here --
                    # see its docstring), so these corners arrive already in
                    # BLENDER winding and only one of the two flips gets
                    # applied. Negating supplies the other, and is exactly
                    # equivalent to passing (i0, i2, i1): reversing a
                    # triangle negates its unit normal, which negates the
                    # accumulated sum and every zero-length fallback, and
                    # leaves the |raw| area that picks the dominant triangle
                    # unchanged. Without it every synthesised normal points
                    # into the surface -- all 22 shipped .ase assets with no
                    # *MESH_NORMALS imported inside-out under Engine Shading,
                    # which reads as hard faceting once a normal map is on.
                    out[loop_of[t][j]] = -fn[k * 3 + j]
        return out

    # FILE: average within shared *MESH_SMOOTHING bits
    masks = obj_data.get('face_smoothing') or []
    if len(masks) != len(tris):
        masks = [0] * len(tris)
    face_n = []
    for (i0, i1, i2) in tris:
        a, b, c = pos[i0], pos[i1], pos[i2]
        n = (b - a).cross(c - a)
        face_n.append(n.normalized() if n.length_squared > 0.0 else n)
    at_point = {}
    for t, tri in enumerate(tris):
        for idx in tri:
            at_point.setdefault(_position_key(pos[idx]), []).append(t)
    for t, tri in enumerate(tris):
        mine = masks[t]
        for j, idx in enumerate(tri):
            acc = face_n[t].copy()
            if mine:
                for h in at_point[_position_key(pos[idx])]:
                    if h != t and (masks[h] & mine):
                        acc = acc + face_n[h]
            out[loop_of[t][j]] = (acc.normalized() if acc.length_squared > 0.0
                                  else face_n[t])
    return out


def _ase_apply_face_attrs(mesh, obj_data):
    """*MESH_SMOOTHING and the AB:/BC:/CA: edge-visibility flags as editable
    INT face attributes. Written only when the file actually used them."""
    for attr_name, key, default in ((ASE_SMOOTH_ATTR, 'face_smoothing', 0),
                                    (ASE_EDGE_VIS_ATTR, 'face_edge_vis', 7)):
        vals = obj_data.get(key) or []
        if not vals or len(vals) != len(mesh.polygons):
            continue
        if all(v == default for v in vals):
            continue
        attr = mesh.attributes.new(attr_name, 'INT', 'FACE')
        if attr_name == ASE_SMOOTH_ATTR:
            vals = [_ase_signed32(v) for v in vals]
        attr.data.foreach_set('value', list(vals))


def build_ase_meshes(filepath, scale, name_hint=None,
                      shading=SHADING_ENGINE, shading_overrides=None):
    """Build Blender meshes from an .ase file. Local space (identity
    rotation/translation), only *scale* applied. Returns
    [(name, mesh, [material_name,...]), ...]. name is the GEOMOBJECT's
    real *NODE_NAME when the file set one; otherwise it falls back to
    *name_hint* if given (the model's full reference path as written in
    the .map "model" key — see the .map importer's use of this function,
    or build_lwo_meshes), or the model's own bare filename when no hint
    is given, suffixed with an index when the file has more than one
    unnamed object — rather than a generic placeholder that would be
    identical, and therefore meaningless, across every other unnamed
    .ase in the same scene.

    Vertex color (*MESH_CVERTLIST/*MESH_CFACELIST) is written to a "Col"
    face-corner color attribute whenever present, falling back to opaque
    white per corner otherwise (matching ConvertASEToModelSurfaces' own
    `identityColor` default for a mesh with `colorsParsed == false`).

    *shading*: one of SHADING_ITEMS, defaulting to SHADING_ENGINE.
    Under ENGINE or FILE, and when the object's *MESH_NORMALS block was
    present, its per-corner normals are applied as Blender custom split
    normals — exactly what the real engine does for any material WITHOUT
    a renderBump command (a material WITH one has its file normals
    discarded in favor of freshly regenerated ones; that distinction
    reaches here through *shading_overrides*). A face missing a specific
    corner's normal (partial *MESH_NORMALS* coverage — not expected from
    a real 3ds Max export, but handled defensively) falls back to that
    triangle's own flat normal rather than leaving it unset. An object
    with no *MESH_NORMALS at all skips custom normals entirely, as do
    SMOOTH and FLAT.
    """
    mode = resolve_shading_mode(shading)
    use_file_normals = mode in (SHADING_ENGINE, SHADING_FILE)
    rb_mats, un_mats = shading_overrides or (set(), set())
    objects, materials = parse_ase_file(filepath)
    _ase_identity_by_name = {}
    results = []
    fallback_base = (name_hint or '').strip().strip('"') or os.path.splitext(os.path.basename(filepath))[0]

    for obj_idx, obj_data in enumerate(objects):
        verts = obj_data['verts']
        faces = obj_data['faces']
        if not verts or not faces:
            continue

        tverts  = obj_data['tverts']
        tfaces  = obj_data['tfaces']
        has_uv  = bool(tverts) and len(tfaces) == len(faces)

        cverts = obj_data['cverts']
        cfaces = obj_data['cfaces']
        has_color = bool(cverts) and len(cfaces) == len(faces)

        vertex_normals = obj_data['vertex_normals']
        # An .ase with no *MESH_NORMALS is where ENGINE and FILE part company:
        # the engine regenerates from geometry and never looks at
        # *MESH_SMOOTHING, while the file's intent IS those groups.
        has_normals = use_file_normals and bool(vertex_normals)
        synth_normals = use_file_normals and not vertex_normals

        matinfo  = materials.get(obj_data['mat_ref'])
        submats  = matinfo['submats'] if matinfo else {}
        base_mat = _ase_material_name(matinfo)
        _ase_identity_by_name[base_mat] = _ase_material_identity(matinfo)

        # UV tiling/offset/rotation — verified against
        # ConvertASEToModelSurfaces: uOffset is NEGATED, vOffset is not;
        # angle is already in radians (idMath::Sin/Cos take it directly,
        # no conversion). Identity (0 offset, 1 tiling, 0 angle) whenever
        # there's no material info, matching the engine's own
        # `ase->materials.Num() == 0` fallback branch.
        uOffset = -matinfo['uOffset'] if matinfo else 0.0
        vOffset = matinfo['vOffset'] if matinfo else 0.0
        uTiling = matinfo['uTiling'] if matinfo else 1.0
        vTiling = matinfo['vTiling'] if matinfo else 1.0
        angle   = matinfo['angle'] if matinfo else 0.0
        sin_a = math.sin(angle)
        cos_a = math.cos(angle)

        def mat_name_for(mtlid, submats=submats, base_mat=base_mat):
            if submats:
                if mtlid in submats:
                    entry = submats[mtlid]
                else:
                    entry = next(iter(submats.values()))
                nm = _ase_material_name(entry)
                _ase_identity_by_name[nm] = _ase_material_identity(entry)
                return nm
            return base_mat

        bm = bmesh.new()
        uv_lay  = bm.loops.layers.uv.new("UVMap")
        col_lay = bm.loops.layers.float_color.new("Col")
        nrm_lay = bm.loops.layers.float_vector.new("temp_normal") if has_normals else None
        node_tm_n = _ase_node_tm_normal(obj_data.get('node_tm'))
        bverts = [bm.verts.new(Vector((v.x * scale, v.y * scale, v.z * scale))) for v in verts]
        bm.verts.ensure_lookup_table()

        mat_idx_map = {}
        any_normals_written = False
        for i, (a, b, c) in enumerate(faces):
            if a >= len(bverts) or b >= len(bverts) or c >= len(bverts):
                continue
            try:
                f = bm.faces.new((bverts[a], bverts[b], bverts[c]))
            except ValueError:
                continue

            mtlid = obj_data['face_mtl'][i] if i < len(obj_data['face_mtl']) else 0
            mn = mat_name_for(mtlid)
            if mn not in mat_idx_map:
                mat_idx_map[mn] = len(mat_idx_map)
            f.material_index = mat_idx_map[mn]

            if has_uv:
                ta, tb, tc = tfaces[i]
                uvs = []
                for ti in (ta, tb, tc):
                    if ti < len(tverts):
                        u, v = tverts[ti]
                    else:
                        u, v = 0.0, 0.0
                    u = u * uTiling + uOffset
                    v = v * vTiling + vOffset
                    uvs.append((u * cos_a + v * sin_a, u * -sin_a + v * cos_a))
                for loop, uv in zip(f.loops, uvs):
                    loop[uv_lay].uv = uv

            if has_color:
                ca, cb, cc = cfaces[i]
                cols = []
                for ci in (ca, cb, cc):
                    if ci < len(cverts):
                        r, g, bl = cverts[ci]
                    else:
                        r, g, bl = 1.0, 1.0, 1.0
                    cols.append((r, g, bl, 1.0))
            else:
                cols = [(1.0, 1.0, 1.0, 1.0)] * 3
            for loop, col in zip(f.loops, cols):
                loop[col_lay] = col

            if nrm_lay is not None:
                face_nrm = vertex_normals.get(i, {})
                fallback_n = None
                norms = []
                for vidx, bv in ((a, bverts[a]), (b, bverts[b]), (c, bverts[c])):
                    n = face_nrm.get(vidx)
                    if n is None:
                        if fallback_n is None:
                            fallback_n = _face_flat_normal(bverts[a].co, bverts[b].co, bverts[c].co)
                        n = fallback_n
                    norms.append(n)
                if node_tm_n is not None:
                    # The engine multiplies every stored normal by *NODE_TM
                    # and leaves positions alone -- see _ase_node_tm_normal.
                    norms = [(node_tm_n @ Vector(n)).normalized() for n in norms]
                for loop, n in zip(f.loops, norms):
                    loop[nrm_lay] = n
                any_normals_written = True

        if not bm.faces:
            bm.free()
            continue

        bm.normal_update()
        raw_name = (obj_data['name'] or '').strip()
        if raw_name:
            name = raw_name
        elif len(objects) > 1:
            name = f"{fallback_base}_{obj_idx}"
        else:
            name = fallback_base
        mesh = bpy.data.meshes.new(name)
        bm.to_mesh(mesh)
        bm.free()
        mesh.validate(clean_customdata=False)
        mesh.update()
        mark_primary_color_attribute(mesh)

        if synth_normals or any_normals_written:
            mark_smooth_for_custom_normals(mesh)

        if synth_normals:
            def _mat_of_poly(t, _mesh=mesh, _names=None):
                mats = _mesh.materials
                if not mats:
                    return ''
                mi = _mesh.polygons[t].material_index
                m = mats[mi] if mi < len(mats) else None
                return engine_canonical_decl(
                    material_decl_for_export(m, m.name if m else ''))
            synth = _ase_synth_normals(mesh, obj_data, mode, rb_mats, un_mats,
                                       _mat_of_poly)
            if synth and all(v is not None for v in synth):
                mesh.normals_split_custom_set([tuple(v) for v in synth])
                mesh.update()

        if any_normals_written:
            nrm_attr = mesh.attributes.get("temp_normal")
            if nrm_attr is not None:
                loop_normals = [tuple(nrm_attr.data[i].vector) for i in range(len(mesh.loops))]
                mesh.normals_split_custom_set(loop_normals)
                # Re-fetch by name rather than reusing nrm_attr: on Blender
                # 4.5, normals_split_custom_set() invalidates the handle
                # (its .name reads back empty, and removing an attribute by
                # an empty name raises "The attribute name must not be
                # empty"). 5.0+ happens not to invalidate it, which is why
                # this only broke on 4.5.
                nrm_attr = mesh.attributes.get("temp_normal")
                if nrm_attr is not None:
                    mesh.attributes.remove(nrm_attr)
                mesh.update()

        _ase_apply_face_attrs(mesh, obj_data)
        if obj_data.get('node_tm'):
            # Kept verbatim so an export can write the same matrix back. Rows
            # 0..2 are the basis, row 3 the position.
            mesh[ASE_NODE_TM_PROP] = [c for row in obj_data['node_tm'] for c in row]
        mat_names = list(mat_idx_map.keys())
        stash_material_identities(mesh, [_ase_identity_by_name.get(n) for n in mat_names])
        results.append((name, mesh, mat_names))

    return results


# ── LWO (Lightwave Object, LWO2) ────────────────────────────────────

def _lwo_read_vx(data, offset):
    """Decode one LWO2 variable-length index (VX) starting at *offset*.
    Returns (index, new_offset)."""
    val = struct.unpack_from('>H', data, offset)[0]
    if val < 0xFF00:
        return val, offset + 2
    val2 = struct.unpack_from('>H', data, offset + 2)[0]
    return ((val & 0x00FF) << 16) | val2, offset + 4


def _lwo_read_cstr(data, offset):
    """Null-terminated string; each string (incl. terminator) is padded
    to an even byte length in LWO2 chunks. Returns (string, new_offset)."""
    z = data.find(b'\x00', offset)
    if z == -1:
        return '', len(data)
    s = data[offset:z].decode('utf-8', 'replace')
    length = z - offset + 1
    if length % 2 == 1:
        length += 1
    return s, offset + length


def parse_lwo_file(filepath):
    """
    Minimal LWO2 IFF-chunk parser. Returns (layers, tags) where:
      tags   = [surface_name, ...]                    (index -> name)
      layers = [ {name, pivot:Vector, points:[Vector,...],
                  polys:[[idx,...],...], poly_types:[b'FACE'|..., ...],
                  ptag_surf:{poly_idx: tag_idx},
                  uv_chunks:[{'perpoly':bool, 'data':{...}}, ...] }, ... ]

    Point coordinates are converted from Lightwave's axis convention to
    idTech4's. LightWave is Y-up, LEFT-handed (X-right, Y-up, Z-away-from-
    -viewer); Blender/idTech4 are Z-up, RIGHT-handed. Simply relabeling
    axes (e.g. idtech.y = -lwo.z) keeps determinant +1, which is a proper
    rotation — but converting a left-handed source into a right-handed
    target needs a parity flip (determinant -1), or the model comes out
    mirrored. So Y and Z are swapped WITHOUT an extra sign flip:
        idtech.x =  lwo.x
        idtech.y =  lwo.z
        idtech.z =  lwo.y
    (determinant -1, correct handedness).

    Polygon winding DOES need to be reversed, despite that determinant
    flip — verified directly against the real Doom 3 GPL source
    (renderer/Model_lwo.cpp, lwGetPolyNormals): the engine computes each
    polygon's own normal as `cross(v[1]-v[0], v[last]-v[0])`, evaluated
    on the RAW (unswapped) LWO point positions, i.e. it is NOT the
    "cross product of the closing edge" a literal reading of the LWO SDK
    docs suggests — it's anchored at v[0] both times. Working through
    that exact formula on a concrete polygon and comparing against what
    Blender computes from the same index order on the axis-swapped
    points (confirmed with a synthetic .lwo run through this file's own
    parser) shows the two do NOT cancel out: building the Blender face
    with the index order kept as-is yields a normal exactly opposite the
    real engine's. build_lwo_meshes reverses each polygon's vertex order
    before creating the BMesh face to correct for this.

    LAYR pivot: a layer's PNTS data is stored RELATIVE to that layer's own
    pivot point (also carried in the LAYR chunk, right before the name) —
    NOT already in final/absolute position. A multi-layer .lwo commonly
    has each layer's pivot moved in Layout to place that piece within the
    overall model, so a layer with a non-zero pivot that's read without
    adding it back renders at (near) its own local origin instead of its
    real position — i.e. the whole layer/mesh ends up in the wrong place.
    The pivot is parsed here and returned per-layer; build_lwo_meshes adds
    it back onto every point before use.

    UV chunks: verified directly against the real idTech4 engine source
    (renderer/Model.cpp, idRenderModelStatic::ConvertLWOToModelSurfaces).
    A layer can carry more than one TXUV VMAP/VMAD chunk (e.g. one
    discontinuous UV channel PER MATERIAL is a real, common exporter
    pattern), and the real engine does NOT select one by name — a
    chunk's own name string, read but discarded here, is never consulted
    anywhere in the real conversion code. Instead EVERY TXUV chunk is
    appended, in exact file order, to uv_chunks — matching how the real
    parser's lwGetObject() links every VMAP/VMAD chunk onto one single
    linked list (layer->vmap) regardless of name, each chunk its own
    separate node even if it shares a name with an earlier one. Each
    entry's 'perpoly' flag distinguishes a continuous per-POINT chunk
    (VMAP, False) from a discontinuous per-POLYGON-CORNER chunk (VMAD,
    True) — build_lwo_meshes walks this list to resolve each corner's
    UV the same two-stage way the engine does.

    V is deliberately NOT flipped here. The real engine DOES flip it
    (`tvList[k+offset].y = 1.0f - vm->val[k][1];  // invert the t`), but
    that converts LightWave's raw value INTO idTech4's own internal
    top-down texture convention for ITS renderer — an engine-internal
    detail, not a statement that LWO's raw V needs correcting
    universally. LightWave (like most DCC/artist tools) uses the same
    bottom-left-origin UV convention Blender does, so the raw value
    already IS what Blender wants. The same holds for .ase's raw V —
    see parse_ase_file's docstring for why an earlier attempt to flip
    it there (mirroring the engine's own internal correction) was
    reverted after it produced upside-down materials on real assets.

    Vertex color (VMAP/VMAD type RGBA) and the per-surface SURF chunk
    (base/fallback color + SMAN smoothing angle) are captured the same
    way — see build_lwo_meshes for how they're resolved and applied.
    PTAG SMGP (smoothing group) is captured per-polygon alongside SURF.

    Returns (layers, tags, surfaces) — surfaces is {name: {'color':
    (r,g,b), 'smooth': radians}}, keyed by the same names as *tags*
    (SURF chunks are per-FILE, not per-layer).

    Also supports legacy pre-LWO2 "LWOB" (LightWave 5.x) files — detected
    from the FORM type at byte 8 — via _parse_lwob, matching the real
    engine's own lwGetObject5/lwGetPolygons5/lwGetSurface5 fallback path
    for objects that were never re-saved in LWO2 format.
    """
    with open(filepath, 'rb') as fh:
        data = fh.read()

    if len(data) < 12 or data[0:4] != b'FORM':
        raise ValueError("Not a valid IFF/LWO file")

    form_type = data[8:12]
    form_size = struct.unpack_from('>I', data, 4)[0]
    end = min(len(data), 8 + form_size)

    if form_type == b'LWOB':
        return _parse_lwob(data, end)
    if form_type != b'LWO2':
        raise ValueError(f"Unsupported .lwo FORM type {form_type!r} (expected LWO2 or LWOB)")

    off = 12  # skip FORM, size, and the 4-byte form type (LWO2)

    tags       = []
    layers     = []
    surfaces   = {}
    cur_layer  = None

    def new_layer(name):
        # Keep the RAW name exactly as parsed, including empty/blank —
        # do NOT substitute a generic "LayerN" placeholder here. Many real
        # single-layer .lwo files never set a layer name at all, and a
        # placeholder like "Layer0" would be IDENTICAL across every one of
        # them, causing meaningless Blender-side name collisions across
        # totally unrelated model files (auto-uniquified into things like
        # "Layer0.252"). build_lwo_meshes has the actual file path and can
        # fall back to something far more useful (the model's own
        # filename) when this comes back empty.
        return {
            'name': (name or '').strip(),
            'pivot': Vector((0.0, 0.0, 0.0)),
            'ptag_part': {}, 'point_maps': [],
            'points': [], 'points_raw': [], 'polys': [], 'poly_types': [],
            'ptag_surf': {}, 'ptag_smgp': {}, 'uv_chunks': [], 'color_chunks': [],
        }

    escrow = []          # verbatim bytes of every chunk we do not model
    while off + 8 <= end:
        cid = data[off:off + 4]; off += 4
        csize = struct.unpack_from('>I', data, off)[0]; off += 4
        cstart = off
        cend = min(off + csize, end)
        cdata = data[cstart:cend]
        off = cend
        if csize % 2 == 1 and off < end:
            off += 1  # chunk padding byte

        if cid == b'TAGS':
            p = 0
            while p < len(cdata):
                z = cdata.find(b'\x00', p)
                if z == -1:
                    break
                tags.append(cdata[p:z].decode('utf-8', 'replace'))
                length = z - p + 1
                if length % 2 == 1:
                    length += 1
                p += length
            continue

        if cid == b'LAYR':
            if cur_layer is not None:
                layers.append(cur_layer)
            name = ''
            if len(cdata) >= 16:
                name, _ = _lwo_read_cstr(cdata, 16)
            cur_layer = new_layer(name)
            if len(cdata) >= 16:
                # Pivot vector sits at bytes [4:16) (after U2 number + U2
                # flags, before the name) — same raw axis convention as
                # PNTS, so apply the same x/z/y swap used there.
                ppx, ppy, ppz = struct.unpack_from('>fff', cdata, 4)
                cur_layer['pivot'] = Vector((ppx, ppz, ppy))
            continue

        if cur_layer is None:
            cur_layer = new_layer(None)

        if cid == b'PNTS':
            pts = cur_layer['points']
            raw = cur_layer['points_raw']
            n = (len(cdata) // 12) * 12
            for p in range(0, n, 12):
                x, y, z = struct.unpack_from('>fff', cdata, p)
                pts.append(Vector((x, z, y)))
                raw.append((x, y, z))
            continue

        if cid == b'POLS':
            if len(cdata) < 4:
                continue
            ptype = cdata[0:4]
            p = 4
            while p + 2 <= len(cdata):
                numv = struct.unpack_from('>H', cdata, p)[0] & 0x03FF
                p += 2
                idxs = []
                ok = True
                for _ in range(numv):
                    if p + 2 > len(cdata):
                        ok = False
                        break
                    idx, p = _lwo_read_vx(cdata, p)
                    idxs.append(idx)
                if not ok:
                    break
                cur_layer['polys'].append(idxs)
                cur_layer['poly_types'].append(ptype)
            continue

        if cid == b'PTAG':
            if len(cdata) < 4:
                continue
            subtype = cdata[0:4]
            p = 4
            while p + 2 <= len(cdata):
                poly_idx, p = _lwo_read_vx(cdata, p)
                if p + 2 > len(cdata):
                    break
                tag_idx = struct.unpack_from('>H', cdata, p)[0]; p += 2
                if subtype == b'SURF':
                    cur_layer['ptag_surf'][poly_idx] = tag_idx
                elif subtype == b'SMGP':
                    cur_layer['ptag_smgp'][poly_idx] = tag_idx
                elif subtype == b'PART':
                    cur_layer['ptag_part'][poly_idx] = tag_idx
            continue

        if cid == b'VMAP':
            if len(cdata) < 6:
                continue
            vtype = cdata[0:4]
            dim = struct.unpack_from('>H', cdata, 4)[0]
            vname, p = _lwo_read_cstr(cdata, 6)
            if vtype == b'TXUV' and dim >= 2:
                # Verified directly against the real idTech4 engine
                # source (renderer/Model.cpp,
                # idRenderModelStatic::ConvertLWOToModelSurfaces): it
                # does NOT select a UV channel by name (a chunk's own
                # name string, read above, is never consulted anywhere
                # for UV purposes — confirmed no reference to it exists
                # in the conversion code at all). Every TXUV-type VMAP
                # chunk is instead appended, in the exact order chunks
                # appear in the file, to ONE combined list — see
                # build_lwo_meshes for how that list is walked per
                # polygon corner. Each chunk is kept as its own separate
                # entry (not merged into any previous chunk sharing the
                # same name) since the real lwGetObject() parser does
                # the same: every VMAP/VMAD chunk becomes its own
                # separate linked-list node regardless of name reuse.
                #
                # V is NOT flipped here. The engine's own `1.0f -
                # vm->val[k][1]` (// invert the t) converts LightWave's
                # raw value INTO idTech4's own internal top-down texture
                # convention for its renderer — that's an engine-internal
                # detail, not a statement that LWO's raw V is "wrong" or
                # needs correcting universally. LightWave (like most
                # DCC/artist tools — confirmed by this importer's .ase
                # *MESH_TVERT handling, which has never flipped V and
                # works correctly) uses the SAME bottom-left-origin UV
                # convention Blender does. So the raw value already IS
                # what Blender wants; flipping it here would convert
                # bottom-up (correct for Blender) into top-down (correct
                # only for idTech4's own internal renderer), which is
                # backwards for an importer whose target is Blender, not
                # idTech4's renderer.
                chunk_uv = {}
                while p + 2 <= len(cdata):
                    vidx, p = _lwo_read_vx(cdata, p)
                    if p + 4 * dim > len(cdata):
                        break
                    vals = struct.unpack_from('>%df' % dim, cdata, p); p += 4 * dim
                    chunk_uv[vidx] = (vals[0], vals[1])
                cur_layer['uv_chunks'].append({'perpoly': False, 'data': chunk_uv,
                                               'name': vname})
            elif vtype == b'RGBA' and dim >= 4:
                # Vertex color — same two-list (VMAP=continuous/point,
                # VMAD=discontinuous/corner) + file-order-only, name-blind
                # resolution as TXUV above. Verified against Model.cpp:
                # only VMAP/VMAD type RGBA is ever consulted (no plain
                # 3-channel "RGB " type is checked anywhere in the real
                # conversion code).
                chunk_col = {}
                while p + 2 <= len(cdata):
                    vidx, p = _lwo_read_vx(cdata, p)
                    if p + 4 * dim > len(cdata):
                        break
                    vals = struct.unpack_from('>%df' % dim, cdata, p); p += 4 * dim
                    chunk_col[vidx] = (vals[0], vals[1], vals[2], vals[3])
                cur_layer['color_chunks'].append({'perpoly': False, 'data': chunk_col,
                                                  'name': vname})
            elif vtype in (b'WGHT', b'MORF', b'SPOT', b'PICK'):
                # Per-POINT maps the engine never reads but the file really
                # carries: weight maps, morph targets (MORF relative, SPOT
                # absolute) and selection sets (PICK, dimension 0).
                want = {b'WGHT': 1, b'MORF': 3, b'SPOT': 3, b'PICK': 0}[vtype]
                chunk_vals = {}
                while p + 2 <= len(cdata):
                    vidx, p = _lwo_read_vx(cdata, p)
                    if p + 4 * dim > len(cdata):
                        break
                    vals = struct.unpack_from('>%df' % dim, cdata, p) if dim else ()
                    p += 4 * dim
                    chunk_vals[vidx] = vals[:want] if want else (1.0,)
                cur_layer['point_maps'].append(
                    {'type': vtype, 'name': vname, 'dim': dim, 'data': chunk_vals})
            continue

        if cid == b'VMAD':
            if len(cdata) < 6:
                continue
            vtype = cdata[0:4]
            dim = struct.unpack_from('>H', cdata, 4)[0]
            vname, p = _lwo_read_cstr(cdata, 6)
            if vtype == b'TXUV' and dim >= 2:
                # Same reasoning as VMAP above — appended in file order,
                # name never consulted, V NOT flipped (see VMAP comment
                # for why).
                chunk_uv = {}
                while p + 2 <= len(cdata):
                    vidx, p = _lwo_read_vx(cdata, p)
                    if p + 2 > len(cdata):
                        break
                    pidx, p = _lwo_read_vx(cdata, p)
                    if p + 4 * dim > len(cdata):
                        break
                    vals = struct.unpack_from('>%df' % dim, cdata, p); p += 4 * dim
                    chunk_uv[(pidx, vidx)] = (vals[0], vals[1])
                cur_layer['uv_chunks'].append({'perpoly': True, 'data': chunk_uv,
                                               'name': vname})
            elif vtype == b'RGBA' and dim >= 4:
                chunk_col = {}
                while p + 2 <= len(cdata):
                    vidx, p = _lwo_read_vx(cdata, p)
                    if p + 2 > len(cdata):
                        break
                    pidx, p = _lwo_read_vx(cdata, p)
                    if p + 4 * dim > len(cdata):
                        break
                    vals = struct.unpack_from('>%df' % dim, cdata, p); p += 4 * dim
                    chunk_col[(pidx, vidx)] = (vals[0], vals[1], vals[2], vals[3])
                cur_layer['color_chunks'].append({'perpoly': True, 'data': chunk_col,
                                                  'name': vname})
            continue

        if cid == b'SURF':
            # Per-FILE (not per-layer) surface definitions, keyed by name
            # (matches TAGS/PTAG SURF names). Only the two fields the real
            # engine actually consults for geometry conversion are read —
            # base/fallback vertex color (COLR) and the max smoothing
            # angle (SMAN, radians) used by build_lwo_meshes' vertex-
            # normal averaging — verified against Model_lwo.cpp
            # lwGetSurface/lwGetVertNormals. Everything else in a SURF
            # chunk (reflection, texture layers, etc.) drives LightWave's
            # own preview shading only; idTech4 rendering uses the .mtr
            # material system instead, keyed by this same name.
            name, p = _lwo_read_cstr(cdata, 0)
            _srcname, p = _lwo_read_cstr(cdata, p)
            color = None
            smooth = 0.0
            surf_extra = []
            while p + 6 <= len(cdata):
                sub_id = cdata[p:p + 4]
                sub_sz = struct.unpack_from('>H', cdata, p + 4)[0]
                p += 6
                sub_data = cdata[p:p + sub_sz]
                if sub_id == b'COLR' and len(sub_data) >= 12:
                    color = struct.unpack_from('>fff', sub_data, 0)
                elif sub_id == b'SMAN' and len(sub_data) >= 4:
                    smooth = struct.unpack_from('>f', sub_data, 0)[0]
                else:
                    # BLOK texture blocks, DIFF/SPEC/REFL/TRAN/SIDE and the
                    # rest. None of it is modelled -- no shader graph is built
                    # from a .lwo -- but it is real data, so the raw bytes are
                    # kept to put straight back on export.
                    surf_extra.append(cdata[p - 6:p + sub_sz + (sub_sz & 1)])
                p += sub_sz + (sub_sz & 1)
            # 0.78431 matches lwDefaultSurface()'s non-zero default (LW's
            # own default surface grey) — the fallback whenever no COLR
            # subchunk is present.
            surfaces[name] = {'color': color or (0.78431, 0.78431, 0.78431),
                               'smooth': max(smooth, 0.0),
                               'extra': b''.join(surf_extra)}
            continue

        if cid != b'BBOX':
            # Unmodelled but real: ENVL, CLIP, DESC, TEXT, ICON, anything
            # unrecognised. BBOX is excluded because it is derived data the
            # writer recomputes from the points it actually has.
            escrow.append(cid + struct.pack('>I', csize) + cdata
                          + (b'\x00' if csize % 2 else b''))
        continue

    if cur_layer is not None:
        layers.append(cur_layer)

    for lay in layers:
        lay['escrow'] = escrow
    return layers, tags, surfaces


def _parse_lwob(data, end):
    """
    Parse a legacy pre-LWO2 "LWOB" (LightWave 5.x) object — same return
    shape as the LWO2 path above: (layers, tags, surfaces), with exactly
    one layer (LWOB has no LAYR chunk / layer concept at all — the real
    engine's lwGetObject5 allocates a single implicit layer at zero
    pivot). Verified directly against the real Doom 3 GPL source
    (renderer/Model_lwo.cpp: lwGetObject5, lwGetPolygons5, lwGetSurface5):

      - PNTS is byte-identical to the LWO2 case (lwGetPoints is shared
        between both formats) — same x/z/y axis swap applies.
      - POLS embeds each polygon's surface index directly (a trailing
        signed I2 per polygon, 1-based, negated+followed by a "detail
        polygon" marker I2 to skip when negative) — LWOB has no separate
        PTAG chunk, and vertex indices are plain U2 (no VX encoding).
      - SRFS is a plain null-terminated name list, byte-identical to
        LWO2's TAGS (reused here via the same parsing as that chunk).
      - SURF subchunks use 2-byte sizes in BOTH formats (LWO2 kept this
        from the old format for its own SURF chunk too — only the
        OUTER/top-level chunk framing differs, which is already 4-byte
        IFF standard in both). LWOB's smoothing angle can come from
        either an explicit SMAN float, or — if the surface only sets the
        old FLAG bit 4 ("Smoothing" checkbox, no custom angle in this
        older format) — a fixed 1.56207 rad (~89.5°) default, matching
        lwGetSurface5 exactly. LWOB has no PTAG SMGP either, so every
        polygon implicitly shares smoothing group 0.
    """
    off = 12  # skip FORM, size, and the 4-byte form type ('LWOB')
    tags     = []
    surfaces = {}
    layer = {
        'name': '', 'pivot': Vector((0.0, 0.0, 0.0)),
        'ptag_part': {}, 'point_maps': [],
        'points': [], 'points_raw': [], 'polys': [], 'poly_types': [],
        'ptag_surf': {}, 'ptag_smgp': {}, 'uv_chunks': [], 'color_chunks': [],
    }

    while off + 8 <= end:
        cid = data[off:off + 4]; off += 4
        csize = struct.unpack_from('>I', data, off)[0]; off += 4
        cstart = off
        cend = min(off + csize, end)
        cdata = data[cstart:cend]
        off = cend
        if csize % 2 == 1 and off < end:
            off += 1

        if cid == b'PNTS':
            pts = layer['points']
            raw = layer['points_raw']
            n = (len(cdata) // 12) * 12
            for p in range(0, n, 12):
                x, y, z = struct.unpack_from('>fff', cdata, p)
                pts.append(Vector((x, z, y)))
                raw.append((x, y, z))
            continue

        if cid == b'POLS':
            p = 0
            pi = 0
            while p + 2 <= len(cdata):
                numv = struct.unpack_from('>H', cdata, p)[0]; p += 2
                idxs = []
                ok = True
                for _ in range(numv):
                    if p + 2 > len(cdata):
                        ok = False
                        break
                    idxs.append(struct.unpack_from('>H', cdata, p)[0]); p += 2
                if not ok or p + 2 > len(cdata):
                    break
                surf_raw = struct.unpack_from('>h', cdata, p)[0]; p += 2
                if surf_raw < 0:
                    surf_raw = -surf_raw
                    p += 2  # skip the trailing detail-polygon count/marker
                layer['polys'].append(idxs)
                layer['poly_types'].append(b'FACE')
                if surf_raw >= 1:
                    layer['ptag_surf'][pi] = surf_raw - 1
                pi += 1
            continue

        if cid == b'SRFS':
            p = 0
            while p < len(cdata):
                z = cdata.find(b'\x00', p)
                if z == -1:
                    break
                tags.append(cdata[p:z].decode('utf-8', 'replace'))
                length = z - p + 1
                if length % 2 == 1:
                    length += 1
                p += length
            continue

        if cid == b'SURF':
            name, p = _lwo_read_cstr(cdata, 0)
            color = None
            smooth = 0.0
            while p + 6 <= len(cdata):
                sub_id = cdata[p:p + 4]
                sub_sz = struct.unpack_from('>H', cdata, p + 4)[0]
                p += 6
                sub_data = cdata[p:p + sub_sz]
                if sub_id == b'COLR' and len(sub_data) >= 3:
                    color = (sub_data[0] / 255.0, sub_data[1] / 255.0, sub_data[2] / 255.0)
                elif sub_id == b'FLAG' and len(sub_data) >= 2:
                    flags = struct.unpack_from('>H', sub_data, 0)[0]
                    if flags & 4:
                        smooth = 1.56207
                elif sub_id == b'SMAN' and len(sub_data) >= 4:
                    smooth = struct.unpack_from('>f', sub_data, 0)[0]
                p += sub_sz + (sub_sz & 1)
            surfaces[name] = {'color': color or (0.78431, 0.78431, 0.78431),
                               'smooth': max(smooth, 0.0)}
            continue

        # unknown/unneeded chunk (CRVS, PCHS, MBAL, ENVL, ...): skip
        continue

    return [layer], tags, surfaces


# The engine's own merge tolerances, from renderer/Model.cpp:
#   r_slopVertex   "0.01"   "merge xyz coordinates this far apart"
#   r_slopTexCoord "0.001"  "merge texture coordinates this far apart"
#   r_slopNormal   "0.02"   "merge normals that dot less than this"
# Only ever used by the "Build Engine Render Mesh" import mode — a plain
# import reproduces the FILE, which the engine never welds on load.
_R_SLOP_VERTEX = 0.01
_R_SLOP_TEXCOORD = 0.001
_R_SLOP_NORMAL = 0.02


def _lwo_weld_indices(values, epsilon, box_hash_size=32):
    """idVectorSubset<type,dimension>::FindVector (idlib/containers/VectorSet.h)
    run over a whole list: returns, for each entry, either its own index or
    the index of an EARLIER entry within *epsilon* on every axis.

    Faithful to the engine's version: the bounds are expanded by
    2 * boxHashSize * epsilon, the cell is found by offsetting half a box so
    a value near a boundary still lands next to its neighbours, all 2**dim
    corner cells are probed, the test is per-axis (Chebyshev, not Euclidean),
    and the first hit inside a cell wins with the most recently added
    candidate checked first (idHashIndex chains are LIFO).

    The engine flattens the cell coordinate into a masked integer key, so
    unrelated cells can share a hash bucket; this uses the cell tuple
    directly instead. That can only ever REMOVE candidates that would have
    failed the epsilon test anyway — two values within epsilon on every axis
    always share one of the probed cells — so the result is the same.
    """
    n = len(values)
    if n == 0:
        return []
    dim = len(values[0])
    mins, inv, half = [], [], []
    for k in range(dim):
        lo = min(v[k] for v in values) - 2 * box_hash_size * epsilon
        hi = max(v[k] for v in values) + 2 * box_hash_size * epsilon
        box = (hi - lo) / float(box_hash_size)
        if box <= 0.0:
            box = 1.0
        mins.append(lo)
        inv.append(1.0 / box)
        half.append(box * 0.5)

    cells = {}
    remap = [0] * n
    corners = 1 << dim
    for i, v in enumerate(values):
        partial = [int((v[k] - mins[k] - half[k]) * inv[k]) for k in range(dim)]
        found = -1
        for c in range(corners):
            key = tuple(partial[k] + ((c >> k) & 1) for k in range(dim))
            bucket = cells.get(key)
            if bucket:
                for j in reversed(bucket):
                    if all(abs(values[j][k] - v[k]) <= epsilon for k in range(dim)):
                        found = j
                        break
            if found >= 0:
                break
        if found >= 0:
            remap[i] = found
            continue
        key = tuple(int((v[k] - mins[k]) * inv[k]) for k in range(dim))
        cells.setdefault(key, []).append(i)
        remap[i] = i
    return remap


def _lwo_vecangle(a, b):
    """Angle in radians between two already-unit-length vectors — matches
    the real engine's vecangle(), used by lwGetVertNormals' smoothing-
    angle comparison (acos of the dot product, clamped for float noise)."""
    d = max(-1.0, min(1.0, a.dot(b)))
    return math.acos(d)


def _lwo_compute_corner_normals(layer, tags, surfaces):
    # NOTE: *tags* must be the RAW surface names straight out of the file,
    # because *surfaces* is keyed by those. Passing the canonical display
    # names silently loses every smoothing angle whose name canonicalises to
    # something different -- see the archived changelog.
    """
    Faithful port of Model_lwo.cpp's lwGetPolyNormals + lwGetVertNormals:
    for every polygon corner, start from that polygon's own flat normal;
    if its surface's SMAN smoothing angle is > 0, sum in the (already
    unit) normals of every OTHER polygon sharing that point AND
    smoothing group (PTAG SMGP; 0 for both when absent, so untagged
    files compare equal and smoothing is never group-restricted), whose
    angle to this polygon's normal is within that threshold, then
    re-normalize. A surface with no SMAN (smooth <= 0, the default for
    any surface that never sets one — LightWave's "Smoothing" checkbox
    off) stays fully faceted: every corner just gets the polygon's own
    flat normal, unmodified.

    Computed entirely in RAW (pre axis-swap) LWO space, exactly like the
    engine, then axis-swapped once at the end — cross products are
    pseudovectors, so computing on the already-swapped points would (as
    with polygon winding — see parse_lwo_file) give a mismatched result,
    not just a relabeled one.

    Only FACE/PTCH polygons participate (matching what build_lwo_meshes
    ever turns into renderable geometry) — a stray BONE/MBAL entry
    sharing a point with real geometry doesn't affect its shading.

    Returns {poly_index: [Vector, ...]} in idtech/Blender space, one
    normal per corner, in the SAME (file, unreversed) order as
    layer['polys'][poly_index] — build_lwo_meshes reverses it in lockstep
    with the vertex/UV reversal it already does for winding.
    """
    raw   = layer['points_raw']
    polys = layer['polys']
    poly_types = layer['poly_types']
    ptag_surf  = layer['ptag_surf']
    ptag_smgp  = layer['ptag_smgp']

    poly_normal    = {}
    poly_smooth    = {}
    poly_smgp      = {}
    point_to_polys = {}
    for pi, idxs in enumerate(polys):
        if poly_types[pi] not in LWO_POLY_TYPES or len(idxs) < 3:
            continue
        if any(i >= len(raw) for i in idxs):
            continue
        tag_idx = ptag_surf.get(pi)
        mn = tags[tag_idx] if (tag_idx is not None and tag_idx < len(tags)) else 'unknown'
        surf = surfaces.get(mn)
        poly_smooth[pi] = surf['smooth'] if surf else 0.0
        poly_smgp[pi]   = ptag_smgp.get(pi, 0)

        p0    = Vector(raw[idxs[0]])
        p1    = Vector(raw[idxs[1]])
        plast = Vector(raw[idxs[-1]])
        n = (p1 - p0).cross(plast - p0)
        if n.length_squared > 0.0:
            n.normalize()
        poly_normal[pi] = n

        for i in idxs:
            point_to_polys.setdefault(i, []).append(pi)

    result = {}
    for pi, idxs in enumerate(polys):
        if pi not in poly_normal:
            continue
        own_n  = poly_normal[pi]
        smooth = poly_smooth[pi]
        corners = []
        for point_idx in idxs:
            if smooth <= 0.0:
                acc = Vector(own_n)
            else:
                acc = Vector(own_n)
                for h in point_to_polys.get(point_idx, ()):
                    if h == pi or poly_smgp[h] != poly_smgp[pi]:
                        continue
                    if _lwo_vecangle(own_n, poly_normal[h]) > smooth:
                        continue
                    acc += poly_normal[h]
                if acc.length_squared > 0.0:
                    acc.normalize()
                else:
                    acc = Vector(own_n)
            corners.append(Vector((acc.x, acc.z, acc.y)))  # axis-swap once, at the end
        result[pi] = corners

    return result


def lwo_vmap_groups(chunks, default_name):
    """Group VMAP/VMAD chunks into one entry per LightWave map NAME.

    Returns [(name, [chunk, ...]), ...] in first-appearance order. Within a
    group the reader still does the engine's two passes -- every continuous
    VMAP first, then every discontinuous VMAD on top -- so a map split across
    a VMAP/VMAD pair resolves exactly as it always did; the only change is
    that a DIFFERENT name no longer overwrites this one.
    """
    groups, order = {}, []
    for ch in chunks:
        nm = (ch.get('name') or '').strip() or default_name
        if nm not in groups:
            groups[nm] = []
            order.append(nm)
        groups[nm].append(ch)
    return [(nm, groups[nm]) for nm in order]


MAX_BLENDER_UV_LAYERS = 8      # Blender's own MAX_MTFACE


def lwo_group_coverage(chunks, polys):
    """Which polygon corners a group of chunks actually covers.

    A discontinuous VMAD names the corner directly; a continuous VMAP names a
    point, so it covers every corner using that point.
    """
    cov = set()
    for ch in chunks:
        if ch['perpoly']:
            cov.update(ch['data'].keys())
        else:
            pts = ch['data']
            for pi, idxs in enumerate(polys):
                for v in idxs:
                    if v in pts:
                        cov.add((pi, v))
    return cov


def lwo_pack_vmap_groups(groups, polys, default_name, max_layers=MAX_BLENDER_UV_LAYERS):
    """Turn named chunk groups into the Blender layers they should become.

    Two groups whose coverage does not intersect are channels of one map --
    the one-TXUV-chunk-per-material pattern every Doom 3 .lwo uses -- and
    merge. Two that do intersect are genuinely different maps and stay apart.
    Returns [(layer_name, [chunk, ...]), ...] plus a list of warnings.
    """
    packed, notes = [], []
    for name, chunks in groups:
        cov = lwo_group_coverage(chunks, polys)
        for entry in packed:
            if not (entry['cov'] & cov):
                entry['chunks'].extend(chunks)
                entry['cov'] |= cov
                entry['merged'] += 1
                break
        else:
            packed.append({'name': name, 'chunks': list(chunks), 'cov': cov,
                           'merged': 0})
    if len(packed) > max_layers:
        notes.append("%d overlapping UV maps, more than the %d Blender can "
                     "hold; the surplus was merged into the last layer."
                     % (len(packed), max_layers))
        tail = packed[max_layers - 1:]
        keep = packed[:max_layers - 1]
        merged = {'name': tail[0]['name'], 'chunks': [], 'cov': set(), 'merged': 1}
        for e in tail:
            merged['chunks'].extend(e['chunks'])
        keep.append(merged)
        packed = keep
    out = []
    for e in packed:
        # a composite of several channels is no longer "the Lightmap chunk",
        # so it gets the neutral name rather than an arbitrary member's
        out.append((default_name if e['merged'] else e['name'], e['chunks']))
    return out, notes


def lwo_resolve_from_group(chunks, pi, vidx, fallback):
    """VMAP pass then VMAD pass, last match winning in each -- the engine's
    own order (ConvertLWOToModelSurfaces), scoped to one named map."""
    val = fallback
    for ch in chunks:
        if not ch['perpoly'] and vidx in ch['data']:
            val = ch['data'][vidx]
    for ch in chunks:
        if ch['perpoly'] and (pi, vidx) in ch['data']:
            val = ch['data'][(pi, vidx)]
    return val


def lwo_group_fallback(chunks, default):
    """What a corner no chunk in this group covers gets. The engine's own
    fallback is `tv = 0`, the first value of the flattened table; scoped to a
    group that is the first value of the group's first non-empty chunk, which
    is identical for a single-map file."""
    for ch in chunks:
        if ch['data']:
            return next(iter(ch['data'].values()))
    return default


def _lwo_file_topology(layer, pts, pivot, mat_of, base_rgba_of,
                       resolve_uv, resolve_color, corner_normals):
    """The mesh the FILE describes: one vertex per PNTS entry, in file order.

    No welding of any kind — see the archived changelog for why
    the engine does none either. A Blender vertex index is therefore the .lwo
    point index, points no polygon references stay as loose vertices, and
    quad/ngon polygons are kept as-is (the engine drops them, but seeing them
    is the whole point of an importer; import_model_file flags them).

    Returns (positions, faces). positions are pre-scale and already have the
    layer pivot added; each face is (material_name, corners) with corners in
    FILE order, each a (vertex_index, uv, rgba, normal_or_None).
    """
    positions = [p + pivot for p in pts]
    faces = []
    for pi, idxs in enumerate(layer['polys']):
        ptype = layer['poly_types'][pi]
        if ptype not in LWO_POLY_TYPES:
            continue
        if len(idxs) < 3 or any(i >= len(pts) for i in idxs):
            continue
        mn = mat_of(pi)
        base_rgba = base_rgba_of(mn)
        norms = corner_normals[pi] if (corner_normals and pi in corner_normals) else None
        corners = [(idxs[k],
                    resolve_uv(pi, idxs[k]),
                    resolve_color(pi, idxs[k], base_rgba),
                    norms[k] if norms is not None else None,
                    (pi, idxs[k]))
                   for k in range(len(idxs))]
        faces.append((mn, corners, LWO_POLY_TYPES.index(ptype), pi))
    return positions, faces


def _lwo_engine_render_topology(layer, pts, pivot, mat_of, base_rgba_of,
                                resolve_uv, resolve_color, corner_normals):
    """The mesh the ENGINE renders, rather than the one the file describes.

    A faithful port of idRenderModelStatic::ConvertLWOToModelSurfaces'
    non-fastLoad path (renderer/Model.cpp) plus the two steps of
    R_CleanupTriangles (renderer/tr_trisurf.cpp) that actually change
    topology. In order:

      1. vRemap  — weld the PNTS list at r_slopVertex (0.01), keeping the
                   first point of each cluster. "It seems like the tools our
                   artists are using often generate verts and texcoords
                   slightly separated that should be merged."
      2. tvRemap — weld the texcoord table at r_slopTexCoord (0.001).
      3. matchVert — one render vertex per unique (welded point, welded
                   texcoord, colour, normal) combination, so a UV seam or a
                   hard normal splits a shared point into several vertices.
                   Normals only take part when the file's own normals are in
                   play: the engine skips that test for a surface whose
                   material carries a renderBump command (normalsParsed
                   false), which is exactly the case this importer models by
                   turning "Use File Normals/Smoothing" off.
      4. R_RemoveDegenerateTriangles — drop any triangle two of whose
                   corners share one position. The engine tests silIndexes,
                   which weld on EXACT xyz; after step 1 every surviving
                   representative is more than epsilon from every other, so
                   "same exact position" and "same welded point" coincide.
      5. R_DuplicateMirroredVertexes — "bust apart any verts that are shared
                   by both positive and negative texture polarities, so
                   tangent space smoothing at the vertex doesn't degenerate."

    Only triangles take part; the engine refuses anything else outright
    ("model %s has too many verts for a poly! Make sure you triplet it
    down"). Steps this deliberately does NOT do, because they change no
    topology: R_CreateSilIndexes, R_IdentifySilEdges, R_CreateDupVerts, and
    the tangent/normal derivation.

    Two things this view cannot show faithfully, both worth knowing:

      * The 0.01 weld can collapse one triangle onto the vertex set of
        another, which a Blender mesh cannot hold. The engine keeps those —
        R_RemoveDuplicatedTriangles is commented out in R_CleanupTriangles on
        purpose, "this may remove valid overlapped transparent triangles" — so
        rather than drop them the face loop gives each its own copies of those
        (already welded) vertices, exactly as it does for duplicates in the
        file itself. gizmo1_hi.lwo splits 909 triangles this way and keeps its
        full 138316.
      * It only reflects what the engine really does for a model loaded the
        NORMAL way. A renderbump high poly goes through PartialInitFromFile
        instead, where fastLoad makes both remaps the identity and
        R_CleanupTriangles never runs at all, so none of this happens to it.

    Same (positions, faces) shape as _lwo_file_topology.
    """
    tri_polys = [pi for pi, idxs in enumerate(layer['polys'])
                 if layer['poly_types'][pi] in LWO_POLY_TYPES
                 and len(idxs) == 3
                 and not any(i >= len(pts) for i in idxs)]

    vlist = [p + pivot for p in pts]
    vremap = _lwo_weld_indices([(v.x, v.y, v.z) for v in vlist], _R_SLOP_VERTEX)

    # Per-corner attributes in file order. The engine welds its tvList (the
    # VMAP/VMAD value table in chunk order); welding the per-corner list
    # instead groups identically and can only pick a different representative
    # within the same cluster, i.e. a difference of at most one ULP.
    corner_uv, corner_col, corner_nrm = [], [], []
    for pi in tri_polys:
        idxs = layer['polys'][pi]
        base_rgba = base_rgba_of(mat_of(pi))
        norms = corner_normals[pi] if (corner_normals and pi in corner_normals) else None
        for k in range(3):
            corner_uv.append(tuple(resolve_uv(pi, idxs[k])))
            corner_col.append(tuple(resolve_color(pi, idxs[k], base_rgba)))
            corner_nrm.append(norms[k] if norms is not None else None)
    tvremap = _lwo_weld_indices(corner_uv, _R_SLOP_TEXCOORD)

    normals_parsed = corner_normals is not None
    normal_epsilon = 1.0 - _R_SLOP_NORMAL

    verts = []          # [welded point index, uv, rgba, normal]
    mv_hash = {}
    indexes = []
    poly_of = []
    c = 0
    for pi in tri_polys:
        idxs = layer['polys'][pi]
        for k in range(3):
            v = vremap[idxs[k]]
            tv = tvremap[c]
            nrm = corner_nrm[c]
            # the engine's key is the byte colour, not the float source
            colb = tuple(min(255, max(0, int(255 * x))) for x in corner_col[c])
            chain = mv_hash.setdefault((v, tv, colb), [])
            found = -1
            for mi in chain:
                if not normals_parsed:
                    break_here = True
                else:
                    have = verts[mi][3]
                    break_here = (have is not None and nrm is not None
                                  and have.dot(nrm) > normal_epsilon)
                if break_here:
                    found = mi
                    break
            if found < 0:
                found = len(verts)
                # st comes from tvList[mv->tv], i.e. the cluster's representative
                verts.append([v, corner_uv[tv], corner_col[c], nrm])
                chain.append(found)
            indexes.append(found)
            c += 1
        poly_of.append(pi)

    # ---- R_RemoveDegenerateTriangles ----
    keep = [t for t in range(len(poly_of))
            if len({verts[indexes[3 * t + j]][0] for j in range(3)}) == 3]

    # ---- R_DuplicateMirroredVertexes ----
    # R_FaceNegativePolarity's sign is taken on the engine's own st, whose t
    # is inverted relative to LightWave's (and Blender's) v -- so t is flipped
    # back here to get the same partition the engine gets.
    used = {}
    negative = {}
    for t in keep:
        a, b, d = (verts[indexes[3 * t + j]][1] for j in range(3))
        d0s, d0t = b[0] - a[0], (1.0 - b[1]) - (1.0 - a[1])
        d1s, d1t = d[0] - a[0], (1.0 - d[1]) - (1.0 - a[1])
        p = 1 if (d0s * d1t - d0t * d1s) < 0 else 0
        negative[t] = p
        for j in range(3):
            used.setdefault(indexes[3 * t + j], [False, False])[p] = True

    remap_negative = {}
    for vi in sorted(used):
        flags = used[vi]
        if flags[0] and flags[1]:
            remap_negative[vi] = len(verts)
            verts.append(list(verts[vi]))
    if remap_negative:
        for t in keep:
            if negative[t] != 1:
                continue
            for j in range(3):
                vi = indexes[3 * t + j]
                if vi in remap_negative:
                    indexes[3 * t + j] = remap_negative[vi]

    positions = [vlist[rv[0]] for rv in verts]
    faces = []
    for t in keep:
        corners = []
        for j in range(3):
            vi = indexes[3 * t + j]
            rv = verts[vi]
            corners.append((vi, rv[1], rv[2], rv[3], None))
        pt = layer['poly_types'][poly_of[t]]
        faces.append((mat_of(poly_of[t]), corners, LWO_POLY_TYPES.index(pt),
                      poly_of[t]))
    return positions, faces


def _lwo_apply_engine_overrides(mesh, faces, positions, scale, loop_keys,
                                rb_mats, un_mats, identity_by_name):
    """Replace the file's normals where the engine would throw them away.

    A renderBump material gets R_DeriveTangents' regenerated normals; an
    unsmoothedTangents material gets the dominant triangle's. Both are
    computed per material, because a srfTriangles_t holds one material and
    neither sum crosses that boundary.
    """
    by_mat = {}
    corner = 0
    for mn, corners, _ptype, _src in faces:
        ident = identity_by_name.get(mn)
        decl = engine_canonical_decl((ident or {}).get('decl') or mn)
        if len(corners) == 3 and (decl in rb_mats or decl in un_mats):
            by_mat.setdefault(decl, []).append((corner, [c[0] for c in corners]))
        corner += len(corners)

    if not by_mat:
        return
    pts = [Vector((p.x * scale, p.y * scale, p.z * scale)) for p in positions]
    replaced = {}
    for decl, entries in by_mat.items():
        tris = [tuple(idx) for _at, idx in entries]
        fn = (engine_dominant_normals(pts, tris) if decl in un_mats
              else engine_regenerated_normals(pts, tris))
        for k, (at, _idx) in enumerate(entries):
            for j in range(3):
                replaced[at + j] = fn[k * 3 + j]

    attr = mesh.attributes.get("temp_normal")
    if attr is None or len(loop_keys) != len(mesh.loops):
        return
    # `faces` is in FILE corner order; the loops it produced were reversed
    flat = []
    li = 0
    corner = 0
    for _mn, corners, _ptype, _src in faces:
        n = len(corners)
        for j in range(n):
            src_corner = corner + (n - 1 - j)      # undo the winding reversal
            v = replaced.get(src_corner)
            if v is None:
                v = attr.data[li].vector
            flat.extend((v[0], v[1], v[2]))
            li += 1
        corner += n
    if li == len(mesh.loops):
        attr.data.foreach_set('vector', flat)


def _lwo_apply_face_tags(mesh, layer, faces):
    """PTAG SMGP / PART onto editable INT face attributes.

    Only written when the file actually used them, so an ordinary model
    carries no extra attributes. They are editable on purpose -- an artist can
    retag smoothing islands in the spreadsheet -- and the idTech4 export
    validator is what catches a state the format cannot express.

    *faces* is the topology builder's list, in the same order the faces were
    created, so entry i is polygon i of the mesh.
    """
    if len(faces) != len(mesh.polygons):
        return                      # a face was refused; indices no longer line up
    for attr_name, table in ((LWO_SMGP_ATTR, layer.get('ptag_smgp') or {}),
                             (LWO_PART_ATTR, layer.get('ptag_part') or {})):
        if not table:
            continue
        vals = [int(table.get(f[3], 0)) for f in faces]
        if not any(vals):
            continue
        attr = mesh.attributes.new(attr_name, 'INT', 'FACE')
        attr.data.foreach_set('value', vals)


POINT_MAP_KEY = 'idtech4_point_maps'


def _lwo_apply_point_maps(mesh, layer, pts):
    """Park WGHT / PICK / MORF / SPOT on the mesh.

    WGHT and PICK become float POINT attributes here, which is the lossless
    store: a Blender vertex GROUP clamps to 0..1 and LightWave weight maps are
    not so restricted. apply_point_maps_to_object then adds the matching
    vertex group for usability, and export reads the attribute back.

    MORF (relative displacement) and SPOT (absolute position) become shape
    keys, but shape keys hang off an Object rather than a Mesh, so their
    coordinates are parked here and applied later.
    """
    maps = layer.get('point_maps') or []
    if not maps or not mesh.vertices:
        return
    meta = []
    nverts = len(mesh.vertices)
    for m in maps:
        data = m['data']
        if not data:
            continue
        kind = m['type'].decode('latin1')
        nm = (m['name'] or '').strip() or kind
        if m['type'] in (b'WGHT', b'PICK'):
            if mesh.attributes.get(nm) is not None:
                nm = '%s.%s' % (nm, kind)
            vals = [0.0] * nverts
            outside = False
            for vidx, v in data.items():
                if 0 <= vidx < nverts:
                    vals[vidx] = float(v[0])
                    if not (0.0 <= vals[vidx] <= 1.0):
                        outside = True
            attr = mesh.attributes.new(nm, 'FLOAT', 'POINT')
            attr.data.foreach_set('value', vals)
            meta.append({'name': nm, 'kind': kind, 'outside': outside})
            kinds = dict(mesh.get('idtech4_point_map_kinds') or {})
            kinds[nm] = kind
            mesh['idtech4_point_map_kinds'] = kinds
        elif m['type'] in (b'MORF', b'SPOT'):
            base = [tuple(v.co) for v in mesh.vertices]
            moved = []
            for i, co in enumerate(base):
                d = data.get(i)
                if d is None:
                    moved.append(co)
                elif m['type'] == b'MORF':
                    # relative displacement, same axis order as PNTS
                    moved.append((co[0] + d[0], co[1] + d[2], co[2] + d[1]))
                else:
                    moved.append((d[0], d[2], d[1]))
            mesh['%s:coords' % nm] = [c for co in moved for c in co]
            meta.append({'name': nm, 'kind': kind, 'outside': False})
    if meta:
        mesh[POINT_MAP_KEY] = json.dumps(meta)


def apply_point_maps_to_object(obj):
    """Finish what _lwo_apply_point_maps parked: vertex groups and shape keys.

    Called once the Object exists. Returns the number of maps realised.
    """
    mesh = obj.data
    raw = mesh.get(POINT_MAP_KEY)
    if not raw:
        return 0
    try:
        meta = json.loads(raw)
    except Exception:
        meta = []
    done = 0
    for entry in meta:
        nm, kind = entry.get('name'), entry.get('kind')
        if kind in ('WGHT', 'PICK'):
            attr = mesh.attributes.get(nm)
            if attr is None:
                continue
            vals = [0.0] * len(mesh.vertices)
            attr.data.foreach_get('value', vals)
            vg = obj.vertex_groups.get(nm) or obj.vertex_groups.new(name=nm)
            for i, w in enumerate(vals):
                if w != 0.0:
                    # clamped by Blender; the float attribute above keeps the
                    # true value for anything outside 0..1
                    vg.add([i], min(max(w, 0.0), 1.0), 'REPLACE')
            done += 1
        elif kind in ('MORF', 'SPOT'):
            coords = mesh.get('%s:coords' % nm)
            if not coords:
                continue
            if mesh.shape_keys is None:
                obj.shape_key_add(name='Basis', from_mix=False)
            key = obj.shape_key_add(name=nm, from_mix=False)
            flat = list(coords)
            if len(flat) == len(mesh.vertices) * 3:
                key.data.foreach_set('co', flat)
            del mesh['%s:coords' % nm]
            done += 1
    try:
        del mesh[POINT_MAP_KEY]
    except Exception:
        pass
    return done


def build_lwo_meshes(filepath, scale, first_layer_only=False, name_hint=None,
                      engine_render_mesh=False,
                      shading=SHADING_ENGINE, shading_overrides=None):
    """
    Build Blender meshes from an .lwo (LWO2) file, one mesh PER LWO LAYER
    (the same granularity as one-mesh-per-GEOMOBJECT for .ase files, and
    matching how the source file's author actually split the model into
    separate pieces).

    IMPORTANT: nothing here welds vertices. Every PNTS entry becomes one
    BMVert, in file order, so a Blender vertex index IS the .lwo point index
    and a point no polygon references survives as a loose vertex. That
    matches the engine on both of its load paths (see the archived
    changelog), and it is what makes the import lossless: an earlier
    version pooled BMVerts by position rounded to 4 decimals and then
    swallowed every polygon that collapsed onto an existing vertex set,
    which cost a real asset 514 points and 1024 polygons without a word.
    Material-index assignment is still scoped strictly to a single layer. An
    even earlier version pooled geometry by MATERIAL NAME across the *entire
    file*, which flattened two unrelated layers into one mesh whenever they
    happened to share a material/tag name (common for modular or symmetric
    pieces built at/near the local origin before being moved into place by
    their own layer pivot).

    *engine_render_mesh*: build the mesh the ENGINE renders instead of the
    mesh the file describes — the 0.01 position weld, the 0.001 texcoord
    weld, one vertex per unique (point, texcoord, colour, normal), degenerate
    triangles removed and mirrored-UV vertices split. Useful for seeing where
    idTech4 will actually split a model; useless as a source of truth about
    the file. See _lwo_engine_render_topology. Off by default.

    Adds each layer's pivot point back onto its raw PNTS data before use
    (see parse_lwo_file) — otherwise a layer whose pivot was moved to
    place it within the overall model would import sitting at/near its
    own local origin instead of its real position.

    *first_layer_only*: when True, only the FIRST layer in the file (the
    first LAYR chunk encountered) is built — matching real idTech4/Doom 3
    engine behavior. Verified directly against the Doom 3 GPL source
    (renderer/Model.cpp, idRenderModelStatic::ConvertLWOToModelSurfaces):
    the engine does `lwLayer* layer = lwo->layer;` — literally the head of
    the layer linked list, i.e. the first LAYR chunk in the file — and
    never reads any other layer. Every additional layer is fully parsed
    by the low-level chunk reader but then simply never touched again.
    So any extra layer in a .lwo (e.g. leftover/reference geometry from
    a modeling-tool copy/paste that never got cleaned up) is dead data
    in-engine: it's never rendered, regardless of its "hidden" flag.

    Local space (identity rotation/translation), only *scale* applied.
    Returns [(layer_name, mesh, [material_name, ...]), ...]. layer_name
    is the layer's real name when the file set one; otherwise it falls
    back to *name_hint* if given (the model's full reference path as
    written in the .map "model" key, e.g.
    "models/mapobjects/lab/diamondbox/diamondbox.lwo" — see the .map
    importer's use of this function), or the model's own bare filename
    when no hint is given (e.g. for a standalone .lwo import with no
    .map reference to draw from). Falling back to something derived
    from the file rather than a generic placeholder matters because
    MANY real single-layer .lwo files never set a layer name at all,
    and a placeholder identical across every one of them would be
    meaningless once multiple different models are loaded into the
    same Blender scene (e.g. via a .map file referencing dozens of
    distinct, all-unnamed-layer models). A file with more than one
    unnamed layer gets the layer index appended to keep them apart.

    *shading*: one of SHADING_ITEMS, defaulting to SHADING_ENGINE.
    ENGINE and FILE are identical for a .lwo (the format stores no
    normals, so the file's smoothing angle is both): both replicate the
    real engine's own shading — per-corner normals from the file's
    smoothing-angle-averaged data (see _lwo_compute_corner_normals) — as
    Blender custom split normals, instead of Blender's generic
    winding-only normal_update(). That is what the engine does for any
    material WITHOUT a renderBump command; a material WITH one has its
    file normals discarded in favor of freshly regenerated ones, which
    reaches here through *shading_overrides*. SMOOTH and FLAT write no
    custom normals at all.
    """
    mode = resolve_shading_mode(shading)
    use_file_normals = mode in (SHADING_ENGINE, SHADING_FILE)
    rb_mats, un_mats = shading_overrides or (set(), set())
    layers, tags, surfaces = parse_lwo_file(filepath)
    surface_sman = {n: float(sf.get('smooth', 0.0)) for n, sf in surfaces.items()}
    surface_extra = {n: sf.get('extra') or b'' for n, sf in surfaces.items()}
    surface_colr = {n: tuple(sf.get('color') or (1.0, 1.0, 1.0))
                    for n, sf in surfaces.items()}
    tag_identities = disambiguate_surface_names(
        [material_identity_from_lwo(t) for t in tags])
    raw_tags = list(tags)          # what `surfaces` is keyed by
    tags = [i['display'] for i in tag_identities]
    identity_by_name = {i['display']: i for i in tag_identities}
    if first_layer_only:
        layers = layers[:1]

    # Fallback base for layers with no real name: prefer the model's full
    # reference path as written in the .map "model" key (name_hint) over
    # just the bare filename, since the reference path is what actually
    # identifies which model this is among many that could share a
    # filename in different folders (e.g. two different "diamondbox.lwo"
    # under different models/mapobjects/... subfolders).
    fallback_base = (name_hint or '').strip().strip('"') or os.path.splitext(os.path.basename(filepath))[0]

    results = []
    for li, layer in enumerate(layers):
        pts   = layer['points']
        pivot = layer['pivot']   # add back on: PNTS data is pivot-relative (see parse_lwo_file)

        # Resolve each polygon corner's UV exactly the way the real engine
        # does — verified directly against renderer/Model.cpp,
        # idRenderModelStatic::ConvertLWOToModelSurfaces:
        #
        #   tv = 0                                    // fallback, see below
        #   for each chunk covering this POINT (VMAP-type only):
        #       if chunk applies to this point: tv = chunk's value   // last wins
        #   for each chunk covering this POLY CORNER (VMAD-type only):
        #       if chunk applies to this corner: tv = chunk's value  // last wins,
        #                                                             // overrides point stage
        #
        # i.e. two separate passes — every VMAP-type chunk first (in file
        # order, last match wins), then every VMAD-type chunk on top (in
        # file order, last match wins, overriding the VMAP pass whenever
        # any VMAD chunk applies to that corner at all). There is NO
        # selection by chunk name anywhere in the real algorithm — every
        # TXUV chunk found in the file takes part, which is exactly why
        # a real exporter pattern like "one discontinuous UV channel per
        # material" (confirmed directly against delelevlf.lwo — two
        # channels, zero overlap, together covering the whole mesh) just
        # works: every corner is covered by exactly one chunk, so which
        # pass or order "wins" never actually matters for that file.
        #
        # Real fallback when NEITHER stage applies to a corner at all is
        # `tv = 0`, an index into the engine's own combined/flattened UV
        # array — i.e. the first vertex value of the very first TXUV
        # chunk in the file (or (0,0) if the layer has no TXUV data at
        # all, matching the engine's own zero-filled single-vertex
        # fallback in that case). Replicated here as fallback_uv.
        uv_groups, _uvnotes = lwo_pack_vmap_groups(
            lwo_vmap_groups(layer['uv_chunks'], 'UVMap'), layer['polys'], 'UVMap')
        for _n in _uvnotes:
            print("idTech4 LWO import: %s: %s" % (os.path.basename(filepath), _n))
        primary_uv = uv_groups[0][1] if uv_groups else []
        fallback_uv = lwo_group_fallback(primary_uv, (0.0, 0.0))

        def resolve_uv(pi, vidx, chunks=primary_uv, fallback_uv=fallback_uv):
            return lwo_resolve_from_group(chunks, pi, vidx, fallback_uv)

        # Vertex color: same two-pass (point VMAP, then corner VMAD
        # override) resolution as UV above — see parse_lwo_file. Falls
        # back to the polygon's own SURFACE base color (COLR, alpha 1.0)
        # rather than a fixed white, matching Model.cpp exactly (it seeds
        # `color[]` from `lwoSurf->color.rgb` before any VMAP override).
        color_groups, _colnotes = lwo_pack_vmap_groups(
            lwo_vmap_groups(layer['color_chunks'], 'Col'), layer['polys'], 'Col')
        for _n in _colnotes:
            print("idTech4 LWO import: %s: %s" % (os.path.basename(filepath), _n))
        primary_col = color_groups[0][1] if color_groups else []

        def resolve_color(pi, vidx, fallback_rgba, chunks=primary_col):
            return lwo_resolve_from_group(chunks, pi, vidx, fallback_rgba)

        corner_normals = (_lwo_compute_corner_normals(layer, raw_tags, surfaces)
                          if use_file_normals else None)

        def mat_of(pi, layer=layer, tags=tags):
            tag_idx = layer['ptag_surf'].get(pi)
            return tags[tag_idx] if (tag_idx is not None and tag_idx < len(tags)) else 'unknown'

        def base_rgba_of(mn, surfaces=surfaces):
            surf = surfaces.get(mn)
            return (*(surf['color'] if surf else (0.78431, 0.78431, 0.78431)), 1.0)

        builder = (_lwo_engine_render_topology if engine_render_mesh
                   else _lwo_file_topology)
        positions, faces = builder(layer, pts, pivot, mat_of, base_rgba_of,
                                   resolve_uv, resolve_color, corner_normals)

        bm = bmesh.new()
        uv_lay  = bm.loops.layers.uv.new("UVMap")
        col_lay = bm.loops.layers.float_color.new("Col")
        nrm_lay = bm.loops.layers.float_vector.new("temp_normal") if corner_normals else None
        mat_idx_map = {}   # material name -> material_index, scoped to THIS layer's mesh
        any_normals_written = False
        refused = 0
        split_count = 0
        split_source = []       # (BMVert, originating file point index)

        verts = [bm.verts.new(Vector((p.x * scale, p.y * scale, p.z * scale)))
                 for p in positions]

        face_ptypes = []
        loop_keys = []          # (poly, point) per created loop, in loop order
        for _face_index, (mn, corners, ptype, _srcpoly) in enumerate(faces):
            # Reversed vs. file order — see parse_lwo_file's docstring
            # (axis-conversion section) for why the real engine's own
            # winding ends up opposite what keeping file order would give.
            # UV/colour/normal ride along on each corner, so they stay in
            # lockstep with the reversal automatically.
            corners = list(reversed(corners))
            try:
                f = bm.faces.new([verts[c[0]] for c in corners])
            except ValueError:
                # Another polygon already uses this exact vertex set, which a
                # Blender mesh cannot hold -- but overlapping geometry is fine,
                # so give this one its own copies. They go on the end, so every
                # index below len(pts) is still its own file point.
                dup = []
                for c in corners:
                    src = verts[c[0]]
                    nv = bm.verts.new(src.co.copy())
                    split_source.append((nv, int(c[0])))
                    dup.append(nv)
                try:
                    f = bm.faces.new(dup)
                except ValueError:
                    refused += 1
                    continue
                split_count += 1
            face_ptypes.append(ptype)

            if mn not in mat_idx_map:
                mat_idx_map[mn] = len(mat_idx_map)
            f.material_index = mat_idx_map[mn]

            for c in corners:
                loop_keys.append(c[4])
            for loop, c in zip(f.loops, corners):
                loop[uv_lay].uv = c[1]
                loop[col_lay] = c[2]
                if nrm_lay is not None and c[3] is not None:
                    loop[nrm_lay] = c[3]
                    any_normals_written = True

        if split_count:
            print("idTech4 LWO import: %s layer %d: %d polygon(s) share a vertex "
                  "set with another, which a Blender mesh cannot hold, so each "
                  "got its own copies of those vertices. The engine draws these "
                  "(R_RemoveDuplicatedTriangles is disabled), so they are real "
                  "geometry: often a double-sided face or a second material on "
                  "one triangle. Export folds the copies back while they stay "
                  "coincident."
                  % (os.path.basename(filepath), li, split_count))
        if refused:
            print("idTech4 LWO import: %s layer %d: %d polygon(s) could not be "
                  "created at all" % (os.path.basename(filepath), li, refused))

        if not bm.faces:
            bm.free()
            continue

        bm.verts.index_update()
        # resolve to plain indices while the bmesh is still alive
        split_source = [(bv.index, origin) for bv, origin in split_source]
        bm.normal_update()
        _pending_ptypes = face_ptypes
        if layer['name']:
            name = layer['name']
        elif len(layers) > 1:
            name = f"{fallback_base}_layer{li}"
        else:
            name = fallback_base
        mesh = bpy.data.meshes.new(name)
        bm.to_mesh(mesh)
        bm.free()
        mesh.validate(clean_customdata=False)
        mesh.update()

        # Every named LightWave map beyond the first becomes its own Blender
        # layer. The first was already filled through the bmesh above; these
        # are written straight onto the mesh in loop order, which bmesh
        # preserves through to_mesh().
        if loop_keys and len(loop_keys) == len(mesh.loops):
            for gname, gchunks in uv_groups[1:]:
                fb = lwo_group_fallback(gchunks, (0.0, 0.0))
                lay_uv = mesh.uv_layers.new(name=gname, do_init=False)
                if lay_uv is None:
                    print("idTech4 LWO import: %s: could not add UV layer %r "
                          "(Blender allows %d)"
                          % (os.path.basename(filepath), gname,
                             MAX_BLENDER_UV_LAYERS))
                    continue
                flat = []
                for key in loop_keys:
                    v = fb if key is None else lwo_resolve_from_group(
                        gchunks, key[0], key[1], fb)
                    flat.extend((v[0], v[1]))
                lay_uv.uv.foreach_set('vector', flat)
            for gname, gchunks in color_groups[1:]:
                lay_col = mesh.color_attributes.new(name=gname, type='FLOAT_COLOR',
                                                    domain='CORNER')
                if lay_col is None:
                    print("idTech4 LWO import: %s: could not add colour layer %r"
                          % (os.path.basename(filepath), gname))
                    continue
                flat = []
                for key in loop_keys:
                    v = ((1.0, 1.0, 1.0, 1.0) if key is None
                         else lwo_resolve_from_group(gchunks, key[0], key[1],
                                                     (1.0, 1.0, 1.0, 1.0)))
                    flat.extend((v[0], v[1], v[2], v[3] if len(v) > 3 else 1.0))
                lay_col.data.foreach_set('color', flat)
            mesh.update()

        # After the extra maps, not before: color_attributes.new() claims the
        # active and render slots for whatever it created LAST, so running
        # this first would hand a model's second colour map the slot its
        # primary one is meant to hold. The engine only ever reads the
        # primary one.
        mark_primary_color_attribute(mesh)

        if split_source:
            src = [-1] * len(mesh.vertices)
            for vidx, origin in split_source:
                if 0 <= vidx < len(src):
                    src[vidx] = origin
            attr = mesh.attributes.new(LWO_SOURCE_POINT_ATTR, 'INT', 'POINT')
            attr.data.foreach_set('value', src)

        blob = layer.get('escrow') or []
        if blob:
            mesh[LWO_ESCROW_PROP] = base64.b64encode(b''.join(blob)).decode('ascii')
            mesh[LWO_ESCROW_SHAPE_PROP] = [len(pts), len(layer['polys'])]

        if mode == SHADING_ENGINE and (rb_mats or un_mats) and corner_normals:
            _lwo_apply_engine_overrides(mesh, faces, positions, scale,
                                        loop_keys, rb_mats, un_mats,
                                        identity_by_name)

        _lwo_apply_face_tags(mesh, layer, faces)
        _lwo_apply_point_maps(mesh, layer, pts)

        # Which POLS chunk each face came from, editable in the spreadsheet.
        # Only stored when the layer actually mixes types or uses a
        # non-default one -- an all-FACE mesh, the overwhelmingly common
        # case, carries no extra attribute at all.
        if _pending_ptypes and any(t != 0 for t in _pending_ptypes) \
                and len(_pending_ptypes) == len(mesh.polygons):
            attr = mesh.attributes.new(LWO_POLY_TYPE_ATTR, 'INT', 'FACE')
            attr.data.foreach_set('value', _pending_ptypes)

        if any_normals_written:
            # Every loop that exists got a normal written above (uv/col/
            # normal are all written together, right after a face is
            # successfully created — see the per-polygon loop) — pull
            # them back out in mesh.loops order and hand them to Blender
            # as custom split normals, then drop the temp attribute.
            mark_smooth_for_custom_normals(mesh)
            nrm_attr = mesh.attributes.get("temp_normal")
            if nrm_attr is not None:
                loop_normals = [tuple(nrm_attr.data[i].vector) for i in range(len(mesh.loops))]
                mesh.normals_split_custom_set(loop_normals)
                # Re-fetch by name rather than reusing nrm_attr: on Blender
                # 4.5, normals_split_custom_set() invalidates the handle
                # (its .name reads back empty, and removing an attribute by
                # an empty name raises "The attribute name must not be
                # empty"). 5.0+ happens not to invalidate it, which is why
                # this only broke on 4.5.
                nrm_attr = mesh.attributes.get("temp_normal")
                if nrm_attr is not None:
                    mesh.attributes.remove(nrm_attr)
                mesh.update()

        mat_names = list(mat_idx_map.keys())
        _idents = []
        for n in mat_names:
            ident = dict(identity_by_name.get(n) or material_identity_from_lwo(n))
            raw = ident.get('raw') or n
            if raw in surface_sman:
                ident['sman'] = surface_sman[raw]
            if surface_extra.get(raw):
                ident['surf_extra'] = base64.b64encode(
                    surface_extra[raw]).decode('ascii')
            if raw in surface_colr:
                ident['colr'] = list(surface_colr[raw])
            _idents.append(ident)
        stash_material_identities(mesh, _idents)
        results.append((name, mesh, mat_names))

    return results


# ── Binary render model cache (.base / .blwo) ──────────────────────
# Doom 3 BFG can convert a plaintext .ase/.lwo (or .obj/.ma) source
# model into a binary "render model cache" for faster loading:
# generated/rendermodels/<name>.b<ext> (.ase -> .base, .lwo -> .blwo).
# Unlike the MD5 binary cache (which has its own dedicated per-format
# block — see idTech4_MD5_Tools.py's parse_bmd5mesh), a compiled ASE/
# LWO model is just idRenderModelStatic's OWN generic cache: every
# format-specific loader (Model_ase.cpp/Model_lwo.cpp) only ever
# produces ordinary idRenderModelStatic surfaces, and it's
# idRenderModelStatic::WriteBinaryModel/LoadBinaryModel (Model.cpp)
# that reads/writes them — verified directly against the GPL source
# (RBDOOM-3-BFG's neo/renderer/Model.cpp). That means .base and .blwo
# share one parser below, with no branch on which extension produced
# the cache — and this same parser would work unchanged for a
# compiled .bobj/.bma too, if this addon ever grew import for those.
#
# Layout verified byte-for-byte against every sample cache file found
# under this repo's generated/rendermodels/ (119 files, every one
# ending at exactly the right byte offset).
#
# Two on-disk versions exist, selected by the header's low version
# byte:
#   BRM_VERSION_BFG (108) — the original BFG format; every sample file
#     found in this repo is this version.
#   BRM_VERSION_MOC_DATA (110) — a newer engine addition that drops
#     several BFG-only fields (ambientViewCount, a stubbed-out
#     preLightShadowVertexes block, a stubbed-out silhouette-edge
#     block, 3 trailing stub ints) in favor of appending Masked
#     Occlusion Culling data instead. Which fields are present is a
#     hard branch on the version byte in Model.cpp itself (unlike
#     .bmd5mesh's header, where the version is safely ignorable — see
#     idTech4_MD5_Tools.py's BRM_TAG note), so this parser branches on
#     it too, per surface, rather than assuming one fixed layout.
#
# Endianness follows the same idFile convention documented in
# idTech4_MD5_Tools.py's _BinReader: WriteVec3/WriteVec4/WriteFloat/
# WriteString/WriteInt are native little-endian no-ops on a PC host;
# WriteBig<T>/WriteBigArray are always true big-endian.
BRM_TAG              = (ord('B') << 16) | (ord('R') << 8) | ord('M')
BRM_VERSION_BFG      = 108
BRM_VERSION_MOC_DATA = 110


class _BRMReader:
    """Sequential cursor over a binary render-model file's bytes — same
    native-LE/true-BE split as idTech4_MD5_Tools.py's _BinReader (kept
    as its own copy here rather than importing across addons, matching
    this module's existing policy — see the module docstring above —
    of not depending on another addon's internals)."""
    __slots__ = ('data', 'pos')

    def __init__(self, data):
        self.data = data
        self.pos  = 0

    def _read(self, n):
        p = self.pos
        end = p + n
        if end > len(self.data):
            raise EOFError("Unexpected end of file while parsing binary render model data")
        self.pos = end
        return self.data[p:end]

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

    def u32_be(self):
        return struct.unpack_from('>I', self._read(4))[0]

    def i32_be(self):
        return struct.unpack_from('>i', self._read(4))[0]

    def i64_be(self):
        return struct.unpack_from('>q', self._read(8))[0]

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


def _brm_half_to_float(h):
    """Decode a GPU half-float bit pattern — identical formula to
    idTech4_MD5_Tools.py's _half_to_float (F16toF32 in idlib/geometry/
    DrawVert.h); kept as its own copy for the same reason as
    _BRMReader above."""
    e = (h >> 10) & 0x1F
    m = h & 0x3FF
    s = -1.0 if (h & 0x8000) else 1.0
    if 0 < e < 31:
        return s * (2.0 ** (e - 15)) * (1.0 + m / 1024.0)
    elif m == 0:
        return s * 0.0
    return s * (2.0 ** -14) * (m / 1024.0)


def _brm_byte_to_float(b):
    """VERTEX_BYTE_TO_FLOAT(x) from idlib/geometry/DrawVert.h — decodes
    one packed normal byte back to a [-1, 1] float component."""
    return b * (2.0 / 255.0) - 1.0


def parse_binary_render_model(filepath):
    """Parse a compiled .base/.blwo (idRenderModelStatic::WriteBinaryModel,
    renderer/Model.cpp) — the same generic binary render-model cache
    used for any compiled static model alike (see module note above).

    Returns a list of surfaces:
        [{'shader': str, 'positions': [(x,y,z),...], 'uvs': [(u,v),...],
          'normals': [(x,y,z),...] or None, 'colors': [(r,g,b,a),...],
          'tris': [(a,b,c),...]}, ...]

    'normals' is None for a surface whose generateNormals flag is set
    (the engine regenerates its normals rather than trusting the
    file's own — the file's normal bytes for such a surface are never
    actually written meaningfully) — callers should fall back to
    computed normals for that surface instead of trusting file data.
    """
    with open(filepath, 'rb') as fh:
        data = fh.read()
    r = _BRMReader(data)

    magic = r.u32_be()
    if (magic >> 8) != BRM_TAG:
        raise ValueError(
            "Not a Doom 3 BFG binary render model (bad header magic) — "
            "expected a .base/.blwo produced by the game's binary model cache")
    version = magic & 0xFF
    has_moc = (version != BRM_VERSION_BFG)   # see module note: only 108 keeps the BFG-only fields

    r.i64_be()   # timeStamp — unused for import

    num_surfaces = r.i32_be()
    surfaces = []
    for _ in range(num_surfaces):
        r.i32_be()          # id — unused, surface order is preserved as-is
        shader = r.string()
        if not r.byte():    # isGeometry
            continue

        r.vec3_le(); r.vec3_le()   # bounds[0], bounds[1] — unused, Blender recomputes
        if not has_moc:
            r.i32_be()      # ambientViewCount (BFG-only)
        generate_normals = bool(r.byte())
        r.byte(); r.byte(); r.byte()   # tangentsCalculated/perfectHull/referencedIndexes — unused

        num_verts   = r.i32_be()
        num_in_file = r.i32_be()
        positions = []
        uvs       = []
        normals   = []
        colors    = []
        if num_in_file > 0:
            for _ in range(num_verts):
                positions.append(r.vec3_le())
                s = r.u16_be(); t = r.u16_be()
                # V is flipped here (1 - t), NOT kept raw like the text .ase/
                # .lwo parsers' UVs — verified empirically by comparing a
                # compiled model against its own text source face-by-face
                # (tbox4.ase/.base, armor_shard.lwo/.blwo): the compiler's
                # own internal V axis (baked into every sample file found)
                # is the OPPOSITE of the raw Max/LWO V this addon's text
                # importers already use as-is. Skipping this flip doesn't
                # just look upside-down — on a multi-region UV atlas (e.g.
                # a box unwrap) it makes each face sample a DIFFERENT
                # region entirely, which is what actually gets reported as
                # "textures on the wrong faces".
                uvs.append((_brm_half_to_float(s), 1.0 - _brm_half_to_float(t)))
                normal = r.bytes_raw(4)
                r.bytes_raw(4)      # tangent — unused, Blender recomputes
                color  = r.bytes_raw(4)
                r.bytes_raw(4)      # color2 — skinning weights on an MD5 mesh; unused/zero here
                normals.append((_brm_byte_to_float(normal[0]),
                                 _brm_byte_to_float(normal[1]),
                                 _brm_byte_to_float(normal[2])))
                colors.append((color[0] / 255.0, color[1] / 255.0,
                               color[2] / 255.0, color[3] / 255.0))

        if not has_moc:
            num_shadow = r.i32_be()   # preLightShadowVertexes — BFG-only, always stubbed out even by the engine itself
            if num_shadow > 0:
                for _ in range(num_shadow):
                    r.f32_le(); r.f32_le(); r.f32_le(); r.f32_le()

        num_indexes = r.i32_be()
        indexes = r.u16_be_array(num_indexes) if num_indexes > 0 else []
        num_sil = r.i32_be()
        if num_indexes > 0 and num_sil > 0:
            r.u16_be_array(num_indexes)   # silIndexes — unused

        num_mirrored = r.i32_be()
        if num_mirrored > 0:
            r.i32_be_array(num_mirrored)   # mirroredVerts — unused

        num_dup = r.i32_be()
        if num_dup > 0:
            r.i32_be_array(num_dup * 2)    # dupVerts — unused

        if not has_moc:
            num_sil_edges = r.i32_be()     # silEdges — BFG-only, always stubbed out even by the engine itself
            if num_sil_edges > 0:
                for _ in range(num_sil_edges):
                    r.u16_be(); r.u16_be(); r.u16_be(); r.u16_be()

        if r.byte():                        # dominantTris != NULL
            for _ in range(num_verts):
                r.u16_be(); r.u16_be()
                r.f32_le(); r.f32_le(); r.f32_le()

        if not has_moc:
            r.i32_be(); r.i32_be(); r.i32_be()   # 3 trailing BFG-only stub ints

        if has_moc:
            num_moc_verts = r.i32_be()
            if num_moc_verts > 0:
                for _ in range(num_moc_verts):
                    r.f32_le(); r.f32_le(); r.f32_le(); r.f32_le()   # mocVerts — unused
            num_moc_indexes = r.i32_be()
            if num_moc_indexes > 0:
                r.bytes_raw(4 * num_moc_indexes)   # mocIndexes (uint32) — unused

        # Last two corners swapped (equivalent to reversing the whole
        # triangle — a triangle only has 2 distinct windings, and every
        # rotation of "reversed" is the same winding) — verified
        # empirically the same way as the V-flip above: matching
        # compiled triangles back to their text-format source by
        # position shows the file's raw index order is consistently the
        # OPPOSITE winding from what this addon's Blender-facing
        # convention (and the source's own recorded face normals) want,
        # for both a compiled .ase (tbox4) and a compiled .lwo
        # (armor_shard) — so this is the generic engine loader's own
        # doing, not something specific to one source format. Per-vertex
        # data (position/uv/normal/color) is unaffected either way, since
        # it's keyed by vertex index, not by a corner's position in the
        # triangle.
        tris = [(indexes[i], indexes[i + 2], indexes[i + 1])
                for i in range(0, len(indexes) - 2, 3)]
        surfaces.append({
            'shader':    shader,
            'positions': positions,
            'uvs':       uvs,
            'normals':   None if generate_normals else normals,
            'colors':    colors,
            'tris':      tris,
        })

    # Trailing idRenderModelStatic fields (model bounds, name, flags) —
    # unused for import, but consumed anyway so the cursor lands at
    # exactly end-of-file (verified against every sample file).
    r.vec3_le(); r.vec3_le()
    r.i32_be(); r.i32_be(); r.i32_be()
    r.string()
    for _ in range(9):
        r.byte()

    return surfaces


def build_binary_meshes(filepath, scale, name_hint=None):
    """Build a single Blender mesh from a compiled .base/.blwo binary
    render model cache. Unlike build_ase_meshes/build_lwo_meshes (one
    mesh per GEOMOBJECT/layer), the binary cache has already been
    flattened by the engine into a flat list of surfaces — one per
    material — with no source-object grouping left to recover, so
    every surface is combined into ONE mesh here, each keeping its own
    material slot (matching how the real compiled model actually
    renders: one idRenderModel made of several modelSurface_t's, not
    several separate objects).

    Returns [(name, mesh, [material_name,...])] — a single-entry list,
    the same shape load_model_meshes' other builders return. Local
    space (identity rotation/translation), only *scale* applied.

    Vertex color is written to a "Col" face-corner color attribute
    (matching build_ase_meshes/build_lwo_meshes), and every loop gets a
    custom split normal: the file's own baked per-vertex normal where
    the source surface has one (already split at every UV/hard-edge
    seam by the compiler — these are the exact normals the real engine
    renders with), or Blender's own computed smooth vertex normal as a
    fallback for a surface whose generateNormals flag is set (see
    parse_binary_render_model) — there's no "use file normals" toggle
    here the way the text .ase/.lwo importers have, since unlike a
    from-scratch text-format parse there's no alternative
    reconstruction for the caller to choose between.
    """
    surfaces = parse_binary_render_model(filepath)
    fallback_base = (name_hint or '').strip().strip('"') or os.path.splitext(os.path.basename(filepath))[0]

    bm = bmesh.new()
    uv_lay  = bm.loops.layers.uv.new("UVMap")
    col_lay = bm.loops.layers.float_color.new("Col")
    nrm_lay = bm.loops.layers.float_vector.new("temp_normal")

    mat_idx_map     = {}
    _bin_identities = {}   # material name -> identity dict, see material_identity_from_lwo
    fallback_loops  = []   # loops needing a generateNormals surface's fallback — filled in after bm.normal_update()
    any_face = False
    for surf in surfaces:
        positions = surf['positions']
        if not positions or not surf['tris']:
            continue
        uvs     = surf['uvs']
        normals = surf['normals']
        colors  = surf['colors']

        _bin_ident = material_identity_from_lwo(surf['shader'] or '')
        mat_name = _bin_ident['display'] if (surf['shader'] or '') else fallback_base
        _bin_identities[mat_name] = _bin_ident
        if mat_name not in mat_idx_map:
            mat_idx_map[mat_name] = len(mat_idx_map)
        mat_index = mat_idx_map[mat_name]

        bverts = [bm.verts.new(Vector((p[0] * scale, p[1] * scale, p[2] * scale))) for p in positions]
        bm.verts.ensure_lookup_table()

        for a, b, c in surf['tris']:
            if a >= len(bverts) or b >= len(bverts) or c >= len(bverts):
                continue
            try:
                f = bm.faces.new((bverts[a], bverts[b], bverts[c]))
            except ValueError:
                continue
            any_face = True
            f.material_index = mat_index
            for loop, vidx in zip(f.loops, (a, b, c)):
                loop[uv_lay].uv = uvs[vidx] if vidx < len(uvs) else (0.0, 0.0)
                loop[col_lay]   = colors[vidx] if vidx < len(colors) else (1.0, 1.0, 1.0, 1.0)
                if normals is not None and vidx < len(normals):
                    loop[nrm_lay] = normals[vidx]
                else:
                    fallback_loops.append(loop)

    if not any_face:
        bm.free()
        return []

    bm.normal_update()
    for loop in fallback_loops:
        loop[nrm_lay] = tuple(loop.vert.normal)

    mesh = bpy.data.meshes.new(fallback_base)
    bm.to_mesh(mesh)
    bm.free()
    mesh.validate(clean_customdata=False)
    mesh.update()
    mark_primary_color_attribute(mesh)

    nrm_attr = mesh.attributes.get("temp_normal")
    if nrm_attr is not None:
        mark_smooth_for_custom_normals(mesh)
        loop_normals = [tuple(nrm_attr.data[i].vector) for i in range(len(mesh.loops))]
        mesh.normals_split_custom_set(loop_normals)
        # Re-fetch by name rather than reusing nrm_attr — see the
        # matching comment in build_ase_meshes for why (Blender 4.5
        # invalidates the handle here, 5.0+ doesn't).
        nrm_attr = mesh.attributes.get("temp_normal")
        if nrm_attr is not None:
            mesh.attributes.remove(nrm_attr)
            mesh.update()

    mat_names = list(mat_idx_map.keys())
    stash_material_identities(mesh, [_bin_identities.get(n) for n in mat_names])
    return [(fallback_base, mesh, mat_names)]


class UnsupportedModelFormat(ValueError):
    """load_model_meshes was handed a path whose extension it has no
    loader for. Distinct from "the file loaded but held no geometry",
    which is an empty return - the two used to be the same empty list,
    so handing the importer a directory (what the file browser leaves in
    filepath when no file is selected) reported "No usable mesh data
    found in file" and still returned {'FINISHED'}."""


def load_model_meshes(filepath, scale, lwo_first_layer_only=False, name_hint=None,
                       engine_render_mesh=False,
                       shading=SHADING_ENGINE, shading_overrides=None):
    """Dispatch to the correct static-mesh loader based on file extension.
    Returns [(name, mesh, [material_name,...]), ...]; an empty list means
    the file was read but held no usable geometry. An extension with no
    loader raises UnsupportedModelFormat rather than returning empty -
    callers need to tell "not a model file" from "an empty model file".

    This is the one function other idTech4 addons (currently only
    idTech4_map_io.py) call into — see the module docstring above
    for the detection/availability contract those callers follow.

    *lwo_first_layer_only*: passed straight to build_lwo_meshes (.lwo
    only — .ase has no equivalent "layer" concept, and the real engine
    uses every .ase GEOMOBJECT unconditionally, so this never applies to
    .ase files). See build_lwo_meshes for why this matches real idTech4/
    Doom 3 engine behavior.
    *name_hint*: passed straight to build_ase_meshes / build_lwo_meshes /
    build_binary_meshes as their fallback-naming base — see any of them
    for details.
    *shading*: passed straight to build_ase_meshes / build_lwo_meshes —
    see either for details. Doesn't apply to a compiled .base/.blwo —
    see build_binary_meshes for why.
    *engine_render_mesh*: passed straight to build_lwo_meshes (.lwo only) —
    see it for what the engine's render mesh is and why you would want it.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.ase':
        return build_ase_meshes(filepath, scale, name_hint=name_hint,
                                shading=shading,
                                shading_overrides=shading_overrides)
    if ext == '.lwo':
        return build_lwo_meshes(filepath, scale, first_layer_only=lwo_first_layer_only, name_hint=name_hint,
                                 engine_render_mesh=engine_render_mesh,
                                 shading=shading,
                                 shading_overrides=shading_overrides)
    if ext in ('.base', '.blwo'):
        return build_binary_meshes(filepath, scale, name_hint=name_hint)
    raise UnsupportedModelFormat(
        "%s is not a static mesh format this importer reads (expected "
        ".ase, .base, .lwo or .blwo)" % (ext or "that file"))


def report_model_import_result(op, result):
    """Turn import_model_file's status string into operator reports plus
    the operator's own return value.

    The protocol is a prefix on the first line: "ERROR: " (nothing was
    imported - refuse), or "OK: " optionally followed by a "\nNOTICE: "
    second part for something that imported fine but is worth saying out
    loud. There is deliberately no warning-but-finished prefix: every way
    this import can produce no objects is an ERROR, because a refusal
    that returns {'FINISHED'} pushes an undo step over an unchanged
    scene and hides the failure from scripted callers.

    Prefixes are split off rather than sliced by a hardcoded length -
    the old slices were each one character short, so every message
    reached the status bar with a leading space ("Info:  5 mesh
    object(s)").
    """
    kind, sep, message = result.partition(": ")
    if not sep:
        # No prefix at all - not something this protocol produces, but
        # reporting the string itself beats reporting an empty message.
        kind, message = "", result
    if kind == "ERROR":
        op.report({'ERROR'}, message)
        return {'CANCELLED'}
    ok_part, notice_sep, notice_part = message.partition("\nNOTICE: ")
    op.report({'INFO'}, ok_part)
    if notice_sep:
        op.report({'WARNING'}, notice_part)
    return {'FINISHED'}


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


# ─────────────────────────────────────────────────────────────────────
#  STANDALONE .ase / .lwo IMPORT
# ─────────────────────────────────────────────────────────────────────

def model_material_names(filepath, lwo_first_layer_only=False):
    """The decl names a model uses, without building any geometry.

    Cheap enough to run before the import proper, which is what the .mtr
    lookup needs -- renderBump and unsmoothedTangents decide how the normals
    are computed, so the answer has to be in hand first.
    """
    ext = os.path.splitext(filepath)[1].lower()
    out = set()
    try:
        if ext == '.lwo':
            _layers, tags, _surfaces = parse_lwo_file(filepath)
            for t in tags:
                ident = material_identity_from_lwo(t)
                if ident.get('decl'):
                    out.add(ident['decl'])
        elif ext == '.ase':
            _objects, materials = parse_ase_file(filepath)
            for entry in materials.values():
                ident = _ase_material_identity(entry)
                if ident.get('decl'):
                    out.add(ident['decl'])
                for sub in (entry.get('submaterials') or {}).values():
                    ident = _ase_material_identity(sub)
                    if ident.get('decl'):
                        out.add(ident['decl'])
    except Exception:
        return set()
    return out


def import_model_file(context, filepath, scale, rotation_key='NONE',
                      lwo_first_layer_only=False,
                      engine_render_mesh=False, shading=SHADING_ENGINE,
                      import_materials=True, material_mode='GOOD',
                      material_parameters=None,
                      derive_from_model=False, override_base_directory='',
                      override_source_path='', save_derived_as_default=False):
    """Shared implementation for the standalone .ase / .lwo import
    operators. Creates one collection named after the file, containing one
    object per mesh piece found in it (an .ase can contain several
    GEOMOBJECTs; an .lwo can contain several layers).

    *lwo_first_layer_only*: see load_model_meshes / build_lwo_meshes.
    *shading*: see load_model_meshes / build_lwo_meshes /
    build_ase_meshes.
    *engine_render_mesh*: see build_lwo_meshes (.lwo only).
    *import_materials*/*material_mode*/*material_parameters*/
    *derive_from_model*/*override_base_directory*/*override_source_path*/
    *save_derived_as_default*: see _resolve_material_sources and
    generate_materials_for_objects — needs the companion "idTech4
    Materials" addon; silently does nothing (leaving the blank
    placeholder materials get_or_create_material
    makes) if that addon isn't installed/enabled.

    Both the Object and its Mesh datablock are named after the model
    piece itself (the .lwo layer name / .ase *NODE_NAME), not a
    generated compound name — Blender auto-uniquifies with a ".001"
    suffix on any collision, same as any other duplicated object name.

    Any resulting object with a quad/ngon face gets flagged in the
    returned status message: the real engine's static-mesh loader only
    ever accepts already-triangulated polygons (Model.cpp/Model_lwo.cpp —
    a non-triangle .lwo polygon is dropped with a warning, "make sure you
    triplet it down"; .ase's own *MESH_FACE format is triangle-only by
    construction, so this only ever applies to .lwo). This importer keeps
    quad/ngon geometry as-is rather than silently dropping or
    auto-triangulating it — useful for reviewing/editing the source art —
    but that means what's shown here can be MORE geometry than what
    actually renders in-game."""
    filename = os.path.splitext(os.path.basename(filepath))[0]

    # The .mtr lookup has to happen BEFORE the meshes are built, because
    # renderBump and unsmoothedTangents change the normals themselves, not
    # anything applied afterwards. Only Engine Shading asks: the other modes
    # are not claiming to be the engine.
    shading_overrides = (set(), set())
    mode = resolve_shading_mode(shading)
    if mode == SHADING_ENGINE and import_materials:
        try:
            _base, _mod, _src = _resolve_material_sources(
                filepath, derive_from_model, override_base_directory,
                override_source_path, save_derived_as_default)
            if _src:
                names = model_material_names(filepath, lwo_first_layer_only)
                shading_overrides = material_shading_overrides(names, _src)
        except Exception:
            shading_overrides = (set(), set())

    try:
        sub_meshes = load_model_meshes(filepath, scale, lwo_first_layer_only=lwo_first_layer_only,
                                        engine_render_mesh=engine_render_mesh,
                                        shading=shading,
                                        shading_overrides=shading_overrides)
    except UnsupportedModelFormat as exc:
        # Not a parse failure - there was never a loader to try. A
        # traceback would only bury the one useful sentence.
        return "ERROR: " + str(exc)
    except Exception:
        return "ERROR: " + traceback.format_exc()

    if not sub_meshes:
        # The file read cleanly and held nothing usable. Still a refusal:
        # no collection is created and no object reaches the scene, so
        # reporting anything but an error would push an undo step over an
        # unchanged scene and tell a scripted caller nothing went wrong.
        return ("ERROR: No usable mesh data in \"%s\" - the file was read, "
                "but it contains no geometry this importer can build."
                % os.path.basename(filepath))

    scene = context.scene
    col = bpy.data.collections.new(filename)
    scene.collection.children.link(col)

    mat_cache  = {}
    rot_matrix = ROTATION_PRESETS.get(rotation_key) or Matrix.Identity(4)

    all_objects   = []
    nontri_objects = []
    for m_idx, (mname, mesh, mat_names) in enumerate(sub_meshes):
        idents = take_material_identities(mesh)
        for slot, mn in enumerate(mat_names):
            mat = get_or_create_material(mn, mat_cache)
            if slot < len(idents):
                apply_material_identity(mat, idents[slot])
            mesh.materials.append(mat)
        apply_mesh_smoothing(mesh, mode == SHADING_SMOOTH)

        obj_name = mname or f"{filename}_{m_idx:02d}"
        obj = bpy.data.objects.new(obj_name, mesh)
        col.objects.link(obj)
        apply_point_maps_to_object(obj)
        obj.matrix_basis = rot_matrix
        all_objects.append(obj)

        if any(len(p.vertices) > 3 for p in mesh.polygons):
            nontri_objects.append(obj_name)

    for o in all_objects:
        try:
            o.select_set(True)
        except RuntimeError:
            pass
    if all_objects:
        context.view_layer.objects.active = all_objects[0]

    material_issues = []
    material_report = None
    if import_materials and all_objects:
        base_directory, mod_directory, source_path = _resolve_material_sources(
            filepath, derive_from_model, override_base_directory,
            override_source_path,
            save_derived_as_default=save_derived_as_default)
        material_issues, built_count, material_report = \
            generate_materials_for_objects(
                all_objects, base_directory, source_path, material_mode,
                context, mod_directory=mod_directory,
                material_parameters=material_parameters)
        if built_count > 0:
            _set_viewport_shading_material_preview(context)

    result = f"OK: {len(all_objects)} mesh object(s), {len(mat_cache)} materials"
    if material_report:
        # Same sentence the Materials panel's own Generate Materials button
        # ends on, so the two read alike. It goes on the OK line rather than
        # a NOTICE: a report that exists is not a warning.
        result += f' - full report in the Text Editor as "{material_report}"'
    if nontri_objects:
        result += ("\nNOTICE: " + f"{len(nontri_objects)} object(s) contain quad/ngon faces — "
                   "idTech4 only renders already-triangulated polygons, so this geometry may "
                   "show more here than actually renders in-game: " + ", ".join(nontri_objects))
    for issue in material_issues:
        result += "\nNOTICE: " + issue
    return result


class IMPORT_OT_idtech4_ase(bpy.types.Operator, ImportHelper, ImportFileGuardMixin,
                            MD5_ImportScaleRotMixin, MaterialGenMixin):
    """Import an idTech4 .ase static mesh — either the standard ASCII
    .ase, or the Doom 3 BFG compiled binary cache of one (.base)"""
    bl_idname  = "import_scene.idtech4_ase"
    bl_label   = "Import idTech4 ASE (.ase, .base)"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".ase"
    filter_glob: StringProperty(default="*.ase;*.base", options={'HIDDEN'}, maxlen=255)

    shading: EnumProperty(
        name="Shading",
        description="How to shade the imported mesh",
        items=SHADING_ITEMS,
        default=SHADING_FILE,
    )

    def _is_binary(self):
        return os.path.splitext(self.filepath)[1].lower() == '.base'

    def draw(self, context):
        self.draw_transforms(self.layout)
        box = self.layout.box()
        box.label(text="Shading:")
        box.prop(self, "shading", text="")
        if self._is_binary():
            box.label(text="compiled .base carries its own normals",
                      icon='INFO')
        self.draw_material_gen(context, self.layout)
        self._draw_vertex_color_note(self.layout)

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
        result = import_model_file(context, self.filepath,
                                   scale=self.get_scale(),
                                   rotation_key=self.get_rotation(),
                                   shading=self.shading,
                                   import_materials=self.import_materials,
                                   material_mode=self.material_mode,
                                   material_parameters=self.material_parameters,
                                   derive_from_model=self.derive_from_model,
                                   override_base_directory=self.override_base_directory,
                                   override_source_path=self.override_source_path,
                                   save_derived_as_default=self.save_derived_as_default)
        return report_model_import_result(self, result)


class IMPORT_OT_idtech4_lwo(bpy.types.Operator, ImportHelper, ImportFileGuardMixin,
                            MD5_ImportScaleRotMixin, MaterialGenMixin):
    """Import an idTech4 .lwo (Lightwave, LWO2) static mesh — either the
    standard .lwo, or the Doom 3 BFG compiled binary cache of one (.blwo)"""
    bl_idname  = "import_scene.idtech4_lwo"
    bl_label   = "Import idTech4 LWO (.lwo, .blwo)"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".lwo"
    filter_glob: StringProperty(default="*.lwo;*.blwo", options={'HIDDEN'}, maxlen=255)

    shading: EnumProperty(
        name="Shading",
        description="How to shade the imported mesh",
        items=SHADING_ITEMS,
        default=SHADING_FILE,
    )
    first_layer_only: BoolProperty(
        name="Only import 1st layer",
        description="Import only the first layer in the .lwo file. This "
                    "matches real idTech4/Doom 3 engine behavior: verified "
                    "against the GPL source "
                    "(idRenderModelStatic::ConvertLWOToModelSurfaces), the "
                    "engine only ever reads the FIRST layer of a .lwo — "
                    "any additional layer is parsed but never rendered. "
                    "Disable to import every layer instead, each as its "
                    "own object",
        default=True,
    )
    engine_render_mesh: BoolProperty(
        name="Build Engine Render Mesh",
        description="Build the mesh idTech4 actually RENDERS instead of the "
                    "mesh the file describes — useful for seeing exactly "
                    "where the engine will split your model, useless as a "
                    "source of truth about the file itself. Replicates the "
                    "GPL source's own non-fastLoad load path "
                    "(ConvertLWOToModelSurfaces + the topology-changing half "
                    "of R_CleanupTriangles): coincident points welded at "
                    "r_slopVertex 0.01, texcoords welded at r_slopTexCoord "
                    "0.001, then one vertex per unique combination of "
                    "position, texcoord, vertex color and normal — so every "
                    "UV seam and every hard edge becomes a real vertex split "
                    "— degenerate triangles dropped, and any vertex used by "
                    "both texture polarities duplicated. Vertex count will be "
                    "HIGHER than the file's and the mesh will no longer "
                    "correspond to it point-for-point. Two caveats: Blender "
                    "cannot hold two faces on one vertex set, so a triangle "
                    "the weld collapses onto another is dropped here where "
                    "the engine would keep it (the count is printed to the "
                    "console); and this is only what the engine really does "
                    "for a normally-loaded model — a renderbump HIGH poly is "
                    "loaded with fastLoad instead, which welds nothing at "
                    "all. Off by default",
        default=False,
    )
    def _is_binary(self):
        return os.path.splitext(self.filepath)[1].lower() == '.blwo'

    def draw(self, context):
        self.draw_transforms(self.layout)
        box = self.layout.box()
        box.label(text="Shading:")
        box.prop(self, "shading", text="")
        if not self._is_binary():
            box = self.layout.box()
            box.label(text="Geometry:")
            box.prop(self, "first_layer_only")
            box.prop(self, "engine_render_mesh")
        self.draw_material_gen(context, self.layout)
        self._draw_vertex_color_note(self.layout)

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
        result = import_model_file(context, self.filepath,
                                   scale=self.get_scale(),
                                   rotation_key=self.get_rotation(),
                                   lwo_first_layer_only=self.first_layer_only,
                                   shading=self.shading,
                                   engine_render_mesh=self.engine_render_mesh,
                                   import_materials=self.import_materials,
                                   material_mode=self.material_mode,
                                   material_parameters=self.material_parameters,
                                   derive_from_model=self.derive_from_model,
                                   override_base_directory=self.override_base_directory,
                                   override_source_path=self.override_source_path,
                                   save_derived_as_default=self.save_derived_as_default)
        return report_model_import_result(self, result)


# ─────────────────────────────────────────────────────────────────────
#  STANDALONE .ase / .lwo EXPORT
#
#  Every idiosyncrasy below is verified directly against the real Doom 3
#  GPL source (renderer/Model_ase.cpp, Model_lwo.cpp, Model.cpp).
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
#  EXPORT SCOPE, VALIDATION AND PREPARATION
#
#  Three separate things, deliberately:
#
#    Validate            reads only. Reports every finding, each tagged with
#                        who is expected to deal with it.
#    Prepare for Export  changes the model, undoably, and only for the
#                        findings that genuinely need the geometry changed.
#    Export              never touches the scene. It applies the changes that
#                        exist only in the written bytes, and says so.
#
#  A finding's TIER is which of those owns it:
#    AUTO     the writer handles it; nothing to do but know about it
#    PREPARE  the mesh itself has to change, so the artist should see it
#    MANUAL   needs a decision no tool should make
# ─────────────────────────────────────────────────────────────────────

TIER_AUTO = 'AUTO'
TIER_PREPARE = 'PREPARE'
TIER_MANUAL = 'MANUAL'

IDTECH4_EXCLUDE_PROP = 'idtech4_exclude'

EXPORT_SCOPE_ITEMS = [
    ('COLLECTION', "Collection",
     "Every mesh in the active object's collection. This is what an import "
     "produces — one collection per file — so it is the natural unit of "
     "\"a model\""),
    ('SELECTED', "Selected Objects", "Only the selected mesh objects"),
    ('SCENE', "Whole Scene", "Every mesh object in the scene"),
]

VALIDATE_TARGET_ITEMS = [
    ('BOTH', ".lwo and .ase", "Report what applies to either format"),
    ('LWO', ".lwo only", "Only the constraints a .lwo export has to meet"),
    ('ASE', ".ase only", "Only the constraints an .ase export has to meet"),
]


def object_is_hidden(obj, context):
    """Hidden for export purposes = the OUTLINER EYE is closed.

    Deliberately not obj.hide_viewport (the monitor icon) and not
    hide_render: those are different controls with different meanings, and
    file contents should follow one obvious one. hide_get() is per view layer,
    so the manifest is what makes the choice visible rather than mysterious.
    """
    try:
        return obj.hide_get()
    except RuntimeError:
        # not in this view layer at all (its collection is excluded)
        return True


def iter_export_objects(context, scope='COLLECTION', collection=None,
                        skip_hidden=True):
    """The objects an export or a validation run will look at."""
    if scope == 'SELECTED':
        pool = list(context.selected_objects)
    elif scope == 'SCENE':
        pool = list(context.scene.objects)
    else:
        coll = collection
        if coll is None:
            act = context.active_object
            if act is not None and act.users_collection:
                coll = act.users_collection[0]
        if coll is None:
            coll = context.scene.collection
        pool = [o for o in coll.all_objects]
    out = []
    for o in pool:
        if o.type != 'MESH':
            continue
        if o.get(IDTECH4_EXCLUDE_PROP):
            continue
        if skip_hidden and object_is_hidden(o, context):
            continue
        out.append(o)
    return out


def to_object_mode(context):
    """Leave Edit/Pose/Sculpt Mode so an export reads finished data.

    Blender does not write an Edit Mode session's BMesh back to the Mesh
    datablock until the mode is exited - the bmesh.update_edit_mesh() that
    runs as you model only refreshes the viewport cage - and everything
    downstream of here reads the datablock (obj.data, obj.to_mesh(), the
    vertex groups). Exporting without tabbing out therefore wrote the
    model as it was when Edit Mode was ENTERED: no error, no warning, just
    the old geometry in the file.

    This is the recipe Blender's own exporters use. io_scene_fbx's save()
    records the active object's mode, switches to Object Mode and switches
    back; io_scene_gltf2's save() switches without restoring. The mode
    round trip is what flushes the edit BMesh, and it flushes every object
    of a multi-object edit session, not just the active one.

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


def export_manifest(context, scope='COLLECTION', collection=None,
                    skip_hidden=True, target='LWO'):
    """A cheap summary of what an export would write, safe to call from draw().

    Counts only — no mesh evaluation, no modifier application — because this
    runs on every redraw of the export dialog.
    """
    included = iter_export_objects(context, scope, collection, skip_hidden)
    considered = iter_export_objects(context, scope, collection, False)
    hidden = [o for o in considered if o not in included]
    excluded = []
    if scope == 'COLLECTION':
        coll = collection
        if coll is None:
            act = context.active_object
            if act is not None and act.users_collection:
                coll = act.users_collection[0]
        if coll is not None:
            excluded = [o for o in coll.all_objects
                        if o.type == 'MESH' and o.get(IDTECH4_EXCLUDE_PROP)]
    tris = 0
    ngons = 0
    for o in included:
        for p in o.data.polygons:
            n = len(p.vertices)
            tris += 1 if n == 3 else (n - 2)
            if n != 3:
                ngons += 1
    return {
        'objects': included, 'hidden': hidden, 'excluded': excluded,
        'triangles': tris, 'ngons': ngons,
        'merged': (target == 'LWO' and len(included) > 1),
    }


def manifest_snapshot(man, scope, target, skip_hidden):
    """Freeze an export_manifest() result into something a Panel can redraw.

    Names and counts only. A bpy Object cannot be parked in a scene property
    and read back later - the objects it points at can be deleted between the
    run and the redraw - so the snapshot keeps the strings the box prints and
    nothing else. The scope/target/skip_hidden it was taken with ride along so
    the panel can say when the controls above the box have moved on.
    """
    return json.dumps({
        'scope': scope, 'target': target, 'skip_hidden': bool(skip_hidden),
        'objects': [o.name for o in man['objects']],
        'hidden': [o.name for o in man['hidden']],
        'triangles': man['triangles'], 'ngons': man['ngons'],
        'merged': man['merged'],
    })


def load_manifest_snapshot(raw):
    """The dict manifest_snapshot() wrote, or None if there is not one yet."""
    if not raw:
        return None
    try:
        snap = json.loads(raw)
    except ValueError:
        return None
    return snap if isinstance(snap, dict) else None


def _finding(tier, message, target='BOTH', objects=()):
    return {'tier': tier, 'message': message, 'target': target,
            'objects': sorted({o.name for o in objects})}


def validate_for_idtech4(objects, target='BOTH', depsgraph=None,
                         apply_modifiers=True):
    """Check *objects* against what the engine will actually accept.

    Every rule here is the engine's, cited where it is not obvious. Read-only:
    meshes are evaluated through to_mesh()/to_mesh_clear() exactly the way the
    exporter does, so nothing in the scene changes.
    """
    findings = []
    want_lwo = target in ('BOTH', 'LWO')
    want_ase = target in ('BOTH', 'ASE')

    no_uv, nontri, degenerate, degen_uv = [], [], [], []
    foreign_mat, dotted_decl, big_poly = [], [], []
    unresolvable_mat = []
    subd_faces, custom_normals, multi_uv = [], [], []
    node_tm, bad_ptype, bad_smgp = [], [], []
    total_points = 0

    for obj in objects:
        owner, temp = obj, False
        if apply_modifiers and depsgraph is not None:
            owner = obj.evaluated_get(depsgraph)
            temp = True
        mesh = owner.to_mesh() if temp else obj.data
        try:
            if mesh is None:
                continue
            total_points += len(mesh.vertices)
            if not mesh.uv_layers:
                no_uv.append(obj)
            elif len(mesh.uv_layers) > 1:
                multi_uv.append(obj)
            if mesh.has_custom_normals:
                custom_normals.append(obj)

            for p in mesh.polygons:
                n = len(p.vertices)
                if n != 3:
                    if obj not in nontri:
                        nontri.append(obj)
                    if n > 1023 and obj not in big_poly:
                        big_poly.append(obj)
                if p.area <= 1e-12 and obj not in degenerate:
                    degenerate.append(obj)

            uvl = mesh.uv_layers.active
            if uvl is not None:
                for p in mesh.polygons:
                    if len(p.vertices) != 3:
                        continue
                    a, b, c = (tuple(uvl.data[i].uv) for i in
                               range(p.loop_start, p.loop_start + 3))
                    if a == b or b == c or c == a:
                        degen_uv.append(obj)
                        break

            pt = mesh.attributes.get(LWO_POLY_TYPE_ATTR)
            if pt is not None and pt.domain == 'FACE':
                vals = [0] * len(mesh.polygons)
                pt.data.foreach_get('value', vals)
                if any(v < 0 or v >= len(LWO_POLY_TYPES) for v in vals):
                    bad_ptype.append(obj)
                elif any(v != 0 for v in vals):
                    subd_faces.append(obj)
            sg = mesh.attributes.get(LWO_SMGP_ATTR)
            if sg is not None and sg.domain == 'FACE':
                vals = [0] * len(mesh.polygons)
                sg.data.foreach_get('value', vals)
                if any(v < 0 or v > 0xFFFF for v in vals):
                    bad_smgp.append(obj)

            if obj.data.get(ASE_NODE_TM_PROP):
                tm = list(obj.data[ASE_NODE_TM_PROP])
                if len(tm) == 12 and tm[:9] != [1, 0, 0, 0, 1, 0, 0, 0, 1]:
                    node_tm.append(obj)
        finally:
            if temp:
                owner.to_mesh_clear()

        for slot, mat in enumerate(obj.data.materials):
            if mat is None:
                continue
            if mat.get('idtech4_material_kind') == MAT_FOREIGN:
                foreign_mat.append(obj)
            decl = material_decl_for_export(mat, '')
            # MakeNameCanonical truncates at the LAST dot anywhere in the
            # string, so a dot in a FOLDER name silently eats the rest of the
            # path. Split on either separator: a name that reaches here can
            # still be backslashed.
            head = decl.replace('\\', '/').rpartition('/')[0]
            if '.' in head:
                dotted_decl.append(obj)
            # And the check that actually decides visibility: does the string
            # each exporter is going to write resolve BACK to this decl? The
            # exporters verify their own choice, so this only fires when no
            # candidate works at all -- but when it does, the model loads with
            # a material the engine cannot find, and that is invisible in game
            # rather than obviously broken.
            for enabled, resolver, wrap in (
                    (want_ase, engine_ase_bitmap_decl,
                     lambda d: "\\base\\" + d.replace('/', '\\')),
                    (want_lwo, engine_lwo_surf_decl, None)):
                if not enabled:
                    continue
                written = material_source_string(mat, decl, resolver, wrap)
                if resolver(written) != decl:
                    unresolvable_mat.append(obj)
                    break

    if nontri:
        findings.append(_finding(
            TIER_PREPARE,
            "%d object(s) have polygons that are not triangles. The engine "
            "renders only already-triangulated polygons and drops the rest "
            "outright (\"make sure you triplet it down\"). Export triangulates "
            "them for you and says so; Prepare for Export does it in the "
            "scene, where you control the method." % len(nontri), 'BOTH', nontri))
    if big_poly:
        findings.append(_finding(
            TIER_MANUAL,
            "%d object(s) have a polygon with more than 1023 vertices, which "
            "LWO's 10-bit POLS vertex-count field cannot express. Those "
            "polygons are skipped." % len(big_poly), 'LWO', big_poly))
    if degenerate:
        findings.append(_finding(
            TIER_PREPARE,
            "%d object(s) contain zero-area faces. R_RemoveDegenerateTriangles "
            "discards them at load." % len(degenerate), 'BOTH', degenerate))
    if no_uv:
        findings.append(_finding(
            TIER_MANUAL,
            "%d object(s) have no UV map; every face gets (0,0) texture "
            "coordinates." % len(no_uv), 'BOTH', no_uv))
    if degen_uv:
        findings.append(_finding(
            TIER_MANUAL,
            "%d object(s) have triangles whose UV corners coincide, giving a "
            "degenerate texture space (R_TestDegenerateTextureSpace). Tangents "
            "there are meaningless." % len(degen_uv), 'BOTH', degen_uv))
    if foreign_mat:
        findings.append(_finding(
            TIER_MANUAL,
            "%d object(s) use a material identified as NOT idTech4-authored, "
            "so it names no .mtr declaration the engine can resolve."
            % len(foreign_mat), 'BOTH', foreign_mat))
    if dotted_decl:
        findings.append(_finding(
            TIER_MANUAL,
            "%d object(s) have a material path with a dot in a FOLDER name. "
            "MakeNameCanonical truncates a decl name at the last dot anywhere "
            "in the string, so the engine will look up a shorter name than "
            "you wrote." % len(dotted_decl), 'BOTH', dotted_decl))
    if unresolvable_mat:
        findings.append(_finding(
            TIER_MANUAL,
            "%d object(s) have a material whose name the engine cannot resolve "
            "back to a declaration. Nothing the exporter can write will reach "
            "the right .mtr, so the surface loads with an implicitly generated "
            "material: a single alpha-blended stage over a default image that "
            "is fully transparent outside developer mode. The model will be "
            "INVISIBLE in game while still showing under r_showtris. Rename "
            "the material to the .mtr declaration it should use."
            % len(unresolvable_mat), 'BOTH', unresolvable_mat))
    if multi_uv and want_lwo:
        findings.append(_finding(
            TIER_AUTO,
            "%d object(s) have more than one UV map. idDrawVert holds a single "
            "idVec2 st, so only one survives; export writes the active map "
            "unless All Vertex Maps is on." % len(multi_uv), 'LWO', multi_uv))
    if multi_uv and want_ase:
        findings.append(_finding(
            TIER_AUTO,
            "%d object(s) have more than one UV map. The .ase reader handles "
            "only *MESH_TVERTLIST — there is no *MAPPINGCHANNEL support at all "
            "— so the others are dropped." % len(multi_uv), 'ASE', multi_uv))
    if custom_normals and want_lwo:
        findings.append(_finding(
            TIER_MANUAL,
            "%d object(s) carry custom split normals. A .lwo cannot store "
            "them: SURF SMAN plus PTAG SMGP reproduce a per-face flat/smooth "
            "toggle and nothing finer. Export .ase if the exact per-corner "
            "normals matter." % len(custom_normals), 'LWO', custom_normals))
    if subd_faces and want_ase:
        findings.append(_finding(
            TIER_MANUAL,
            "%d object(s) have faces tagged as a subdivision POLS type, which "
            ".ase has no equivalent for." % len(subd_faces), 'ASE', subd_faces))
    if subd_faces and want_lwo:
        findings.append(_finding(
            TIER_MANUAL,
            "%d object(s) have faces tagged PTCH/SUBD. The engine ignores the "
            "polygon type and only rejects non-triangles, so a subpatch cage "
            "of quads simply vanishes in-game." % len(subd_faces), 'LWO', subd_faces))
    if bad_ptype:
        findings.append(_finding(
            TIER_MANUAL,
            "%d object(s) have an out-of-range value in the editable "
            "\"%s\" face attribute." % (len(bad_ptype), LWO_POLY_TYPE_ATTR),
            'LWO', bad_ptype))
    if bad_smgp:
        findings.append(_finding(
            TIER_MANUAL,
            "%d object(s) have a smoothing-group id outside 0..65535 in the "
            "editable \"%s\" face attribute; PTAG stores it in a U2."
            % (len(bad_smgp), LWO_SMGP_ATTR), 'LWO', bad_smgp))
    if node_tm and want_ase:
        findings.append(_finding(
            TIER_MANUAL,
            "%d object(s) carry a non-identity *NODE_TM. The engine applies "
            "that matrix to NORMALS but never to positions, so shading and "
            "geometry will disagree in-game." % len(node_tm), 'ASE', node_tm))
    if want_lwo and total_points >= 0xFF00:
        findings.append(_finding(
            TIER_AUTO,
            "%d points: past 65280 the writer switches to 4-byte VX indices, "
            "which the engine reads correctly." % total_points, 'LWO'))
    if want_lwo and len(objects) > 1:
        findings.append(_finding(
            TIER_AUTO,
            "%d objects will be merged into ONE layer. The engine reads only "
            "a .lwo's first layer (`lwLayer *layer = lwo->layer;`), so a "
            "second layer would be dead data." % len(objects), 'LWO'))

    return [f for f in findings
            if f['target'] == 'BOTH' or f['target'] == target
            or (target == 'BOTH')]



class ExportTransformMixin:
    """Shared Scale/Rotation export properties — the inverse counterpart of
    MD5_ImportScaleRotMixin's import-side transforms. Every exported
    position is built as:

        file_pos = get_scale() * ( get_rotation_matrix() @ (obj.matrix_world @ local_co) )

    i.e. each object's full current Blender transform is baked in first
    (so moving/rotating/scaling an object in the viewport is reflected in
    the exported file, like any other Blender exporter), THEN the chosen
    Rotation preset undoes the matching import-side preset, THEN the scale
    factor converts Blender units back to idTech4 file units. Normals go
    through the same rotation (via an inverse-transpose of the 3x3 part,
    so non-uniform object scale doesn't skew them) but never the scalar
    scale factor, since it doesn't change direction."""

    scale_mode: EnumProperty(
        name="",
        description="How the export scale factor below is determined",
        items=[
            ('NONE', "None", "No scaling — export raw Blender units unchanged"),
            ('PRESET', "Blender to idTech4 (m → in)",
             "Convert Blender meters back to idTech4 inches. Multiplies "
             "every exported position by %.6g (the inverse of 1 idTech4 "
             "unit = 1 inch = %.4f m)" % (1.0 / MD5_SCALE_IN_TO_M, MD5_SCALE_IN_TO_M)),
            ('FACTOR', "Custom Factor",
             "Multiply every exported position by the arbitrary numeric "
             "Scale Factor specified below"),
        ],
        default='NONE',
    )
    scale_factor: FloatProperty(
        name="Scale Factor",
        description="Custom uniform scale factor multiplied into every "
                    "exported position",
        default=1.0, min=0.0001, max=10000.0,
    )
    rotation: EnumProperty(
        name="",
        description="Undo a Z-axis rotation preset applied at import time, "
                    "before writing positions/normals to the file",
        items=IMPORT_ROTATION_ITEMS,
        default='NONE',
    )
    export_scope: EnumProperty(
        name="Scope",
        description="Which objects to write. Collection matches what an "
                    "import produces — one collection per file — so it is the "
                    "natural unit of \"a model\", and it ends the \"did I "
                    "select every piece?\" failure",
        items=EXPORT_SCOPE_ITEMS,
        default='COLLECTION',
    )
    skip_hidden: BoolProperty(
        name="Skip Hidden Objects",
        description="Leave out anything whose outliner EYE is closed. The "
                    "monitor and camera icons are deliberately NOT consulted: "
                    "one obvious control decides what reaches the file, and "
                    "the summary below lists exactly what is being skipped, "
                    "before you press Export",
        default=True,
    )

    def _scope_objects(self, context):
        return iter_export_objects(context, self.export_scope, None,
                                   self.skip_hidden)

    def draw_scope(self, layout, context, target):
        """Scope controls plus a live manifest of what will be written.

        Cheap on purpose: this runs on every redraw, so it counts polygons and
        nothing else. The heavy checks live behind the Check Model button.
        """
        col = layout.column(align=True)
        col.prop(self, "export_scope")
        col.prop(self, "skip_hidden")
        # Whether the n-gon line below is a warning or just a note. .ase
        # is triangle-only by construction (its writer always calls
        # _gather_export_triangles with triangulate=True), so there is
        # nothing to ask; .lwo carries polygons natively and has a real
        # Triangulate checkbox further down the dialog.
        will_triangulate = True if target == 'ASE' else self.triangulate

        man = export_manifest(context, self.export_scope, None,
                              self.skip_hidden, target)
        box = layout.box()
        if not man['objects']:
            sub = box.column()
            sub.alert = True
            sub.label(text="Nothing in scope", icon='ERROR')
            return
        box.label(text="%d object(s), %d triangle(s)"
                       % (len(man['objects']), man['triangles']),
                  icon='OUTLINER_OB_MESH')
        for o in man['objects'][:6]:
            box.label(text="    " + o.name)
        if len(man['objects']) > 6:
            box.label(text="    ... and %d more" % (len(man['objects']) - 6))
        if man['ngons']:
            box.label(text="%d non-triangle polygon(s)%s" %
                      (man['ngons'], ", will be triangulated"
                       if will_triangulate else
                       " - the engine drops these"),
                      icon='INFO' if will_triangulate else 'ERROR')
        if man['merged']:
            box.label(text="merged into one layer", icon='INFO')
        if man['hidden']:
            sub = box.column()
            sub.alert = True
            sub.label(text="%d hidden, being skipped" % len(man['hidden']),
                      icon='HIDE_ON')
            for o in man['hidden'][:4]:
                sub.label(text="    " + o.name)
    apply_modifiers: BoolProperty(
        name="Apply Modifiers",
        description="Export each object's modifier-evaluated mesh (e.g. "
                    "Mirror, Subdivision, Smooth by Angle) instead of its "
                    "raw edit-mode geometry. Off by default: the mesh you "
                    "see in the outliner is the mesh that gets written, so "
                    "an export round-trips this addon's own import "
                    "unchanged and a Subdivision modifier left on a cage "
                    "cannot silently multiply the exported triangle count",
        default=False,
    )

    def _target_export_name(self, context):
        """Best-guess name for the object(s) about to be exported, used to
        pre-fill the file dialog's filename instead of leaving it as the
        .blend file's own name (ExportHelper's default). Only returns a
        name when there's one clear target: either a single exportable
        object, or an active object that's itself among the exportable
        set. Multiple objects with no single obvious name are left alone
        — falls back to ExportHelper's own blend-filename default."""
        objs = self._scope_objects(context)
        if len(objs) == 1:
            return objs[0].name
        active = context.active_object
        if active is not None and active in objs:
            return active.name
        return None

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
        """Return the actual multiplier to apply to each exported position."""
        if self.scale_mode == 'NONE':
            return 1.0
        if self.scale_mode == 'FACTOR':
            return self.scale_factor
        return 1.0 / MD5_SCALE_IN_TO_M

    def get_rotation_matrix(self):
        """Inverse of the matching IMPORT_ROTATION_ITEMS preset. These are
        all pure Z rotations, so X_TO_Y/Y_TO_X invert into each other,
        R180 is its own inverse, and NONE is the identity."""
        m = ROTATION_PRESETS.get(self.rotation)
        return m.inverted() if m else Matrix.Identity(4)




def _export_material_name(mat, obj, slot_index):
    """The exported material/surface name IS the in-game shader/material
    path for both formats (.lwo TAGS/SURF names go straight to
    declManager->FindMaterial(); .ase carries it in *BITMAP) — so an unnamed
    slot needs an honest placeholder rather than a blank string, which would
    collide with every other unnamed slot in the file.

    Resolved from the identity properties rather than the datablock name:
    Blender appends ".001" to a duplicate material name, and two models can
    legitimately share one decl, so the name is not reliable. The raw original
    wins when present, which is what makes an import/export round trip
    byte-faithful even though canonicalisation is lossy."""
    return material_decl_for_export(mat, f"{obj.name}_material{slot_index}")


def _export_transforms_for(obj, rot_matrix, scale):
    """Build the (position, normal) transform closures for one object —
    see ExportTransformMixin's docstring for the exact formula."""
    combined = rot_matrix @ obj.matrix_world
    normal_mat = combined.to_3x3().inverted_safe().transposed()

    def co_transform(local_co):
        v = combined @ local_co
        return Vector((v.x * scale, v.y * scale, v.z * scale))

    def normal_transform(local_n):
        n = normal_mat @ local_n
        if n.length_squared > 0.0:
            n.normalize()
        return n

    return co_transform, normal_transform


def _number_export_points(points, verts, source_point):
    """Assign each object vertex its index in the file's PNTS list.

    A vertex the importer split off a duplicate polygon (see the archived
    changelog) folds back onto the point it came from, so the file is written
    exactly as it was read and the engine shades it identically -- but only
    while the two are still in the same place. Move one and it becomes a real
    new point, which is a visible edit to visible geometry.
    """
    out = [None] * len(verts)
    if not source_point or len(source_point) != len(verts):
        base = len(points)
        points.extend(verts)
        return list(range(base, base + len(verts)))

    for i, v in enumerate(verts):
        if source_point[i] < 0:
            out[i] = len(points)
            points.append(v)
    for i, v in enumerate(verts):
        if out[i] is not None:
            continue
        src = source_point[i]
        if 0 <= src < len(verts) and out[src] is not None and verts[src] == v:
            out[i] = out[src]           # still coincident: same point
        else:
            out[i] = len(points)        # moved, or its source is gone
            points.append(v)
    return out


def _gather_export_point_maps(obj, mesh):
    """Per-point maps to write back: weight/selection maps and morph targets.

    A float POINT attribute wins over the same-named vertex group, because
    that is where the importer keeps values a vertex group cannot hold (it
    clamps to 0..1 and LightWave weight maps do not).
    """
    out = {'wght': [], 'pick': [], 'morf': []}
    nverts = len(mesh.vertices)
    meta = {}
    for entry in (obj.data.get('idtech4_point_map_kinds') or {}).items():
        meta[entry[0]] = entry[1]

    seen = set()
    for attr in mesh.attributes:
        if attr.domain != 'POINT' or attr.data_type != 'FLOAT':
            continue
        vals = [0.0] * nverts
        attr.data.foreach_get('value', vals)
        kind = meta.get(attr.name, 'WGHT')
        seen.add(attr.name)
        target = out['pick'] if kind == 'PICK' else out['wght']
        target.append((attr.name, vals))

    for vg in obj.vertex_groups:
        if vg.name in seen:
            continue
        vals = [0.0] * nverts
        any_set = False
        for i in range(nverts):
            try:
                vals[i] = vg.weight(i)
                any_set = True
            except RuntimeError:
                pass            # vertex simply is not in this group
        if any_set:
            out['wght'].append((vg.name, vals))

    keys = mesh.shape_keys
    if keys and len(keys.key_blocks) > 1:
        basis = keys.key_blocks[0]
        for kb in list(keys.key_blocks)[1:]:
            deltas = {}
            for i in range(nverts):
                d = kb.data[i].co - basis.data[i].co
                if d.length_squared > 0.0:
                    deltas[i] = (d.x, d.y, d.z)
            if deltas:
                out['morf'].append((kb.name, deltas))
    return out


def _gather_export_triangles(obj, depsgraph, co_transform, normal_transform,
                              apply_modifiers=True, triangulate=True,
                              vertex_domain_colors=False):
    """Evaluate *obj* and return (verts, tris, has_uv, has_col).

    With *triangulate* (the default) the mesh is split into triangles via
    mesh.calc_loop_triangles(), which is what idTech4 needs — it renders only
    already-triangulated polygons. With it False each Blender polygon is
    emitted as-is, n-gons included, which .lwo supports natively (up to 1023
    vertices) and which a lossless round trip requires. .ase is
    triangle-only by construction, so its writer always passes True.

    Each entry also carries 'ptype', the source POLS chunk type recorded by
    the importer (see LWO_POLY_TYPES); 0 for anything Blender-authored.

      verts = [Vector, ...]     -- already run through co_transform,
                                    indexed exactly like mesh.vertices
      tris  = [ {'v': (i0,i1,i2), 'uv': (uv0,uv1,uv2),
                 'col': (rgba0,rgba1,rgba2) or (None,None,None),
                 'n': (Vector,Vector,Vector), 'smooth': bool, 'mat': name},
                ... ]           -- one entry per triangle, corners in
                                    Blender loop order (mesh.loop_triangles'
                                    own order — NOT yet corrected for
                                    either file format's winding; each
                                    writer applies its own correction)
      has_uv / has_col          -- whether the source mesh had a UV layer /
                                    corner color attribute at all (used to
                                    decide whether to warn, and — for
                                    color — whether to write the block at
                                    all vs. relying on the file format's
                                    own white/surface-color fallback)

    Normals come from mesh.loop_triangles[i].split_normals — the exact
    per-corner normal Blender is CURRENTLY shading with, regardless of
    whether that comes from flat/smooth polygon flags, a Smooth-by-Angle
    modifier, or hand-authored custom split normals. Colors come from the
    mesh's "Col"-named corner color attribute if present (matching what
    this addon's own importer always names it), else the first corner
    color attribute found, else None per corner.
    """
    if apply_modifiers:
        obj_eval = obj.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh()
        mesh_owner = obj_eval
    else:
        mesh = obj.to_mesh()
        mesh_owner = obj

    if mesh is None or not mesh.polygons:
        if mesh is not None:
            mesh_owner.to_mesh_clear()
        return [], [], False, False

    try:
        if triangulate:
            mesh.calc_loop_triangles()

        uv_layer = mesh.uv_layers.active
        color_attr = None
        for ca in mesh.color_attributes:
            if ca.domain == 'CORNER' and ca.name.lower() == 'col':
                color_attr = ca
                break
        if color_attr is None:
            for ca in mesh.color_attributes:
                if ca.domain == 'CORNER':
                    color_attr = ca
                    break
        # Face Corner keeps strict priority: it is the domain both file
        # formats are built around (.ase is corner-indexed by construction,
        # .lwo VMAD is per (polygon, point)) and the domain an import
        # produces, so this option can only ADD a source, never displace
        # one. A POINT attribute is fanned out to corners - every corner of
        # a vertex reads that vertex's single value - which is exactly the
        # degenerate case of per-corner data and therefore lossless.
        if color_attr is None and vertex_domain_colors:
            for ca in mesh.color_attributes:
                if ca.domain == 'POINT':
                    color_attr = ca
                    break

        mat_names = [_export_material_name(m, obj, i) for i, m in enumerate(mesh.materials)]
        if not mat_names:
            mat_names = [obj.name]

        verts = [co_transform(v.co) for v in mesh.vertices]

        smgp_attr = mesh.attributes.get(LWO_SMGP_ATTR)
        smgps = None
        if smgp_attr is not None and smgp_attr.domain == 'FACE':
            smgps = [0] * len(mesh.polygons)
            smgp_attr.data.foreach_get('value', smgps)

        ptype_attr = mesh.attributes.get(LWO_POLY_TYPE_ATTR)
        ptypes = None
        if ptype_attr is not None and ptype_attr.domain == 'FACE':
            ptypes = [0] * len(mesh.polygons)
            ptype_attr.data.foreach_get('value', ptypes)

        shades_smooth = _faces_that_shade_smooth(mesh)

        # Every named map, in Blender's own order, with the active one first
        # so index 0 stays the map the engine will actually use.
        uv_layers = list(mesh.uv_layers)
        if uv_layer is not None:
            uv_layers.sort(key=lambda L: L.name != uv_layer.name)
        col_attrs = [ca for ca in mesh.color_attributes if ca.domain == 'CORNER']
        if vertex_domain_colors:
            col_attrs += [ca for ca in mesh.color_attributes
                          if ca.domain == 'POINT']
        if color_attr is not None:
            col_attrs.sort(key=lambda a: a.name != color_attr.name)

        def _color_reader(ca):
            """f(loop_index) -> rgba, for an attribute on either domain."""
            if ca is None:
                return None
            data = ca.data
            if ca.domain == 'CORNER':
                return lambda li: data[li].color
            loops = mesh.loops
            return lambda li: data[loops[li].vertex_index].color

        read_color = _color_reader(color_attr)

        def _corner(li):
            uv = tuple(uv_layer.data[li].uv) if uv_layer is not None else (0.0, 0.0)
            if read_color is not None:
                c = read_color(li)
                col = (c[0], c[1], c[2], c[3])
            else:
                col = None
            return uv, col

        def _extra(loops):
            """Per-corner values for every map beyond the active one."""
            out_uv = {}
            for L in uv_layers[1:]:
                out_uv[L.name] = tuple(tuple(L.data[li].uv) for li in loops)
            out_col = {}
            for a in col_attrs[1:]:
                rd = _color_reader(a)
                out_col[a.name] = tuple(tuple(rd(li)) for li in loops)
            return out_uv, out_col

        tris = []
        if triangulate:
            for lt in mesh.loop_triangles:
                mi = lt.material_index if lt.material_index < len(mat_names) else 0
                corner_uv, corner_col, corner_n = [], [], []
                for k in range(3):
                    uv, col = _corner(lt.loops[k])
                    corner_uv.append(uv)
                    corner_col.append(col)
                    corner_n.append(normal_transform(Vector(lt.split_normals[k])))
                ex_uv, ex_col = _extra([lt.loops[k] for k in range(3)])
                tris.append({
                    'v': tuple(lt.vertices), 'uv': tuple(corner_uv),
                    'col': tuple(corner_col), 'n': tuple(corner_n),
                    'smooth': shades_smooth[lt.polygon_index], 'mat': mat_names[mi],
                    'ptype': (ptypes[lt.polygon_index] if ptypes else 0),
                    'smgp': (smgps[lt.polygon_index] if smgps else None),
                    'uvs': ex_uv, 'cols': ex_col,
                })
        else:
            # Untriangulated: LWO carries polygons up to 1023 vertices, so a
            # quad/ngon source can round trip. Per-corner normals come from
            # mesh.corner_normals, which is whatever Blender is currently
            # shading with -- the same source loop_triangles.split_normals
            # reads.
            corner_normals = None
            try:
                corner_normals = mesh.corner_normals
            except Exception:
                corner_normals = None
            for poly in mesh.polygons:
                mi = poly.material_index if poly.material_index < len(mat_names) else 0
                idxs, corner_uv, corner_col, corner_n = [], [], [], []
                for li in range(poly.loop_start, poly.loop_start + poly.loop_total):
                    uv, col = _corner(li)
                    idxs.append(mesh.loops[li].vertex_index)
                    corner_uv.append(uv)
                    corner_col.append(col)
                    if corner_normals is not None:
                        corner_n.append(normal_transform(Vector(corner_normals[li].vector)))
                    else:
                        corner_n.append(normal_transform(Vector(poly.normal)))
                ex_uv, ex_col = _extra(list(range(poly.loop_start,
                                                  poly.loop_start + poly.loop_total)))
                tris.append({
                    'v': tuple(idxs), 'uv': tuple(corner_uv),
                    'col': tuple(corner_col), 'n': tuple(corner_n),
                    'smooth': shades_smooth[poly.index], 'mat': mat_names[mi],
                    'ptype': (ptypes[poly.index] if ptypes else 0),
                    'smgp': (smgps[poly.index] if smgps else None),
                    'uvs': ex_uv, 'cols': ex_col,
                })

        has_uv = uv_layer is not None
        has_col = color_attr is not None
        extras = _gather_export_point_maps(obj, mesh)
        src_attr = mesh.attributes.get(LWO_SOURCE_POINT_ATTR)
        extras['source_point'] = None
        if src_attr is not None and src_attr.domain == 'POINT':
            sp = [0] * len(mesh.vertices)
            src_attr.data.foreach_get('value', sp)
            extras['source_point'] = sp
        part_attr = mesh.attributes.get(LWO_PART_ATTR)
        if part_attr is not None and part_attr.domain == 'FACE':
            pvals = [0] * len(mesh.polygons)
            part_attr.data.foreach_get('value', pvals)
            if triangulate:
                for t, lt in zip(tris, mesh.loop_triangles):
                    t['part'] = pvals[lt.polygon_index]
            else:
                for t, poly in zip(tris, mesh.polygons):
                    t['part'] = pvals[poly.index]
    finally:
        mesh_owner.to_mesh_clear()

    return verts, tris, has_uv, has_col, extras


# ── .ase EXPORT ─────────────────────────────────────────────────────

def _ase_vec_str(v):
    return f"{v.x:.6f}\t{v.y:.6f}\t{v.z:.6f}"


def build_ase_export_text(objects, depsgraph, rot_matrix, scale,
                           export_normals=True, export_colors=True,
                           apply_modifiers=True, vertex_domain_colors=False):
    """Build the full text of an .ase file for *objects*. Returns
    (text, material_order, warnings); text is None (nothing written) if no
    object produced any triangle.

    One *GEOMOBJECT per (Blender object, material) combination. Splitting
    by material is a hard
    engine requirement, not a stylistic choice: real idTech4 only ever
    reads ONE material per *GEOMOBJECT (*MATERIAL_REF), and completely
    ignores *MESH_MTLID.

    Each *GEOMOBJECT gets its own private, entirely UNSHARED vertex list —
    one *MESH_VERTEX/*MESH_TVERT/*MESH_VERTCOL entry per triangle CORNER,
    never reused across triangles even where two corners share a position.
    This sidesteps needing to replicate the real engine's own vertex-
    welding pass (ConvertASEToModelSurfaces' "matchVert" hashing, keyed on
    position+uv+color+normal all matching within a slop epsilon) — the
    engine performs that welding itself at load time regardless of
    whether the file already shares indices, so an unshared, per-corner
    vertex list round-trips to IDENTICAL in-game geometry.

    Winding: corners are written in the SAME order Blender stores them
    (A,B,C = corner0,corner1,corner2), no swap — see this module's
    parse_ase_file docstring for why an earlier B/C swap (mirroring the
    engine's own internal correction) was reverted after it was found to
    invert normals on import; the export side is kept symmetric with it.

    Every *GEOMOBJECT gets an explicit identity *NODE_TM — there is an
    uninitialized-memory bug in the engine's parser that makes
    omitting it unsafe (it would silently corrupt any exported
    *MESH_VERTEXNORMAL data via a garbage rotation).

    *MESH_NORMALS is written whenever *export_normals* is True (matching
    the exact per-corner shading Blender currently shows — see
    _gather_export_triangles) — the real engine uses these DIRECTLY for
    any material without a renderBump command (ConvertASEToModelSurfaces),
    so this is the only way to get an exact idTech4 shading match. With it
    False, the engine instead auto-generates its own normals — always
    fully smoothed across every corner sharing a position+uv+color,
    regardless of the source mesh's actual shading.

    *MESH_CVERTLIST/*MESH_CFACELIST is only written for a *GEOMOBJECT that
    actually has vertex-color data (real or from *export_colors*'s
    caller) — omitting it leaves colorsParsed False in-engine, which
    falls back to opaque identity white, matching this addon's own
    importer default.

    A *MATERIAL's only property the real engine ever reads is its
    *MAP_DIFFUSE *BITMAP path (ASE_KeyMATERIAL has no case for
    *MATERIAL_NAME at all) — declManager->FindMaterial() looks up the
    shader by the qpath idFileSystemLocal::OSPathToRelativePath()
    resolves that string down to (framework/FileSystem.cpp), which is
    literally the exact Blender material name once resolved. It's
    written here as "\base\" + that name, backslashed — the leading
    "\base\" is NOT decorative:
    OSPathToRelativePath strips everything up to and including the
    FIRST complete "base" path component it finds, so a bare (no
    synthetic prefix) material name that legitimately contains its own
    "base" subfolder further in (e.g.
    "models/mapobjects/base/chairs/chair3.tga" — Doom 3's real "base"
    episode assets) would have that inner "base" mistaken for the OS
    asset root and get truncated on reload, by this addon's own importer
    AND by the real engine. The synthetic prefix guarantees OUR "base"
    is always the leftmost match, so the full original name always comes
    back intact — a no-op for every material name that was never
    ambiguous to begin with.
    """
    warnings = []
    material_order = []
    material_seen = set()
    material_source = {}
    geoms = []

    for obj in objects:
        # Only the decl reaches the geometry loop below, so remember which
        # datablock produced it while both are still in hand -- the *BITMAP
        # field is chosen from the material's own identity, not from the decl
        # string alone.
        for slot, mat in enumerate(obj.data.materials):
            if mat is None:
                continue
            material_source.setdefault(
                _export_material_name(mat, obj, slot), mat)

        co_xf, n_xf = _export_transforms_for(obj, rot_matrix, scale)
        verts, tris, has_uv, has_col, _extras = _gather_export_triangles(
            obj, depsgraph, co_xf, n_xf, apply_modifiers,
            vertex_domain_colors=vertex_domain_colors)
        if not tris:
            continue
        if not has_uv:
            warnings.append(f'"{obj.name}" has no UV map — every face will '
                            f'import with (0,0) texture coordinates.')

        by_mat = {}
        for tri in tris:
            by_mat.setdefault(tri['mat'], []).append(tri)
        multi = len(by_mat) > 1

        for gi, (mat_name, mtris) in enumerate(by_mat.items()):
            if mat_name not in material_seen:
                material_seen.add(mat_name)
                material_order.append(mat_name)

            g_verts, g_uvs, g_cols = [], [], []
            g_norms = [] if export_normals else None
            g_tris = []
            any_col = False
            for tri in mtris:
                corner_idx = []
                for k in range(3):
                    vi = tri['v'][k]
                    g_verts.append(verts[vi])
                    g_uvs.append(tri['uv'][k])
                    c = tri['col'][k]
                    # "has a colour attribute" is not the test — see below.
                    # Only colour that actually says something is worth the
                    # *MESH_CVERTLIST it would cost.
                    if c is not None and any(abs(x - 1.0) > (1.0 / 255.0)
                                             for x in c[:3]):
                        any_col = True
                    g_cols.append(c if c is not None else (1.0, 1.0, 1.0))
                    if export_normals:
                        g_norms.append(tri['n'][k])
                    corner_idx.append(len(g_verts) - 1)
                g_tris.append(tuple(corner_idx))

            geoms.append({
                'name': obj.name if not multi else f"{obj.name}_{gi}",
                'material': mat_name,
                'verts': g_verts, 'uvs': g_uvs,
                'cols': g_cols if (export_colors and any_col) else None,
                'norms': g_norms,
                'tris': g_tris,
            })

    if not geoms:
        return None, [], ["No exportable mesh geometry found (no "
                          "triangulated faces on any selected object)."]

    if any(g['cols'] is not None for g in geoms):
        # Say this out loud rather than dropping the data or writing it
        # silently. ASE_KeyMESH_CVERTLIST (renderer/Model_ase.cpp) reads the
        # component with atof(token) where `token` is the dispatch keyword
        # "*MESH_VERTCOL", not the value it just tokenised into ase.token — so
        # every component parses as 0.0 and the whole list comes back BLACK.
        # The typo is in all ten engine variants under source/, BFG included,
        # so there is no build where a *MESH_CVERTLIST reads back correctly.
        # The block is still written, because it is valid ASE that this
        # addon's own importer and other tools read properly, and dropping it
        # would lose the author's paint outright.
        warnings.append(
            "Vertex colours were written as *MESH_CVERTLIST. idTech4 itself "
            "misparses that block and reads every colour as BLACK "
            "(ASE_KeyMESH_CVERTLIST passes the keyword to atof instead of the "
            "value) — harmless for a material that ignores vertex colour, but "
            "a material using \"vertexColor\" will render black in game. "
            "Export as .lwo if the colours have to survive.")

    L = []
    L.append("*3DSMAX_ASCIIEXPORT\t200")
    L.append('*COMMENT "Exported by idTech4 .ase/.lwo Exporter for Blender"')
    L.append("*SCENE {")
    L.append('\t*SCENE_FILENAME ""')
    L.append("\t*SCENE_FIRSTFRAME 0")
    L.append("\t*SCENE_LASTFRAME 100")
    L.append("\t*SCENE_FRAMESPEED 30")
    L.append("\t*SCENE_TICKSPERFRAME 160")
    L.append("}")

    L.append("*MATERIAL_LIST {")
    L.append(f"\t*MATERIAL_COUNT {len(material_order)}")
    for mi, mat_name in enumerate(material_order):
        # The engine reads the *BITMAP and nothing else (ASE_KeyMATERIAL has no
        # case for *MATERIAL_NAME at all), so this one string decides whether
        # the model is visible. material_source_string picks it and CHECKS it,
        # by resolving the candidate exactly the way ASE_KeyMAP_DIFFUSE will.
        #
        # The "\base\" wrap it falls back on is NOT decorative. It is what
        # keeps a RELATIVE decl safe from idFileSystemLocal::OSPathToRelativePath
        # (framework/FileSystem.cpp), which strips everything up to and
        # including the FIRST complete "base" path component, left to right. A
        # genuine 3ds Max export has an OS path ahead of its real asset root,
        # so that root is always the leftmost match; a bare decl does not, and
        # one that legitimately contains its own "base" subfolder (e.g.
        # "models/mapobjects/base/chairs/chair3.tga" — Doom 3's "base" episode
        # assets, which really are in the corpus) would be mis-truncated to
        # "chairs/chair3.tga" on reload, by this addon's importer AND by the
        # engine. Prepending our own "\base\" guarantees OUR marker is the
        # leftmost match. The leading backslash matters too: the boundary check
        # wants a separator immediately BEFORE "base", which a bare "base\..."
        # at position 0 would not satisfy.
        #
        # What it must NEVER be applied to is a path that already carries an
        # anchor. Prefixing "\base\" onto "\\purgatory\...\doom\base\models\..."
        # makes OUR marker win, the engine strips at it, and what is left is
        # still the whole original absolute path — which then resolves to a
        # decl that cannot exist. Hence wrap= being reachable only from the
        # by-construction branch, and hence the verification.
        bitmap_path = material_source_string(
            material_source.get(mat_name), mat_name,
            engine_ase_bitmap_decl,
            wrap=lambda d: "\\base\\" + d.replace('/', '\\'))
        L.append(f"\t*MATERIAL {mi} {{")
        L.append(f'\t\t*MATERIAL_NAME "{mat_name}"')
        L.append('\t\t*MATERIAL_CLASS "Standard"')
        L.append("\t\t*MAP_DIFFUSE {")
        L.append('\t\t\t*MAP_NAME "map1"')
        L.append('\t\t\t*MAP_CLASS "Bitmap"')
        L.append(f'\t\t\t*BITMAP "{bitmap_path}"')
        L.append("\t\t\t*UVW_U_OFFSET 0.0000")
        L.append("\t\t\t*UVW_V_OFFSET 0.0000")
        L.append("\t\t\t*UVW_U_TILING 1.0000")
        L.append("\t\t\t*UVW_V_TILING 1.0000")
        L.append("\t\t\t*UVW_ANGLE 0.0000")
        L.append("\t\t}")
        L.append("\t}")
    L.append("}")

    mat_index = {name: i for i, name in enumerate(material_order)}

    for g in geoms:
        n, nt = len(g['verts']), len(g['tris'])
        L.append("*GEOMOBJECT {")
        L.append(f'\t*NODE_NAME "{g["name"]}"')
        L.append("\t*NODE_TM {")
        L.append(f'\t\t*NODE_NAME "{g["name"]}"')
        L.append("\t\t*TM_ROW0 1.0000 0.0000 0.0000")
        L.append("\t\t*TM_ROW1 0.0000 1.0000 0.0000")
        L.append("\t\t*TM_ROW2 0.0000 0.0000 1.0000")
        L.append("\t\t*TM_ROW3 0.0000 0.0000 0.0000")
        L.append("\t}")
        L.append("\t*MESH {")
        L.append("\t\t*TIMEVALUE 0")
        L.append(f"\t\t*MESH_NUMVERTEX {n}")
        L.append(f"\t\t*MESH_NUMFACES {nt}")
        L.append("\t\t*MESH_VERTEX_LIST {")
        for vi, v in enumerate(g['verts']):
            L.append(f"\t\t\t*MESH_VERTEX {vi}\t{_ase_vec_str(v)}")
        L.append("\t\t}")
        L.append("\t\t*MESH_FACE_LIST {")
        for fi, (a, b, c) in enumerate(g['tris']):
            L.append(f"\t\t\t*MESH_FACE {fi}:\tA:\t{a}\tB:\t{b}\tC:\t{c}\t"
                     f"AB:\t1\tBC:\t1\tCA:\t1\t*MESH_SMOOTHING 0\t*MESH_MTLID 0")
        L.append("\t\t}")

        L.append(f"\t\t*MESH_NUMTVERTEX {n}")
        L.append("\t\t*MESH_TVERTLIST {")
        for vi, (u, v) in enumerate(g['uvs']):
            L.append(f"\t\t\t*MESH_TVERT {vi}\t{u:.6f}\t{v:.6f}\t0.0000")
        L.append("\t\t}")
        L.append(f"\t\t*MESH_NUMTVFACES {nt}")
        L.append("\t\t*MESH_TFACELIST {")
        for fi, (a, b, c) in enumerate(g['tris']):
            L.append(f"\t\t\t*MESH_TFACE {fi}\t{a}\t{b}\t{c}")
        L.append("\t\t}")

        if g['cols'] is not None:
            L.append(f"\t\t*MESH_NUMCVERTEX {n}")
            L.append("\t\t*MESH_CVERTLIST {")
            for vi, col in enumerate(g['cols']):
                L.append(f"\t\t\t*MESH_VERTCOL {vi}\t{col[0]:.6f}\t{col[1]:.6f}\t{col[2]:.6f}")
            L.append("\t\t}")
            L.append(f"\t\t*MESH_NUMCVFACES {nt}")
            L.append("\t\t*MESH_CFACELIST {")
            for fi, (a, b, c) in enumerate(g['tris']):
                L.append(f"\t\t\t*MESH_CFACE {fi}\t{a}\t{b}\t{c}")
            L.append("\t\t}")

        if g['norms'] is not None:
            L.append("\t\t*MESH_NORMALS {")
            for fi, (a, b, c) in enumerate(g['tris']):
                flat = _face_flat_normal(g['verts'][a], g['verts'][b], g['verts'][c])
                L.append(f"\t\t\t*MESH_FACENORMAL {fi}\t{_ase_vec_str(flat)}")
                L.append(f"\t\t\t*MESH_VERTEXNORMAL {a}\t{_ase_vec_str(g['norms'][a])}")
                L.append(f"\t\t\t*MESH_VERTEXNORMAL {b}\t{_ase_vec_str(g['norms'][b])}")
                L.append(f"\t\t\t*MESH_VERTEXNORMAL {c}\t{_ase_vec_str(g['norms'][c])}")
            L.append("\t\t}")

        L.append("\t}")
        L.append(f"\t*MATERIAL_REF {mat_index[g['material']]}")
        L.append("}")

    return "\n".join(L) + "\n", material_order, warnings


class EXPORT_OT_idtech4_ase(bpy.types.Operator, ExportHelper, ExportTransformMixin):
    """Export selected mesh objects as an idTech4 .ase static mesh"""
    bl_idname  = "export_scene.idtech4_ase"
    bl_label   = "Export idTech4 .ase"
    bl_options = {'PRESET'}

    filename_ext = ".ase"
    filter_glob: StringProperty(default="*.ase", options={'HIDDEN'}, maxlen=255)

    export_normals: BoolProperty(
        name="Export Normals/Smoothing",
        description="Write each face corner's exact current Blender "
                    "shading normal (flat, smooth, custom split, "
                    "Smooth-by-Angle — whatever the viewport currently "
                    "shows) as *MESH_NORMALS data. Real idTech4/Doom 3 "
                    "uses this directly for any material without a "
                    "renderBump command. Disable to omit *MESH_NORMALS "
                    "entirely and let the engine generate its own "
                    "(always fully smoothed) normals instead",
        default=True,
    )
    export_colors: BoolProperty(
        name="Export Vertex Colors ( Face Corner )",
        description="Export Blender Vertex Paint Face-Corner attributes",
        default=True,
    )
    export_vertex_domain_colors: BoolProperty(
        name="Export Vertex Colors ( Vertex )",
        description="Blender Vertex Paint Vertex Color attributes will be "
                    "converted to Face Corner and export",
        default=False,
    )

    def invoke(self, context, event):
        name = self._target_export_name(context)
        if name:
            directory = os.path.dirname(self.filepath) if self.filepath \
                else os.path.dirname(context.blend_data.filepath)
            self.filepath = os.path.join(directory, name + self.filename_ext)
        return ExportHelper.invoke(self, context, event)

    def draw(self, context):
        self.draw_scope(self.layout, context, 'ASE')
        self.layout.prop(self, "apply_modifiers")
        self.draw_transforms(self.layout)
        self.layout.prop(self, "export_normals")
        self.layout.prop(self, "export_colors")
        self.layout.prop(self, "export_vertex_domain_colors")

    @exports_in_object_mode
    def execute(self, context):
        objects = self._scope_objects(context)
        if not objects:
            considered = iter_export_objects(context, self.export_scope, None,
                                             False)
            if considered and self.skip_hidden:
                self.report({'ERROR'}, "Every mesh in scope (%d) is hidden in "
                                       "the outliner." % len(considered))
            else:
                self.report({'ERROR'}, "No mesh objects in scope to export.")
            return {'CANCELLED'}
        depsgraph = context.evaluated_depsgraph_get()
        text, materials, warnings = build_ase_export_text(
            objects, depsgraph, self.get_rotation_matrix(), self.get_scale(),
            export_normals=self.export_normals, export_colors=self.export_colors,
            apply_modifiers=self.apply_modifiers,
            vertex_domain_colors=self.export_vertex_domain_colors)

        if text is None:
            self.report({'ERROR'}, "; ".join(warnings) or "Nothing to export.")
            return {'CANCELLED'}

        try:
            with open(self.filepath, 'w', encoding='utf-8', newline='\n') as fh:
                fh.write(text)
        except OSError as exc:
            self.report({'ERROR'}, f"Could not write \"{self.filepath}\": {exc}")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Exported {len(objects)} object(s), "
                              f"{len(materials)} material(s) to \"{self.filepath}\"")
        for w in warnings:
            self.report({'WARNING'}, w)
        return {'FINISHED'}


# ── .lwo (LWO2) EXPORT ──────────────────────────────────────────────

def _lwo_write_vx(idx):
    """Encode one LWO2 variable-length index (VX) — inverse of
    _lwo_read_vx (see its docstring for the bit layout)."""
    if idx < 0xFF00:
        return struct.pack('>H', idx)
    return struct.pack('>H', 0xFF00 | ((idx >> 16) & 0xFF)) + struct.pack('>H', idx & 0xFFFF)


def _lwo_write_cstr(s):
    """Inverse of _lwo_read_cstr: null-terminated, padded to an even byte
    length including that terminator."""
    b = s.encode('utf-8', 'replace') + b'\x00'
    if len(b) % 2 == 1:
        b += b'\x00'
    return b


def _lwo_chunk(cid, payload):
    """One top-level IFF chunk: 4-byte id, 4-byte big-endian size (of
    *payload* only), the payload itself, then a single pad byte if that
    size is odd — matching parse_lwo_file's own `if csize % 2 == 1: off
    += 1` chunk-alignment rule exactly."""
    out = cid + struct.pack('>I', len(payload)) + payload
    if len(payload) % 2 == 1:
        out += b'\x00'
    return out


def _lwo_subchunk(cid, payload):
    """A SURF subchunk: identical framing to _lwo_chunk but with a 2-byte
    size field, matching lwGetSurface's own `sub_sz = unpack('>H', ...)`."""
    out = cid + struct.pack('>H', len(payload)) + payload
    if len(payload) % 2 == 1:
        out += b'\x00'
    return out


def build_lwo_export_data(objects, depsgraph, rot_matrix, scale,
                           export_normals=True, export_colors=True,
                           apply_modifiers=True, weld_points=False,
                           triangulate=True, all_layers=False,
                           preserve_smoothing=True,
                           vertex_domain_colors=False, layer_groups=None):
    """Build the full byte content of an .lwo (LWO2) file merging every
    triangle of *objects* into a SINGLE layer. Returns
    (data: bytes, material_order, warnings); data is None if no object
    produced any triangle.

    *layer_groups* is a list of object lists, one per LWO layer;
    omitted, every object goes into one layer. One layer is what the
    engine reads:
    ConvertLWOToModelSurfaces only ever reads `lwo->layer` (literally the
    head of the LAYR linked list), so any further layer is silently dead,
    never-rendered geometry in-game. Several layers are still worth
    writing for a faithful round trip or for authoring in LightWave,
    which is what EXPORT_OT_idtech4_lwo's "Merge Into Single Layer" off
    now produces. TAGS and SURF are file-global and emitted once around
    the layers; everything index-bearing is rebuilt per layer.

    *weld_points* (default False) pools points by rounded position ACROSS
    every object passed in, so touching pieces weld the way a single
    hand-modeled multi-part LWO layer would. It is safe for per-corner
    UV/color fidelity, because both are written as VMAD (poly, point)
    pairs rather than per-point VMAP data — two triangles sharing a welded
    point can still carry completely different UV/color at that corner
    (see the two-stage VMAP-then-VMAD resolution this module's own
    parser/importer already implements the read side of). It is NOT safe
    for geometry: welding can collapse a triangle into a degenerate or a
    duplicate, and nothing here detects that. It is off by default because
    the engine never welds a .lwo on load — see the archived
    changelog — so leaving it off round-trips this addon's own
    lossless import unchanged.

    Axis convention and polygon winding: both are the exact same
    self-inverse operations parse_lwo_file/build_lwo_meshes apply on
    read — see parse_lwo_file's docstring for the full derivation. PNTS
    values are written as (x, z, y) from each already Blender-space
    (Z-up) position, and each polygon's point list is written in REVERSED
    order.

    Normals are NEVER stored directly: the real engine (lwGetVertNormals/_lwo_compute_corner_normals'
    own faithful port of it) always RECOMPUTES per-corner shading from
    each SURF's SMAN smoothing angle + PTAG SMGP smoothing groups + raw
    geometry, never from any stored normal. When *export_normals* is True
    this function instead reproduces each triangle's flat/smooth
    (loop_triangle.use_smooth) shading EXACTLY via SMGP islands: every
    smooth-shaded triangle in a given material shares ONE group id unique
    to that material (so smoothing can never bleed across a material
    boundary, even where two different materials' geometry touches), a
    wide-open SMAN angle (179.9°, i.e. "always merge within the group")
    makes them actually merge, and every flat-shaded triangle gets a group
    id none of the faces it shares a point with uses, so it never merges
    with anything and stays fully faceted. Those flat ids are assigned by
    greedy colouring rather than handed out one per face: lwGetVertNormals
    only ever compares faces that share a point, so per-face uniqueness was
    never needed, and PTAG stores the id in a U2 that one-per-face
    overflowed on any mesh past 65535 flat faces. A material with no smooth-shaded triangles at
    all gets SMAN 0 (fully faceted, matching the format's own default for
    a surface that never sets SMAN). This exactly reproduces a per-face
    flat/smooth toggle — it can NOT reproduce an arbitrary custom split
    normal or a numeric Smooth-by-Angle threshold (LWO simply has no
    mechanism the engine reads for that); export .ase instead when exact
    per-corner normal fidelity matters more than authoring in LightWave.
    With *export_normals* False, every material is forced fully faceted
    (SMAN 0) regardless of the source mesh's shading.

    VMAD (not VMAP) is used for both UV and vertex color, unconditionally
    — a discontinuous per-(polygon,point) chunk can represent anything a
    continuous per-point one can, so this needs no "is this UV/color
    actually continuous across every use of this point" analysis, and
    still resolves correctly through this module's own two-stage
    VMAP-then-VMAD reader (whose VMAP pass simply never finds a match, so
    the VMAD pass alone determines every corner's value). V is NOT
    flipped when writing UV — see parse_lwo_file's docstring for why LWO's
    raw convention already matches Blender's (unlike .ase's).

    Vertex color is only written at all when *export_colors* is True AND
    at least one exported triangle actually has color data (from
    _gather_export_triangles) — a file with none omits the VMAD RGBA
    chunk entirely, so the real engine falls back to each SURF's COLR
    default (matching this addon's own importer behavior for a .lwo with
    no vertex color).
    """
    warnings = []
    material_order = []
    material_index = {}
    material_source = {}   # decl -> the datablock it came from, for TAGS/SURF
    # One group per LWO layer. The default is every object in one layer,
    # which is what the engine reads (ConvertLWOToModelSurfaces only ever
    # looks at lwo->layer, the head of the LAYR list).
    if layer_groups is None:
        layer_groups = [list(objects)]

    surf_has_smooth = {}   # material -> does ANY layer shade it smooth
    surf_sman_out = {}     # material -> the angle to write, when preserved
    layer_chunks = []      # every layer's chunks, in file order
    escrow_blobs = []
    # One pass per LAYER. TAGS and the SURF chunks are file-global in
    # LWO2 - a PTAG SURF index names a position in the single TAGS list -
    # so the material table and every surface property stay outside this
    # loop, while points, polygons, smoothing groups and vertex maps are
    # rebuilt per layer. Every chunk after a LAYR belongs to that layer,
    # which is what makes simple concatenation correct here.
    for _layer_i, _layer_objs in enumerate(layer_groups):
        _layer_name = (_layer_objs[0].name if len(_layer_objs) == 1
                       else 'Layer%d' % (_layer_i + 1))
        point_pool = {}
        points = []
        point_maps = {}     # (kind, name) -> {exported point index: values}
        surf_colr = {}      # material name -> its LightWave COLR
        surf_sman = {}      # material name -> its LightWave SMAN, in radians
        surf_extra = {}     # material name -> unmodelled SURF subchunk bytes
        polys = []
        any_uv_missing = False
        any_geometry = False

        for obj in _layer_objs:
            co_xf, n_xf = _export_transforms_for(obj, rot_matrix, scale)
            verts, tris, has_uv, has_col, extras = _gather_export_triangles(
                obj, depsgraph, co_xf, n_xf, apply_modifiers, triangulate,
                vertex_domain_colors=vertex_domain_colors)
            if not tris:
                continue
            any_geometry = True
            if not has_uv:
                any_uv_missing = True

            if weld_points:
                obj_point_idx = []
                for v in verts:
                    key = (round(v.x, 4), round(v.y, 4), round(v.z, 4))
                    pi = point_pool.get(key)
                    if pi is None:
                        pi = len(points)
                        point_pool[key] = pi
                        points.append(v)
                    obj_point_idx.append(pi)
            else:
                obj_point_idx = _number_export_points(points, verts,
                                                      extras.get('source_point'))

            for slot, mat in enumerate(obj.data.materials):
                if mat is None:
                    continue
                nm = material_decl_for_export(mat, _export_material_name(mat, obj, slot))
                material_source.setdefault(nm, mat)
                if mat.get(LWO_SMAN_PROP) is not None:
                    surf_sman[nm] = float(mat[LWO_SMAN_PROP])
                if mat.get(LWO_COLR_PROP) is not None:
                    surf_colr[nm] = tuple(mat[LWO_COLR_PROP])[:3]
                blob = mat.get(LWO_SURF_EXTRA_PROP)
                if blob:
                    try:
                        surf_extra[nm] = base64.b64decode(blob)
                    except Exception:
                        pass
            blob = obj.data.get(LWO_ESCROW_PROP)
            if blob:
                shape = list(obj.data.get(LWO_ESCROW_SHAPE_PROP) or (0, 0))
                escrow_blobs.append((obj.name, blob, shape,
                                     len(obj.data.vertices), len(obj.data.polygons)))

            for nm, vals in extras['wght']:
                point_maps.setdefault(('WGHT', nm), {}).update(
                    {obj_point_idx[i]: (v,) for i, v in enumerate(vals) if v != 0.0})
            for nm, vals in extras['pick']:
                point_maps.setdefault(('PICK', nm), {}).update(
                    {obj_point_idx[i]: () for i, v in enumerate(vals) if v != 0.0})
            for nm, deltas in extras['morf']:
                point_maps.setdefault(('MORF', nm), {}).update(
                    {obj_point_idx[i]: d for i, d in deltas.items()})

            for tri in tris:
                if tri['mat'] not in material_index:
                    material_index[tri['mat']] = len(material_order)
                    material_order.append(tri['mat'])
                if len(tri['v']) > 1023:
                    warnings.append(
                        "A polygon with %d vertices was skipped: LWO's POLS "
                        "vertex-count field is 10 bits, so 1023 is the maximum."
                        % len(tri['v']))
                    continue
                pidx = tuple(obj_point_idx[v] for v in tri['v'])
                polys.append({
                    'mat': tri['mat'], 'points': pidx,
                    'smooth': tri['smooth'], 'uv': tri['uv'], 'col': tri['col'],
                    'ptype': tri.get('ptype', 0), 'part': tri.get('part', 0),
                    'smgp': tri.get('smgp'),
                    'uvs': tri.get('uvs') or {}, 'cols': tri.get('cols') or {},
                })

        smgp = []
        for _m in material_order:
            surf_has_smooth.setdefault(_m, False)
        if export_normals:
            for poly in polys:
                if poly['smooth']:
                    surf_has_smooth[poly['mat']] = True

        # The file's own smoothing, when it has some and the caller wants it.
        # Reproducing SMAN + SMGP verbatim is the only way a .lwo can come back
        # shaded as it went in: the synthesised form below can express a per-face
        # flat/smooth toggle and nothing finer.
        # A recorded SMAN is the trigger, not a recorded SMGP: plenty of real
        # models carry a smoothing angle and no PTAG SMGP chunk at all, and the
        # engine treats a missing one as every polygon being in group 0 (lwPolygon
        # ::smoothgrp is simply left zero), which is what the .get below yields.
        preserved = (export_normals and preserve_smoothing
                     and any(m in surf_sman for m in material_order))
        if preserved:
            ids = [int(p.get('smgp') or 0) for p in polys]
            if any(g < 0 or g > 0xFFFF for g in ids):
                warnings.append(
                    "A smoothing-group id outside 0..65535 was found; PTAG stores "
                    "it in a U2, so the file's own smoothing could not be "
                    "preserved and groups were rebuilt from the flat/smooth flags.")
                preserved = False
            else:
                smgp = ids
                for m in material_order:
                    surf_sman_out[m] = surf_sman.get(m, 0.0)

        if export_normals and not preserved:

            # id 0 is never handed out; 1..len(material_order) are the per-material
            # "these all merge with each other" ids the smooth faces share.
            first_flat_id = 1 + len(material_order)
            smgp = [0] * len(polys)
            for pi, poly in enumerate(polys):
                if poly['smooth']:
                    smgp[pi] = 1 + material_index[poly['mat']]

            # A flat face must merge with nothing, and lwGetVertNormals only ever
            # considers faces that SHARE A POINT with it — so it needs an id its
            # point-neighbours don't have, not a globally unique one. Colouring
            # greedily gives the identical merge relation (and therefore identical
            # normals) while keeping every id inside PTAG's U2 tag field; handing
            # out one id per face overflowed that at 65535 flat faces. In practice
            # this lands at roughly the mesh's maximum vertex degree.
            point_polys = {}
            for pi, poly in enumerate(polys):
                for pt in poly['points']:
                    point_polys.setdefault(pt, []).append(pi)

            max_id = first_flat_id
            for pi, poly in enumerate(polys):
                if poly['smooth']:
                    continue
                taken = set()
                for pt in poly['points']:
                    for h in point_polys[pt]:
                        g = smgp[h]
                        if g:
                            taken.add(g)
                g = first_flat_id
                while g in taken:
                    g += 1
                smgp[pi] = g
                if g > max_id:
                    max_id = g

            if max_id > 0xFFFF:
                # Only reachable on pathological geometry — tens of thousands of
                # flat faces meeting at ONE point. Rather than write a tag no
                # spec-conforming reader could load, fall back to fully faceted,
                # which is what turning the option off would have produced.
                warnings.append(
                    "Too many flat-shaded faces meet at a single point to give "
                    "them distinct LWO smoothing groups (needed %d, the format "
                    "allows 65535). Exported fully faceted instead." % max_id)
                smgp = []
                for _m in material_order:
                    surf_has_smooth[_m] = False

        has_any_color = export_colors and any(any(c is not None for c in p['col']) for p in polys)

        layr_payload = (struct.pack('>HH', _layer_i, 0)
                        + struct.pack('>fff', 0.0, 0.0, 0.0)
                        + _lwo_write_cstr(_layer_name))
        pnts_payload = b''.join(struct.pack('>fff', p.x, p.z, p.y) for p in points)

        # Every payload below is accumulated into a list and joined once. Doing
        # it with `bytes +=` instead is quadratic — each += copies the whole
        # buffer — which cost 172s on a 145k-triangle model, almost all of it in
        # the VMAD UV loop's 437k concatenations onto a growing 5.8MB buffer. The
        # joined form produces byte-identical output in ~0.2s.
        # LWO uses one POLS chunk per polygon type, and a PTAG index is a
        # position in the CONCATENATED order of those chunks -- so the polygon
        # list is grouped by type first and every index below follows that order.
        # The sort is stable, so a single-type file (the normal case) keeps its
        # exact source order.
        polys.sort(key=lambda p: p.get('ptype', 0))

        pols_chunks = []
        for tindex, tname in enumerate(LWO_POLY_TYPES):
            group = [p for p in polys if p.get('ptype', 0) == tindex]
            if not group:
                continue
            parts = [tname]
            for poly in group:
                idxs = list(reversed(poly['points']))
                # the nverts field is 10 bits; the rest are flags
                parts.append(struct.pack('>H', len(idxs) & 0x03FF))
                for i in idxs:
                    parts.append(_lwo_write_vx(i))
            pols_chunks.append(_lwo_chunk(b'POLS', b''.join(parts)))

        ptag_surf_parts = [b'SURF']
        for pi, poly in enumerate(polys):
            ptag_surf_parts.append(_lwo_write_vx(pi)
                                   + struct.pack('>H', material_index[poly['mat']]))
        ptag_surf_payload = b''.join(ptag_surf_parts)

        # TAGS is NOT here: it is file-global and is emitted once, ahead
        # of every layer, because a PTAG SURF index names a position in
        # that single list.
        _layer = [
            _lwo_chunk(b'LAYR', layr_payload),
            _lwo_chunk(b'PNTS', pnts_payload),
        ] + pols_chunks + [
            _lwo_chunk(b'PTAG', ptag_surf_payload),
        ]

        if export_normals and smgp:
            ptag_smgp_parts = [b'SMGP']
            for pi in range(len(polys)):
                ptag_smgp_parts.append(_lwo_write_vx(pi) + struct.pack('>H', smgp[pi]))
            _layer.append(_lwo_chunk(b'PTAG', b''.join(ptag_smgp_parts)))

        vmad_uv_parts = [b'TXUV' + struct.pack('>H', 2) + _lwo_write_cstr("UVMap")]
        for pi, poly in enumerate(polys):
            for pt_idx, uv in zip(reversed(poly['points']), reversed(poly['uv'])):
                vmad_uv_parts.append(_lwo_write_vx(pt_idx) + _lwo_write_vx(pi)
                                     + struct.pack('>ff', uv[0], uv[1]))
        _layer.append(_lwo_chunk(b'VMAD', b''.join(vmad_uv_parts)))

        # Any further named UV / colour map. The engine reads exactly one of each
        # (idDrawVert has a single idVec2 st and a single byte color[4]) and would
        # let whichever chunk covers a corner LAST win, so these are only
        # meaningful for a non-idTech4 round trip -- hence off by default.
        if all_layers:
            extra_uv = []
            for poly in polys:
                for nm in (poly.get('uvs') or {}):
                    if nm not in extra_uv:
                        extra_uv.append(nm)
            for nm in extra_uv:
                parts = [b'TXUV' + struct.pack('>H', 2) + _lwo_write_cstr(nm)]
                for pi, poly in enumerate(polys):
                    vals = (poly.get('uvs') or {}).get(nm)
                    if not vals:
                        continue
                    for pt_idx, uv in zip(reversed(poly['points']), reversed(vals)):
                        parts.append(_lwo_write_vx(pt_idx) + _lwo_write_vx(pi)
                                     + struct.pack('>ff', uv[0], uv[1]))
                _layer.append(_lwo_chunk(b'VMAD', b''.join(parts)))

            # PTAG PART: the file's own polygon part tags.
            if any(p.get('part') for p in polys):
                parts = [b'PART']
                for pi, poly in enumerate(polys):
                    parts.append(_lwo_write_vx(pi)
                                 + struct.pack('>H', int(poly.get('part', 0)) & 0xFFFF))
                _layer.append(_lwo_chunk(b'PTAG', b''.join(parts)))

            # Per-point maps. The engine reads none of these -- it only ever looks
            # at TXUV and RGBA -- so they exist purely so a non-idTech4 model can
            # come back out the way it went in.
            for (kind, nm), table in point_maps.items():
                if not table:
                    continue
                dim = {'WGHT': 1, 'PICK': 0, 'MORF': 3}[kind]
                parts = [kind.encode('latin1') + struct.pack('>H', dim) + _lwo_write_cstr(nm)]
                for idx in sorted(table):
                    v = table[idx]
                    parts.append(_lwo_write_vx(idx))
                    if kind == 'WGHT':
                        parts.append(struct.pack('>f', v[0]))
                    elif kind == 'MORF':
                        # back to raw LightWave axis order, matching PNTS
                        parts.append(struct.pack('>fff', v[0], v[2], v[1]))
                _layer.append(_lwo_chunk(b'VMAP', b''.join(parts)))

            extra_col = []
            for poly in polys:
                for nm in (poly.get('cols') or {}):
                    if nm not in extra_col:
                        extra_col.append(nm)
            for nm in extra_col:
                parts = [b'RGBA' + struct.pack('>H', 4) + _lwo_write_cstr(nm)]
                for pi, poly in enumerate(polys):
                    vals = (poly.get('cols') or {}).get(nm)
                    if not vals:
                        continue
                    for pt_idx, c in zip(reversed(poly['points']), reversed(vals)):
                        r, g, b, a = (tuple(c) + (1.0,))[:4]
                        parts.append(_lwo_write_vx(pt_idx) + _lwo_write_vx(pi)
                                     + struct.pack('>ffff', r, g, b, a))
                _layer.append(_lwo_chunk(b'VMAD', b''.join(parts)))

        if has_any_color:
            vmad_col_parts = [b'RGBA' + struct.pack('>H', 4) + _lwo_write_cstr("Col")]
            for pi, poly in enumerate(polys):
                for pt_idx, col in zip(reversed(poly['points']), reversed(poly['col'])):
                    r, g, b, a = col if col is not None else (1.0, 1.0, 1.0, 1.0)
                    vmad_col_parts.append(_lwo_write_vx(pt_idx) + _lwo_write_vx(pi)
                                          + struct.pack('>ffff', r, g, b, a))
            _layer.append(_lwo_chunk(b'VMAD', b''.join(vmad_col_parts)))

        layer_chunks.extend(_layer)

    if not any_geometry:
        return None, [], ["No exportable mesh geometry found (no "
                          "triangulated faces on any selected object)."]
    if any_uv_missing:
        warnings.append("One or more objects have no UV map — those "
                        "faces will import with (0,0) texture coordinates.")

    # The name in TAGS/SURF is handed straight to declManager->FindMaterial by
    # ConvertLWOToModelSurfaces — no OSPathToRelativePath pass — so unlike the
    # .ase side there is nothing to wrap and nothing that can rescue an
    # absolute authoring path. material_source_string checks each candidate
    # against engine_lwo_surf_decl and falls back to the bare decl, which is
    # what makes an .ase-sourced material (whose idtech4_raw is a 3ds Max OS
    # path) come out as a name the engine can actually find.
    surf_name = {m: material_source_string(material_source.get(m), m,
                                           engine_lwo_surf_decl)
                 for m in material_order}

    # ---- smoothing groups — see this function's own docstring ----
    tags_payload = b''.join(_lwo_write_cstr(surf_name[n]) for n in material_order)
    chunks = [_lwo_chunk(b'TAGS', tags_payload)] + layer_chunks

    for mat_name in material_order:
        # The surface's own colour, not a hardcoded grey -- lwDefaultSurface's
        # 0.78431 is only the fallback for a surface with no COLR at all.
        rgb = surf_colr.get(mat_name) or (0.78431, 0.78431, 0.78431)
        colr_sub = struct.pack('>fff', rgb[0], rgb[1], rgb[2]) + b'\x00\x00'
        if mat_name in surf_sman_out:
            sman_angle = surf_sman_out[mat_name]
        else:
            sman_angle = (math.radians(179.9)
                          if (export_normals and surf_has_smooth[mat_name]) else 0.0)
        surf_payload = b''.join([
            _lwo_write_cstr(surf_name[mat_name]), _lwo_write_cstr(""),
            _lwo_subchunk(b'COLR', colr_sub),
            _lwo_subchunk(b'SMAN', struct.pack('>f', sman_angle)),
            surf_extra.get(mat_name) or b'',
        ])
        chunks.append(_lwo_chunk(b'SURF', surf_payload))

    # Escrowed chunks can reference point and polygon indices, so they may
    # only be re-emitted when this export renumbered nothing: no
    # triangulation, no welding, one object, and the same counts the escrow
    # was captured against.
    if escrow_blobs:
        # `points`/`polys` below are the LAST layer's, since they are now
        # rebuilt inside the layer loop - so a multi-layer file could
        # otherwise validate one layer's escrow against another's counts.
        # Escrowed chunks carry point and polygon indices, so they are
        # only ever re-emitted for a single-layer, single-object export
        # that renumbered nothing.
        safe = (all_layers and not triangulate and not weld_points
                and len(escrow_blobs) == 1 and len(layer_groups) == 1)
        name, blob, shape, nverts, npolys = escrow_blobs[0]
        if safe and list(shape) == [nverts, npolys] and len(points) == nverts:
            try:
                chunks.append(base64.b64decode(blob))
            except Exception as exc:
                warnings.append("Could not restore preserved chunks for "
                                "\"%s\": %s" % (name, exc))
        else:
            why = []
            if not all_layers:
                why.append("All Vertex Maps is off")
            if triangulate:
                why.append("Triangulate is on")
            if weld_points:
                why.append("Weld Coincident Points is on")
            if len(escrow_blobs) > 1:
                why.append("more than one object is being merged")
            if len(layer_groups) > 1:
                why.append("the file is being written with several layers")
            if list(shape) != [nverts, npolys] or len(points) != nverts:
                why.append("the geometry was edited since import")
            warnings.append(
                "\"%s\" carried chunks this importer does not model (envelopes, "
                "image clips, texture blocks and the like). They were NOT "
                "written back, because %s and they can reference point or "
                "polygon indices that would no longer line up."
                % (name, " and ".join(why)))

    body = b'LWO2' + b''.join(chunks)
    data = b'FORM' + struct.pack('>I', len(body)) + body
    return data, material_order, warnings


class EXPORT_OT_idtech4_lwo(bpy.types.Operator, ExportHelper, ExportTransformMixin):
    """Export selected mesh objects as an idTech4 .lwo (LWO2) static mesh"""
    bl_idname  = "export_scene.idtech4_lwo"
    bl_label   = "Export idTech4 .lwo"
    bl_options = {'PRESET'}

    filename_ext = ".lwo"
    filter_glob: StringProperty(default="*.lwo", options={'HIDDEN'}, maxlen=255)

    merge_objects: BoolProperty(
        name="Merge Into Single Layer",
        description="Combine every exported object into ONE LWO2 layer in "
                    "the chosen file — matching real idTech4/Doom 3 engine "
                    "behavior: verified against the GPL source "
                    "(idRenderModelStatic::ConvertLWOToModelSurfaces), the "
                    "engine only ever reads a .lwo's FIRST layer. Disable to "
                    "write ONE file still, but with one LAYER per object, "
                    "the way a hand-modeled multi-part LightWave object is "
                    "built — good for a faithful round trip or for further "
                    "authoring in LightWave, but be aware the engine will "
                    "render ONLY the first layer and silently ignore every "
                    "other one",
        default=True,
    )
    use_smoothing_groups: BoolProperty(
        name="Smoothing Groups (SURF SMAN / PTAG SMGP)",
        description="Reproduce each triangle's flat/smooth shading using "
                    "LWO's own SURF SMAN smoothing-angle + PTAG SMGP "
                    "smoothing-group mechanism — the ONLY mechanism the "
                    "real engine ever uses to shade a .lwo (verified "
                    "against the GPL source, Model_lwo.cpp: a .lwo never "
                    "stores per-vertex normals directly, the engine always "
                    "recomputes them from this data at load time). Exactly "
                    "reproduces a flat/smooth per-face toggle; canNOT "
                    "reproduce an arbitrary custom split normal or a "
                    "numeric Smooth-by-Angle threshold — export .ase "
                    "instead when exact per-corner normal fidelity matters "
                    "more than authoring in LightWave. Disable to force "
                    "every face fully faceted (SMAN 0) instead",
        default=True,
    )
    export_colors: BoolProperty(
        name="Export Vertex Colors ( Face Corner )",
        description="Export Blender Vertex Paint Face-Corner attributes",
        default=True,
    )
    export_vertex_domain_colors: BoolProperty(
        name="Export Vertex Colors ( Vertex )",
        description="Blender Vertex Paint Vertex Color attributes will be "
                    "converted to Face Corner and export",
        default=False,
    )
    preserve_smoothing: BoolProperty(
        name="Preserve File Smoothing",
        description="Write back the SURF SMAN angle and PTAG SMGP groups the "
                    "model was imported with, so the engine recomputes exactly "
                    "the shading it had. Turn it OFF to publish your Blender "
                    "flat/smooth edits instead — but note that is all a .lwo "
                    "can carry: SMAN plus SMGP reproduce a per-face toggle and "
                    "nothing finer, never an arbitrary custom split normal. "
                    "Ignored for a mesh with no imported smoothing data, which "
                    "always uses the flat/smooth flags",
        default=True,
    )
    all_layers: BoolProperty(
        name="All Vertex Maps",
        description="Write every UV map, corner colour attribute, vertex "
                    "group, selection set and shape key the mesh carries — "
                    "as named TXUV/RGBA/WGHT/PICK/MORF chunks — plus polygon "
                    "part tags. The engine reads NONE of this beyond one UV "
                    "and one colour: idDrawVert carries a single idVec2 st "
                    "and a single byte color[4], and "
                    "ConvertLWOToModelSurfaces flattens every TXUV chunk "
                    "into one table where the last one covering a corner "
                    "wins. So these are dead weight in-game and matter only "
                    "for a faithful non-idTech4 round trip. Off writes just "
                    "the active UV and colour map",
        default=False,
    )
    triangulate: BoolProperty(
        name="Triangulate",
        description="Split every polygon into triangles. Required for "
                    "idTech4: the engine renders only already-triangulated "
                    "polygons and drops anything else outright with "
                    "\"make sure you triplet it down\" "
                    "(ConvertLWOToModelSurfaces). Turn it OFF to write "
                    "quads/ngons as-is — .lwo carries polygons up to 1023 "
                    "vertices, so that is what a faithful round trip of a "
                    "non-idTech4 model needs, but the result will not render "
                    "in-game",
        default=True,
    )
    weld_points: BoolProperty(
        name="Weld Coincident Points",
        description="Merge points that round to the same position — across "
                    "every exported object — into one shared PNTS entry, so "
                    "touching pieces weld the way a single hand-modeled "
                    "multi-part LWO layer would. UV and vertex color are "
                    "unaffected (both are written per (polygon, point) as "
                    "VMAD, so welded points still carry different values at "
                    "different corners), but GEOMETRY is not: welding can "
                    "collapse a triangle into a degenerate or a duplicate and "
                    "nothing here detects that. Off by default because the "
                    "engine never welds a .lwo on load — on the fastLoad path "
                    "used for a renderbump high poly the vertex remap is the "
                    "identity, and R_CleanupTriangles leaves "
                    "R_RemoveDuplicatedTriangles commented out on purpose — "
                    "so leaving this off round-trips this addon's own "
                    "lossless import unchanged",
        default=False,
    )

    def invoke(self, context, event):
        name = self._target_export_name(context)
        if name:
            directory = os.path.dirname(self.filepath) if self.filepath \
                else os.path.dirname(context.blend_data.filepath)
            self.filepath = os.path.join(directory, name + self.filename_ext)
        return ExportHelper.invoke(self, context, event)

    def draw(self, context):
        self.draw_scope(self.layout, context, 'LWO')
        self.layout.prop(self, "apply_modifiers")
        self.draw_transforms(self.layout)
        self.layout.prop(self, "merge_objects")
        self.layout.prop(self, "use_smoothing_groups")
        self.layout.prop(self, "preserve_smoothing")
        self.layout.prop(self, "export_colors")
        self.layout.prop(self, "export_vertex_domain_colors")
        self.layout.prop(self, "triangulate")
        self.layout.prop(self, "all_layers")
        self.layout.prop(self, "weld_points")

    @exports_in_object_mode
    def execute(self, context):
        objects = self._scope_objects(context)
        if not objects:
            considered = iter_export_objects(context, self.export_scope, None,
                                             False)
            if considered and self.skip_hidden:
                self.report({'ERROR'}, "Every mesh in scope (%d) is hidden in "
                                       "the outliner." % len(considered))
            else:
                self.report({'ERROR'}, "No mesh objects in scope to export.")
            return {'CANCELLED'}
        depsgraph = context.evaluated_depsgraph_get()
        rot_matrix = self.get_rotation_matrix()
        scale = self.get_scale()

        # One file either way now. Merge on = every object in one layer
        # (what the engine reads); merge off = one layer per object in the
        # same file, which is how a multi-part LightWave object is built.
        layer_groups = [list(objects)] if self.merge_objects else [[o] for o in objects]

        written = 0
        all_warnings = []
        for grp in [objects]:
            data, materials, warnings = build_lwo_export_data(
                grp, depsgraph, rot_matrix, scale,
                export_normals=self.use_smoothing_groups,
                export_colors=self.export_colors,
                vertex_domain_colors=self.export_vertex_domain_colors,
                apply_modifiers=self.apply_modifiers,
                weld_points=self.weld_points,
                triangulate=self.triangulate,
                all_layers=self.all_layers,
                preserve_smoothing=self.preserve_smoothing,
                layer_groups=layer_groups)
            all_warnings.extend(warnings)
            if data is None:
                continue

            out_path = self.filepath
            try:
                with open(out_path, 'wb') as fh:
                    fh.write(data)
            except OSError as exc:
                self.report({'ERROR'}, f"Could not write \"{out_path}\": {exc}")
                return {'CANCELLED'}
            written += 1

        if written == 0:
            self.report({'ERROR'}, "; ".join(all_warnings) or "Nothing to export.")
            return {'CANCELLED'}

        self.report({'INFO'},
                    "Exported %d object(s) as %d layer(s) to \"%s\""
                    % (len(objects), len(layer_groups), self.filepath))
        for w in all_warnings:
            self.report({'WARNING'}, w)
        return {'FINISHED'}


# ─────────────────────────────────────────────────────────────────────
#  SOURCES GATE
# ─────────────────────────────────────────────────────────────────────
# File > Import points straight at each real import operator now —
# unlike the old design, none of the gates below run in front of the
# menu entry. Each only fires from INSIDE its own operator's execute(),
# and only once the file AND every checkbox are already chosen:
# MaterialGenMixin._needs_sources_gate() checks whether Auto-Generate
# Materials is on, its companion addon is installed, and Base Directory/
# Materials Source aren't already resolvable — only then does execute()
# launch the matching gate (via _launch_sources_gate). That means the
# popup below no longer needs its own "is it even needed" check — by
# the time it's shown, that's already been decided.
#
# self.filepath plus every one of the real operator's own already-set
# properties are handed off via _pending_import_kwargs (a plain module
# global — simpler than mirroring ~10 properties onto each gate class
# just to receive them). _launch_target merges in whatever this popup
# resolved and re-invokes the real operator via EXEC_DEFAULT (filepath
# is already known, so there's no reason to reopen the file browser),
# with gate_resolved=True so _needs_sources_gate never fires the same
# gate twice for one import — that matters because Derive Automatically
# can still come back empty (no "materials" folder found walking up
# from the file), and Select Now proceeds with whatever was typed; both
# are meant to move forward, not loop.

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
    """Shared popup + dispatch logic for every gate operator below. Only
    ever shows its popup when the shared idTech4 config file's
    base_directory/source_path (get_shared_paths) aren't BOTH set —
    otherwise it's a silent, instant pass-through to the real import
    operator."""

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


class IMPORT_OT_idtech4_ase_gate(bpy.types.Operator, MaterialsSourceGateMixin):
    """Resolve Base Directory / Materials Source for a .ase import already
    in progress (file and options already chosen), then re-launch it"""
    bl_idname  = "import_scene.idtech4_ase_gate"
    bl_label   = ".ase Import : Base Directory / Materials Source Setup"
    bl_options = {'INTERNAL'}

    filepath: StringProperty(subtype='FILE_PATH', options={'HIDDEN', 'SKIP_SAVE'})

    def _target_op(self):
        return bpy.ops.import_scene.idtech4_ase


class IMPORT_OT_idtech4_lwo_gate(bpy.types.Operator, MaterialsSourceGateMixin):
    """Resolve Base Directory / Materials Source for a .lwo import already
    in progress (file and options already chosen), then re-launch it"""
    bl_idname  = "import_scene.idtech4_lwo_gate"
    bl_label   = ".lwo Import : Base Directory / Materials Source Setup"
    bl_options = {'INTERNAL'}

    filepath: StringProperty(subtype='FILE_PATH', options={'HIDDEN', 'SKIP_SAVE'})

    def _target_op(self):
        return bpy.ops.import_scene.idtech4_lwo


# ─────────────────────────────────────────────────────────────────────
#  MENU + REGISTER
# ─────────────────────────────────────────────────────────────────────

def menu_func_import_ase(self, context):
    self.layout.operator(IMPORT_OT_idtech4_ase.bl_idname,
                         text="idTech4 ASE (.ase, .base) (static mesh)", icon='MESH_DATA')


def menu_func_import_lwo(self, context):
    self.layout.operator(IMPORT_OT_idtech4_lwo.bl_idname,
                         text="idTech4 LWO (.lwo, .blwo) (static mesh)", icon='MESH_DATA')


def menu_func_export_ase(self, context):
    self.layout.operator(EXPORT_OT_idtech4_ase.bl_idname,
                         text="idTech4 .ase (static mesh)", icon='MESH_DATA')


def menu_func_export_lwo(self, context):
    self.layout.operator(EXPORT_OT_idtech4_lwo.bl_idname,
                         text="idTech4 .lwo (static mesh)", icon='MESH_DATA')




class IDTECH4_OT_validate_model(Operator):
    """Check for non-triangle polygons, zero-area faces, polygons over 1023
    vertices, missing UV maps, degenerate UV triangles, material names the
    engine cannot resolve, extra UV maps, custom split normals,
    subdivision-tagged faces, out-of-range idTech4 face attributes and a
    non-identity *NODE_TM. Changes nothing; the full report goes to the
    system console"""
    bl_idname = "idtech4.validate_model"
    bl_label = "Check Model for idTech4"
    bl_options = {'REGISTER'}

    @exports_in_object_mode
    def execute(self, context):
        props = context.scene.idtech4_export_tools
        objects = iter_export_objects(context, props.scope, None,
                                      props.skip_hidden)
        if not objects:
            self.report({'WARNING'}, "Nothing in scope to check.")
            return {'CANCELLED'}

        dg = context.evaluated_depsgraph_get()
        findings = validate_for_idtech4(objects, props.target, dg)
        man = export_manifest(context, props.scope, None, props.skip_hidden,
                              'LWO' if props.target != 'ASE' else 'ASE')

        lines = ["", "idTech4 model check (%s)" % props.target,
                 "  %d object(s), %d triangle(s) after triangulation"
                 % (len(man['objects']), man['triangles'])]
        for o in man['objects']:
            lines.append("    %-32s %5d polygons" % (o.name, len(o.data.polygons)))
        if man['hidden']:
            lines.append("  skipped, hidden in the outliner: %s"
                         % ", ".join(o.name for o in man['hidden']))
        if man['excluded']:
            lines.append("  skipped, flagged idtech4_exclude: %s"
                         % ", ".join(o.name for o in man['excluded']))
        if not findings:
            lines.append("  no problems found")
        for tier, label in ((TIER_MANUAL, "needs a decision from you"),
                            (TIER_PREPARE, "Prepare for Export can fix"),
                            (TIER_AUTO, "the exporter handles, for information")):
            group = [f for f in findings if f['tier'] == tier]
            if not group:
                continue
            lines.append("  %s:" % label)
            for f in group:
                tag = '' if f['target'] == 'BOTH' else ' [%s]' % f['target']
                lines.append("    - %s%s" % (f['message'], tag))
                if f['objects']:
                    lines.append("      %s" % ", ".join(f['objects'][:8])
                                 + (" ..." if len(f['objects']) > 8 else ""))
        text = "\n".join(lines)
        print(text)
        props.last_report = text
        props.last_manifest = manifest_snapshot(man, props.scope, props.target,
                                                props.skip_hidden)

        n_manual = sum(1 for f in findings if f['tier'] == TIER_MANUAL)
        n_prep = sum(1 for f in findings if f['tier'] == TIER_PREPARE)
        if n_manual:
            self.report({'WARNING'}, "%d issue(s) need a decision, %d fixable "
                                     "by Prepare — see the console"
                        % (n_manual, n_prep))
        elif n_prep:
            self.report({'INFO'}, "%d issue(s) Prepare for Export can fix" % n_prep)
        else:
            self.report({'INFO'}, "No problems found in %d object(s)" % len(objects))
        return {'FINISHED'}


class IDTECH4_OT_prepare_export(Operator):
    """Triangulate every non-triangle face and remove zero-area faces"""
    bl_idname = "idtech4.prepare_export"
    bl_label = "Prepare for Export"
    bl_options = {'REGISTER', 'UNDO'}

    # The three settings live on the scene PropertyGroup, not here, so the
    # panel can draw them UNDER the button and they are chosen BEFORE the
    # run. As operator properties they surfaced only in Adjust Last
    # Operation, i.e. after the fact - and "Work on a Copy" defaulting on
    # meant the first click always left a duplicate collection behind
    # before you could see there was a choice.

    @exports_in_object_mode
    def execute(self, context):
        props = context.scene.idtech4_export_tools
        objects = iter_export_objects(context, props.scope, None,
                                      props.skip_hidden)
        if not objects:
            self.report({'WARNING'}, "Nothing in scope to prepare.")
            return {'CANCELLED'}

        targets, coll = [], None
        if props.prepare_work_on_copy:
            base = objects[0].users_collection[0].name if objects[0].users_collection \
                else context.scene.collection.name
            coll = bpy.data.collections.new(base + "_idtech4")
            context.scene.collection.children.link(coll)
            for o in objects:
                dup = o.copy()
                dup.data = o.data.copy()
                coll.objects.link(dup)
                targets.append(dup)
        else:
            targets = list(objects)

        n_tri, n_deg = 0, 0
        for o in targets:
            bm = bmesh.new()
            bm.from_mesh(o.data)
            if props.prepare_degenerate:
                dead = [f for f in bm.faces if f.calc_area() <= 1e-12]
                if dead:
                    n_deg += len(dead)
                    bmesh.ops.delete(bm, geom=dead, context='FACES')
            if props.prepare_triangulate:
                quads = [f for f in bm.faces if len(f.verts) != 3]
                if quads:
                    n_tri += len(quads)
                    bmesh.ops.triangulate(bm, faces=quads)
            bm.to_mesh(o.data)
            bm.free()
            o.data.update()

        where = "copy in \"%s\"" % coll.name if coll else "the originals"
        self.report({'INFO'}, "Prepared %d object(s) in %s: %d polygon(s) "
                              "triangulated, %d zero-area face(s) removed"
                    % (len(targets), where, n_tri, n_deg))
        return {'FINISHED'}


class IDTECH4_ExportToolProps(PropertyGroup):
    scope: EnumProperty(name="Scope", items=EXPORT_SCOPE_ITEMS,
                        default='COLLECTION')
    target: EnumProperty(name="Target", items=VALIDATE_TARGET_ITEMS,
                         default='BOTH')
    skip_hidden: BoolProperty(
        name="Skip Hidden Objects",
        description="Leave out anything whose outliner EYE is closed. The "
                    "monitor and camera icons are deliberately not consulted "
                    "— one obvious control decides what reaches the file, and "
                    "the report lists exactly what was skipped",
        default=True)

    # Prepare for Export's settings. Here rather than on the operator so the
    # panel draws them under the button and they are chosen before the run,
    # not discovered afterwards in Adjust Last Operation.
    prepare_work_on_copy: BoolProperty(
        name="Work on a Copy",
        description="Duplicate everything in scope into a new "
                    "\"<collection>_idtech4\" collection and prepare THAT, "
                    "leaving the authoring meshes untouched, so the "
                    "export-ready model sits beside the cage you keep "
                    "modelling on. Off prepares the originals in place. "
                    "Either way the run is undoable",
        default=False)
    prepare_triangulate: BoolProperty(
        name="Triangulate",
        description="Split every polygon into triangles. The engine drops "
                    "anything that is not one",
        default=True)
    prepare_degenerate: BoolProperty(
        name="Remove Zero-Area Faces",
        description="Delete faces the engine would discard at load anyway "
                    "(R_RemoveDegenerateTriangles). Neither exporter does "
                    "this, so it is the one fix only this button applies",
        default=True)

    last_report: StringProperty(default="")
    # What Check Model last found, as JSON. The box below the controls used to
    # call export_manifest() straight from draw(), and COLLECTION scope with no
    # explicit collection resolves through context.active_object - so clicking
    # anything in the viewport silently repointed the summary at whatever
    # collection that object happened to live in, with no run behind it. It is
    # a snapshot now, written only by Check Model.
    last_manifest: StringProperty(default="")


class IDTECH4_PT_export_tools(Panel):
    bl_label = "Export Check"
    bl_idname = "IDTECH4_PT_export_tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'idTech4'
    # Below the shared Sources panel. Panels in a category sort by bl_order
    # first and registration order second, and Sources is registered LAST
    # (register() calls _register_shared_ui() after its own classes), so
    # without this it would draw underneath. Sources cannot carry an order of
    # its own: every idTech4 addon ships an identical copy of that class and
    # whichever registers first is the one that sticks, so the copies have to
    # stay byte-identical.
    bl_order = 10

    def draw(self, context):
        layout = self.layout
        props = context.scene.idtech4_export_tools
        col = layout.column(align=True)
        col.prop(props, "scope")
        col.prop(props, "target")
        col.prop(props, "skip_hidden")

        self._draw_manifest(layout, props)

        layout.operator(IDTECH4_OT_validate_model.bl_idname, icon='CHECKMARK')
        layout.operator(IDTECH4_OT_prepare_export.bl_idname, icon='MODIFIER')
        # Prepare's settings, under its button and read by execute() from the
        # scene: chosen before the run rather than found afterwards in Adjust
        # Last Operation. Indented so it reads as belonging to the button
        # above rather than to the panel as a whole.
        sub = layout.column(align=True)
        sub.use_property_split = False
        row = sub.row()
        row.separator()
        opts = row.column(align=True)
        opts.prop(props, "prepare_work_on_copy")
        opts.prop(props, "prepare_triangulate")
        opts.prop(props, "prepare_degenerate")

    def _draw_manifest(self, layout, props):
        """What the last Check Model run said would be exported.

        A snapshot, not a live count: this used to run export_manifest() on
        every redraw, which re-resolved COLLECTION scope through the active
        object and so followed the selection around the viewport. Check Model
        resolves the scope exactly the way the exporter does - the same
        iter_export_objects() call, the same "the active object's first
        collection" fallback - and this redraws what that run found.
        """
        box = layout.box()
        snap = load_manifest_snapshot(props.last_manifest)
        if snap is None:
            box.label(text="Nothing checked yet.", icon='INFO')
            box.label(text="Press Check Model for idTech4.")
            return
        objects = snap.get('objects', [])
        if not objects:
            sub = box.column()
            sub.alert = True
            sub.label(text="Nothing in scope", icon='ERROR')
        else:
            box.label(text="%d object(s), %d triangle(s)"
                           % (len(objects), snap.get('triangles', 0)),
                      icon='OUTLINER_OB_MESH')
            if snap.get('ngons'):
                box.label(text="%d non-triangle polygon(s)" % snap['ngons'],
                          icon='ERROR')
            if snap.get('merged'):
                box.label(text="merged into one .lwo layer", icon='INFO')
            for name in objects[:8]:
                box.label(text="    " + name)
            if len(objects) > 8:
                box.label(text="    ... and %d more" % (len(objects) - 8))
        hidden = snap.get('hidden', [])
        if hidden:
            sub = box.column()
            sub.alert = True
            sub.label(text="%d hidden, will be skipped" % len(hidden),
                      icon='HIDE_ON')
            for name in hidden[:4]:
                sub.label(text="    " + name)
        # The snapshot is only as true as the controls it was taken under, and
        # those sit directly above it - saying so beats a box that quietly
        # describes a run nobody asked for any more.
        if (snap.get('scope'), snap.get('target'),
                snap.get('skip_hidden')) != (props.scope, props.target,
                                             props.skip_hidden):
            sub = box.column()
            sub.alert = True
            sub.label(text="settings changed since this check", icon='ERROR')


classes = (
    IMPORT_OT_idtech4_ase,
    IMPORT_OT_idtech4_lwo,
    IMPORT_OT_idtech4_ase_gate,
    IMPORT_OT_idtech4_lwo_gate,
    EXPORT_OT_idtech4_ase,
    EXPORT_OT_idtech4_lwo,
    IDTECH4_ExportToolProps,
    IDTECH4_OT_validate_model,
    IDTECH4_OT_prepare_export,
    IDTECH4_PT_export_tools,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import_ase)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import_lwo)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export_ase)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export_lwo)
    bpy.types.Scene.idtech4_export_tools = PointerProperty(
        type=IDTECH4_ExportToolProps)
    _register_shared_ui()


def unregister():
    _unregister_shared_ui()
    if hasattr(bpy.types.Scene, 'idtech4_export_tools'):
        del bpy.types.Scene.idtech4_export_tools
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export_lwo)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export_ase)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import_lwo)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import_ase)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
