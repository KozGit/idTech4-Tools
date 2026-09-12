"""
idTech 4 Materials Addon for Blender 4.5+
Generates Blender materials from idTech 4 .mtr files - Doom 3, Doom 3 BFG,
Quake 4, Prey and The Dark Mod, from one merged keyword table.

Four fidelity modes crossed with three parameter policies. The modes exist to
make a whole map renderable on a card that is not an RTX 4090; every rung down
the ladder is justified by a measurement in tests/tier_bench.py, not by taste.
"""

# Version: 1.0.0

bl_info = {
    "name": "idTech4 Materials",
    "author": "Samson & Claude",
    "version": (1, 0, 0),
    "blender": (4, 5, 0),
    "location": "3D Viewport > N-Panel > idTech4 > Sources; idTech4 Mtr > Materials",
    "description": "Create Blender materials from idTech 4 .mtr files (Doom 3, BFG, Quake 4, Prey, The Dark Mod)",
    "category": "Material",
}

import bpy
import os
import re
import json
import math
import hashlib
import random
import numpy as np
from bpy.props import (StringProperty, EnumProperty, FloatProperty,
                       BoolProperty, IntProperty, FloatVectorProperty,
                       CollectionProperty, PointerProperty)
from bpy.types import Panel, Operator, PropertyGroup

# Blender 5.1 added an explicit OpenGL/DirectX `convention` switch to the
# Normal Map node (bpy.types.ShaderNodeNormalMap.convention). Before 5.1 the
# node had no such property and always behaved as OpenGL-convention, which
# is why older versions need the manual green-channel-invert node setup to
# convert idTech4's DirectX-convention normal maps.
_NORMAL_MAP_HAS_CONVENTION = bpy.app.version >= (5, 1, 0)

ROUGHNESS_GROUP_NAME = 'idtech4_EstimateRoughness'
PLACEHOLDER_MIX_NAME = 'IDTECH4_PlaceholderMix'


# ---------------------------------------------------------------------------
# The block below is sliced out of this file by tests/mtr_corpus.py, which
# rewinds from the BEGIN banner to the nearest preceding `# ---` comment rule
# and exec()s everything from there to the END banner under plain CPython.
#
# So: this rule must remain the LAST `# ---` line above the banner, and
# nothing between it and the END banner may touch bpy.
# ---------------------------------------------------------------------------


# ===========================================================================
# BEGIN MTR PARSER
#
# An engine-faithful parser for idTech 4 .mtr material declarations, covering
# Doom 3, Doom 3 BFG, Quake 4 (including its `guide` macro layer), Prey and
# The Dark Mod from one merged keyword table - no per-game mode switch.
#
# NOTHING BETWEEN THE BEGIN/END BANNERS MAY TOUCH bpy.
#
# That restriction is load-bearing, not stylistic: tests/mtr_corpus.py slices
# this block straight out of this file by its banner comments and exec()s it
# under plain CPython, so the whole parser can be regression-tested against
# the shipping .mtr corpus without launching Blender. A stray bpy reference
# turns that into a NameError. Keep imports here to the standard library.
#
# The structure mirrors the engine so the two can be diffed by eye:
#
#   _lex()              idlib/Lexer.cpp        idLexer::ReadToken
#   _scan_decls()       framework/DeclManager  idDeclFile::LoadAndParse
#   _expand_guides()    framework/DeclManager  idDeclFile::PreprocessGuides (Q4)
#   _parse_image()      renderer/Image_program R_ParseImageProgram_r
#   _parse_expr()       renderer/Material.cpp  idMaterial::ParseExpression
#   _parse_stage()      renderer/Material.cpp  idMaterial::ParseStage
#   _parse_material()   renderer/Material.cpp  idMaterial::ParseMaterial
#   MtrMaterial.finish  renderer/Material.cpp  idMaterial::Parse (post-pass)
# ===========================================================================

import os as _os
import re as _re
import math as _math
import bisect as _bisect


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
# Every departure from "we understood this and built it properly" produces one
# of these. The addon's post-generation report is just a grouping of them, and
# the corpus test asserts their count does not grow. Nothing is ever dropped
# on the floor silently - that is the whole point of the rewrite.

DIAG_APPROXIMATED = 'approximated'   # built, but not exactly as the engine draws it
DIAG_UNSUPPORTED  = 'unsupported'    # understood, not representable - fallback used
DIAG_UNKNOWN_KW   = 'unknown'        # keyword we do not know; line skipped
DIAG_PARSE_ERROR  = 'error'          # structurally broken; engine would MakeDefault()
DIAG_CONTENT      = 'content'        # the .mtr itself is wrong (engine warns too)

# Diagnostic kinds that name what they are about, so the grouped report can
# tell one from another. summarise_diagnostics() buckets by kind and prints
# only the first few of each, so a fixed slug turns every distinct table (or
# term) in a tree into one row nobody can act on.
TABLE_NOTE_PREFIX = 'undefined-table:'

# An undeclared material name that resolved to an image on disk. The engine
# generates a material from the name itself, so this is normal behaviour, not
# a defect - the diagnostic exists so the Mtr panel can still say which names
# it happened to, and is named here so a report can list those separately
# from the things that genuinely did not build as the .mtr asked.
IMPLICIT_DIAG_KIND = 'implicit image material'

_DIAG_ORDER = {DIAG_PARSE_ERROR: 0, DIAG_UNKNOWN_KW: 1, DIAG_UNSUPPORTED: 2,
               DIAG_APPROXIMATED: 3, DIAG_CONTENT: 4}


class MtrDiagnostic(object):
    __slots__ = ('level', 'kind', 'message', 'filename', 'line', 'material')

    def __init__(self, level, kind, message, filename='', line=0, material=''):
        self.level = level          # one of the DIAG_* constants
        self.kind = kind            # short slug for grouping, e.g. 'fragmentProgram'
        self.message = message
        self.filename = filename
        self.line = line
        self.material = material

    @property
    def sort_key(self):
        return (_DIAG_ORDER.get(self.level, 9), self.kind, self.filename, self.line)

    def location(self):
        """Where in the .mtr tree this came from, spelled out.

        "invisible.mtr: Line 36", not "invisible.mtr:36". This string is
        shown on its own - a row in the report panel, a line in the .map
        import report - with nothing around it to say what the number is,
        and a bare number after a filename reads as a version or a count
        just as readily as a line.

        Empty when there is no source position to give. Not every
        diagnostic has one: the ones raised about a material NAME in use by
        geometry rather than about a declaration (see
        MaterialBuildSummary.record_not_found / record_implicit) have no
        file and no line, and "?:0" is worse than saying nothing.
        """
        if not self.filename:
            return ''
        # The FULL path, not the basename. Two roots can each hold a
        # guis.mtr, and "guis.mtr: Line 412" then names neither of them;
        # even with one root a bare basename makes the reader go looking
        # for a file whose location they already had here.
        return '%s: Line %d' % (self.filename, self.line)

    def __repr__(self):
        return '<%s %s %s @%s>' % (self.level, self.kind, self.message,
                                   self.location() or '?')


# ---------------------------------------------------------------------------
# Lexer - idlib/Lexer.cpp with DECL_LEXER_FLAGS
# ---------------------------------------------------------------------------
# Flags the decl system sets (framework/DeclManager.h):
#   LEXFL_NOSTRINGCONCAT | LEXFL_NOSTRINGESCAPECHARS | LEXFL_ALLOWPATHNAMES
#   | LEXFL_ALLOWMULTICHARLITERALS | LEXFL_ALLOWBACKSLASHSTRINGCONCAT
#   | LEXFL_NOFATALERRORS
#
# The two that shape tokenisation: ALLOWPATHNAMES puts '/', '\', ':' and '.'
# into the name character class (but NOT '-', which stays a punctuation token,
# which is why `rotate time * -0.035` lexes as four tokens), and
# NOSTRINGESCAPECHARS makes quoted strings literal.

TT_STRING, TT_LITERAL, TT_NUMBER, TT_NAME, TT_PUNCT = 1, 2, 3, 4, 5

# idLexer's default_punctuations table, longest first so '>=' never lexes as
# '>' followed by '='. Only the entries that can appear in material text are
# listed; the rest would be a parse error in both the engine and here.
_PUNCT = ('>>=', '<<=', '...', '&&', '||', '>=', '<=', '==', '!=', '*=', '/=',
          '%=', '+=', '-=', '&=', '|=', '^=', '>>', '<<', '++', '--', '::',
          '->', '*', '/', '%', '+', '-', '<', '>', '=', '&', '|', '^', '~',
          '!', '(', ')', '{', '}', '[', ']', '.', ',', ';', ':', '?', '#',
          '\\', '$', '@')

# Source is decoded latin-1 so every byte maps to exactly one character, then
# any byte >= 0x80 is treated as whitespace. That is not a guess: the engine's
# idLexer::ReadWhiteSpace loops `while ( *script_p <= ' ' )` over a *signed*
# char, so on every platform id shipped, high bytes compare as negative and
# are skipped. It matters in practice - The Dark Mod's tdm_cubeLights.mtr is
# indented with 0xA0 (non-breaking space) bytes, which a UTF-8 decode turns
# into U+FFFD and glues onto the following token.
_TOKEN_RE = _re.compile(
    r'(?P<BLOCKCOMMENT>/\*.*?(?:\*/|\Z))'
    r'|(?P<LINECOMMENT>//[^\n]*)'
    r'|(?P<WS>[\x00-\x20\x80-\xff]+)'
    r'|(?P<STRING>"[^"]*"?)'
    r'|(?P<LITERAL>\'[^\']*\'?)'
    r'|(?P<NUMBER>0[xX][0-9a-fA-F]+|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)'
    r'|(?P<NAME>[A-Za-z_][A-Za-z0-9_/\\:.]*)'
    r'|(?P<PUNCT>' + '|'.join(_re.escape(p) for p in _PUNCT) + r')'
    r'|(?P<OTHER>.)',
    _re.DOTALL)


class MtrToken(object):
    __slots__ = ('type', 'val', 'line', 'first_on_line')

    def __init__(self, type_, val, line, first_on_line):
        self.type = type_
        self.val = val
        self.line = line
        self.first_on_line = first_on_line

    def __repr__(self):
        return '<%r@%d>' % (self.val, self.line)


def _lex(text):
    """Tokenise decoded .mtr source. Returns a list of MtrToken."""
    newlines = [m.start() for m in _re.finditer('\n', text)]
    out = []
    last_line = 0
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup
        if kind in ('BLOCKCOMMENT', 'LINECOMMENT', 'WS', 'OTHER'):
            continue
        line = _bisect.bisect_right(newlines, m.start()) + 1
        val = m.group()
        if kind == 'STRING':
            val = val[1:-1] if val.endswith('"') and len(val) > 1 else val[1:]
            ttype = TT_STRING
        elif kind == 'LITERAL':
            val = val[1:-1] if val.endswith("'") and len(val) > 1 else val[1:]
            ttype = TT_LITERAL
        elif kind == 'NUMBER':
            ttype = TT_NUMBER
        elif kind == 'NAME':
            ttype = TT_NAME
        else:
            ttype = TT_PUNCT
        out.append(MtrToken(ttype, val, line, line != last_line))
        last_line = line
    return _join_backslash_strings(out)


def _join_backslash_strings(toks):
    """idLexer::ReadString's LEXFL_ALLOWBACKSLASHSTRINGCONCAT branch.

    DECL_LEXER_FLAGS (framework/DeclManager.h:93) sets it, so a string, a
    lone '\\' and the next string are ONE token to the engine:

        "part one" \\
        "part two"

    _TOKEN_RE already emits the pieces the join needs - a lone backslash
    cannot start a NAME, so it arrives as its own PUNCT token - and it is
    the pairing, not the tokenising, that was missing. No shipping .mtr in
    the five corpora uses it (the .def files do, 105 times), so this fixes
    nothing visible today and the corpus baseline must not move; it is here
    because a material that used it would otherwise parse as three tokens
    where the engine sees one, and every keyword after it would shift.
    """
    if not any(t.type == TT_PUNCT and t.val == '\\' for t in toks):
        return toks
    out = []
    i, n = 0, len(toks)
    while i < n:
        tok = toks[i]
        if tok.type != TT_STRING:
            out.append(tok)
            i += 1
            continue
        j = i + 1
        while (j + 1 < n and toks[j].type == TT_PUNCT and toks[j].val == '\\'
               and toks[j + 1].type == TT_STRING):
            tok.val += toks[j + 1].val
            j += 2
        out.append(tok)
        i = j
    return out


def decode_mtr_bytes(raw):
    """Decode raw .mtr file bytes the way the engine reads them.

    latin-1 is not a guess about the file's real encoding - it is a
    byte-preserving decode that lets _lex() apply the engine's own
    "every high byte is whitespace" rule (see _TOKEN_RE) rather than
    letting a UTF-8 decoder invent replacement characters that then
    glue themselves onto adjacent tokens.
    """
    if raw[:3] == b'\xef\xbb\xbf':          # UTF-8 BOM, seen in a few mods
        raw = raw[3:]
    return raw.decode('latin-1')


class _Cursor(object):
    """Token cursor with the engine's read/unread/peek vocabulary."""
    __slots__ = ('toks', 'i', 'n')

    def __init__(self, toks, start=0, end=None):
        self.toks = toks
        self.i = start
        self.n = len(toks) if end is None else end

    def read(self):
        if self.i >= self.n:
            return None
        tk = self.toks[self.i]
        self.i += 1
        return tk

    def read_on_line(self):
        """idLexer::ReadTokenOnLine - returns None at a line break."""
        if self.i >= self.n:
            return None
        tk = self.toks[self.i]
        if tk.first_on_line:
            return None
        self.i += 1
        return tk

    def unread(self):
        if self.i > 0:
            self.i -= 1

    def peek(self):
        return self.toks[self.i] if self.i < self.n else None

    def skip_rest_of_line(self):
        while self.read_on_line() is not None:
            pass

    def at_end(self):
        return self.i >= self.n

    def line(self):
        tk = self.peek()
        if tk is not None:
            return tk.line
        return self.toks[self.i - 1].line if self.i else 0

    def skip_braced(self, expect_open=True):
        """idLexer::SkipBracedSection. Returns False on an unbalanced run.

        This is the error-recovery primitive the old parser had no equivalent
        of: because a decl's extent is found by counting braces rather than by
        parsing its contents, a malformed material can never swallow the ones
        that follow it.
        """
        depth = 0
        if expect_open:
            tk = self.read()
            if tk is None or tk.val != '{':
                return False
            depth = 1
        else:
            depth = 1
        while depth > 0:
            tk = self.read()
            if tk is None:
                return False
            if tk.val == '{':
                depth += 1
            elif tk.val == '}':
                depth -= 1
        return True


# ---------------------------------------------------------------------------
# Decl file scanning - framework/DeclManager.cpp, idDeclFile::LoadAndParse
# ---------------------------------------------------------------------------
# A .mtr file is a sequence of `[type] name { ... }` blocks whose default type
# is "material". Other decl types do legitimately turn up inside .mtr files -
# the shipping corpus contains 1244 `table` decls plus a handful of `skin` and
# `particle` ones - so the type keyword has to be recognised rather than
# assumed away.

_DECL_TYPES = frozenset((
    'material', 'table', 'skin', 'sound', 'entitydef', 'model', 'export',
    'modelexport', 'particle', 'fx', 'articulatedfigure', 'af', 'pda',
    'video', 'audio', 'email', 'lightdef', 'mapdef', 'moduledef',
))


class MtrDecl(object):
    """One `[type] name { ... }` block, located but not yet parsed."""
    __slots__ = ('type', 'name', 'start', 'end', 'line', 'filename')

    def __init__(self, type_, name, start, end, line, filename):
        self.type = type_
        self.name = name
        self.start = start      # token index just past the opening '{'
        self.end = end          # token index of the matching '}'
        self.line = line
        self.filename = filename

    def __repr__(self):
        return '<%s %s @%s:%d>' % (self.type, self.name,
                                   self.filename, self.line)


def _scan_decls(toks, filename, diags):
    """Locate every decl in a token stream. Mirrors idDeclFile::LoadAndParse,
    including its two recovery paths (a stray '{' and a missing '{')."""
    src = _Cursor(toks)
    out = []
    while True:
        tk = src.read()
        if tk is None:
            break

        if tk.val == '{':
            # "if we ever see an open brace, we somehow missed the prefix"
            diags.append(MtrDiagnostic(
                DIAG_PARSE_ERROR, 'missing-decl-name',
                'stray "{" with no decl name before it', filename, tk.line))
            src.unread()
            src.skip_braced()
            continue

        if tk.type in (TT_NAME, TT_STRING) and tk.val.lower() in _DECL_TYPES:
            dtype = tk.val.lower()
            name_tk = src.read()
        else:
            dtype = 'material'          # the default type for a .mtr file
            name_tk = tk

        if name_tk is None:
            diags.append(MtrDiagnostic(
                DIAG_PARSE_ERROR, 'truncated',
                'decl type with no definition at end of file', filename,
                tk.line))
            break

        if name_tk.val == '{':
            diags.append(MtrDiagnostic(
                DIAG_PARSE_ERROR, 'missing-decl-name',
                'expected a decl name, found "{"', filename, name_tk.line))
            src.unread()
            src.skip_braced()
            continue

        brace = src.read()
        if brace is None:
            diags.append(MtrDiagnostic(
                DIAG_PARSE_ERROR, 'truncated',
                "'%s' has no definition at end of file" % name_tk.val,
                filename, name_tk.line))
            break
        if brace.val != '{':
            # The engine warns and continues from here, which re-reads this
            # token as the next decl's type/name. Guide lines in Quake 4 take
            # this path when guide expansion has not run, so keep it quiet
            # for them and let _expand_guides() report anything it cannot do.
            if name_tk.val.lower() not in ('guide', 'inlineguide'):
                diags.append(MtrDiagnostic(
                    DIAG_PARSE_ERROR, 'expected-brace',
                    'expected "{" after "%s", found "%s"'
                    % (name_tk.val, brace.val), filename, brace.line))
            src.unread()
            continue

        start = src.i
        src.unread()
        if not src.skip_braced():
            diags.append(MtrDiagnostic(
                DIAG_PARSE_ERROR, 'unbalanced-braces',
                "'%s' is missing its closing brace" % name_tk.val,
                filename, name_tk.line))
            break
        out.append(MtrDecl(dtype, name_tk.val, start, src.i - 1,
                           name_tk.line, filename))
    return out


# ---------------------------------------------------------------------------
# Quake 4 guides - framework/DeclManager.cpp, idDeclManagerLocal::ParseGuides
# ---------------------------------------------------------------------------
# Quake 4 adds a text-substitution macro layer that runs before decl scanning.
# It matters: 1398 `guide` plus 140 `inlineGuide` invocations across Quake 4's
# materials, which is a bit over a quarter of the game's materials. Without
# expansion they are not materials at all - they are bare lines the decl
# scanner walks straight past.
#
#   base/guides/*.guide     guide <name>( Parm1, Parm2 ) { <body> }
#   base/materials/*.mtr    guide <matname> <guidename>( "arg1", "arg2" )
#
# Expansion is a literal idStr::Replace of each parameter *name* by its
# argument text over the body - substring substitution, not token
# substitution, which is why guide bodies write `textures/TextureParm_d` with
# the parameter glued into the middle of a path. `inlineGuide` is the same
# thing with the body's outer braces stripped, spliced in place inside a
# material body rather than emitted as a new decl.

class MtrGuide(object):
    __slots__ = ('name', 'parms', 'body', 'inline', 'filename', 'line')

    def __init__(self, name, parms, body, inline, filename, line):
        self.name = name
        self.parms = parms
        self.body = body
        self.inline = inline
        self.filename = filename
        self.line = line

    def expand(self, args):
        body = self.body
        for i, parm in enumerate(self.parms):
            if i < len(args):
                body = body.replace(parm, args[i])
        return body


def parse_guide_file(text, filename, diags):
    """Parse one .guide file into a list of MtrGuide."""
    toks = _lex(text)
    spans = _lex_spans(text)
    src = _Cursor(toks)
    out = []
    while True:
        tk = src.read()
        if tk is None:
            break
        low = tk.val.lower()
        if low not in ('guide', 'inlineguide'):
            diags.append(MtrDiagnostic(
                DIAG_PARSE_ERROR, 'guide-syntax',
                'unexpected token "%s" in guide file' % tk.val,
                filename, tk.line))
            continue
        name_tk = src.read()
        if name_tk is None:
            break
        parms = []
        open_tk = src.read()
        if open_tk is None or open_tk.val != '(':
            diags.append(MtrDiagnostic(
                DIAG_PARSE_ERROR, 'guide-syntax',
                'guide "%s" has no parameter list' % name_tk.val,
                filename, name_tk.line))
            continue
        while True:
            t = src.read()
            if t is None or t.val == ')':
                break
            if t.val == ',':
                continue
            parms.append(t.val)
        # The body is captured as raw source text, because substitution is
        # textual. Find it by brace-matching over the token stream, then slice
        # the original text between the braces' character offsets.
        body_start = src.i
        if not src.skip_braced():
            diags.append(MtrDiagnostic(
                DIAG_PARSE_ERROR, 'guide-syntax',
                'guide "%s" has an unbalanced body' % name_tk.val,
                filename, name_tk.line))
            break
        body = _slice_source(text, spans, body_start, src.i - 1,
                             include_braces=not low.startswith('inline'))
        out.append(MtrGuide(name_tk.val, parms, body, low == 'inlineguide',
                            filename, name_tk.line))
    return out


def _lex_spans(text):
    """Character spans of the tokens _lex() would return, in the same order.

    Guide expansion is the only thing that needs raw source offsets (because
    guide substitution is textual, not token-based), so this stays out of the
    hot path rather than making every token carry two more ints.
    """
    spans = []
    for m in _TOKEN_RE.finditer(text):
        if m.lastgroup in ('BLOCKCOMMENT', 'LINECOMMENT', 'WS', 'OTHER'):
            continue
        spans.append((m.start(), m.end()))
    return spans


def _slice_source(text, spans, first_idx, last_idx, include_braces=True):
    """Source text spanning tokens[first_idx..last_idx]."""
    if first_idx >= len(spans) or last_idx >= len(spans):
        return ''
    if include_braces:
        return text[spans[first_idx][0]:spans[last_idx][1]]
    # first_idx points at '{', last_idx at its matching '}'
    return text[spans[first_idx][1]:spans[last_idx][0]]


class _LineMap(object):
    """Translates a line number in guide-expanded text back to the original.

    Guide bodies are usually longer than the one-line invocation they replace,
    so without this every diagnostic after the first expansion in a Quake 4
    file would point at the wrong line - which would make the report actively
    misleading rather than merely incomplete.
    """
    __slots__ = ('breaks',)

    def __init__(self, breaks=None):
        # sorted list of (first_expanded_line_affected, cumulative_delta)
        self.breaks = breaks or []

    def original(self, line):
        if not self.breaks:
            return line
        idx = _bisect.bisect_right([b[0] for b in self.breaks], line) - 1
        if idx < 0:
            return line
        return max(1, line - self.breaks[idx][1])

    def __bool__(self):
        return bool(self.breaks)

    __nonzero__ = __bool__


def _expand_guides(text, filename, guides, diags):
    """Apply `guide` / `inlineGuide` expansion to one .mtr source.

    Returns (expanded_text, _LineMap). Invocations are replaced in place
    rather than appended (which is what the engine does) so that the
    surrounding material definitions keep their original ordering, and the
    line map puts diagnostics back onto real source lines.
    """
    lowered = text.lower()
    if 'guide' not in lowered:
        return text, _LineMap()
    toks = _lex(text)
    spans = _lex_spans(text)
    edits = []
    i = 0
    n = len(toks)
    while i < n:
        low = toks[i].val.lower()
        if low not in ('guide', 'inlineguide'):
            i += 1
            continue
        inline = (low == 'inlineguide')
        start_char = spans[i][0]
        j = i + 1
        decl_name = None
        if not inline:
            if j >= n:
                break
            decl_name = toks[j].val
            j += 1
        if j >= n:
            break
        gname = toks[j].val
        j += 1
        if j >= n or toks[j].val != '(':
            # Not an invocation after all (e.g. the word appears in prose).
            i += 1
            continue
        j += 1
        args = []
        while j < n and toks[j].val != ')':
            if toks[j].val == ',':
                j += 1
                continue
            args.append(toks[j].val)
            j += 1
        if j >= n:
            break
        end_char = spans[j][1]
        j += 1

        guide = guides.get(engine_canonical_decl(gname))
        if guide is None:
            diags.append(MtrDiagnostic(
                DIAG_UNSUPPORTED, 'missing-guide',
                'guide "%s" is not defined in any .guide file' % gname,
                filename, toks[i].line, decl_name or ''))
            # Blank the invocation so it cannot desync the decl scanner.
            edits.append((start_char, end_char, ''))
            i = j
            continue

        body = guide.expand(args)
        repl = body if inline else '%s\n%s\n' % (decl_name, body)
        edits.append((start_char, end_char, repl))
        i = j

    if not edits:
        return text, _LineMap()

    out = []
    breaks = []
    prev = 0
    expanded_line = 1
    delta = 0
    for a, b, repl in edits:
        chunk = text[prev:a]
        out.append(chunk)
        expanded_line += chunk.count('\n')
        consumed = text[a:b]
        out.append(repl)
        delta += repl.count('\n') - consumed.count('\n')
        expanded_line += repl.count('\n')
        breaks.append((expanded_line, delta))
        prev = b
    out.append(text[prev:])
    return ''.join(out), _LineMap(breaks)


# ---------------------------------------------------------------------------
# Expressions - renderer/Material.cpp, idMaterial::ParseExpressionPriority
# ---------------------------------------------------------------------------
# Four priority levels, and - this is the part that surprises people - the
# operators are RIGHT-associative, because ParseEmitOp recurses at its own
# priority rather than the next one down:
#
#     int idMaterial::ParseEmitOp( src, a, opType, priority ) {
#         b = ParseExpressionPriority( src, priority );      // same priority
#         return EmitOp( a, b, opType );
#     }
#
# So `time - parm4 - 0.5` is time - (parm4 - 0.5), and `4 / 2 / 2` is 4.
# We reproduce that exactly rather than "fixing" it, because the goal is to
# match what the game draws. EXPR_ASSOC_NOTE below is surfaced in the UI and
# expr_grouping_is_surprising() flags the specific expressions where it
# changes the answer, so nobody has to take it on faith.

EXPR_ASSOC_NOTE = ('idTech 4 operators are right-associative and % truncates '
                   'to integers (renderer/Material.cpp ParseEmitOp / '
                   'OP_TYPE_MOD). Driver expressions are emitted fully '
                   'parenthesised so the grouping is visible.')

_OPS_BY_PRIORITY = {
    1: ('*', '/', '%'),
    2: ('+', '-'),
    3: ('>', '>=', '<', '<=', '==', '!='),
    4: ('&&', '||'),
}
_TOP_PRIORITY = 4

# Predefined expression terms. parm0-11 and global0-7 are the shader parm
# registers; the rest are engine state. `sound` is the current sound amplitude
# (OP_TYPE_SOUND) and shows up in ~500 light materials, `distance` is Prey's
# EXP_REG_DISTANCE, and the remainder are Quake 4 additions.
_EXPR_VARS = frozenset(
    ['time', 'sound', 'fragmentprograms', 'glslprograms',
     'distance', 'ismultiplayer', 'vertexrandomizer', 'vieworigin',
     'potcorrectionx', 'potcorrectiony', 'decallife']
    + ['parm%d' % i for i in range(12)]
    + ['global%d' % i for i in range(8)]
)

# The Dark Mod adds min()/max() as expression functions (Material.cpp:627).
# Note `max` is also a *stage keyword* on a parallaxmap stage in TDM - the
# same word means two different things depending on parse position, which a
# position-driven parser gets right for free and a flat key/value walk cannot
# express at all.
_EXPR_FUNCS = frozenset(['min', 'max'])

EXPR_CONST, EXPR_VAR, EXPR_TABLE, EXPR_OP = 'const', 'var', 'table', 'op'


class MtrExpr(object):
    """One node of a parsed material expression."""
    __slots__ = ('kind', 'a', 'b', 'op', 'source')

    def __init__(self, kind, a=None, b=None, op=None, source=''):
        self.kind = kind
        self.a = a
        self.b = b
        self.op = op
        self.source = source        # original .mtr text, for the node property

    # -- construction helpers ------------------------------------------------
    @staticmethod
    def const(v):
        return MtrExpr(EXPR_CONST, float(v))

    # -- inspection ----------------------------------------------------------
    def is_const(self):
        return self.kind == EXPR_CONST

    def walk(self):
        yield self
        for child in (self.a, self.b):
            if isinstance(child, MtrExpr):
                for sub in child.walk():
                    yield sub

    def uses(self):
        """Set of variable/table names this expression depends on."""
        names = set()
        for node in self.walk():
            if node.kind == EXPR_VAR:
                names.add(node.a)
            elif node.kind == EXPR_TABLE:
                names.add('table:' + node.a)
        return names

    def is_dynamic(self):
        """True if the value can change at runtime (time, parms, tables)."""
        for node in self.walk():
            if node.kind in (EXPR_VAR, EXPR_TABLE):
                return True
        return False

    def is_time_varying(self):
        """True if the value advances on its own as the timeline plays.

        Deliberately narrower than is_dynamic(). parm0..11, global0..7 and
        sound are all "dynamic" - they are not constants - but none of them
        moves unless the user moves a panel slider, and a table indexed by
        one of them is just as still. Only `time` advances by itself, and
        only expressions that reach it need a driver; see
        _Builder._attach_expression() for why the distinction is worth
        making.
        """
        for node in self.walk():
            if node.kind == EXPR_VAR and node.a == 'time':
                return True
        return False

    def __repr__(self):
        if self.kind == EXPR_CONST:
            return _fmt_float(self.a)
        if self.kind == EXPR_VAR:
            return self.a
        if self.kind == EXPR_TABLE:
            return '%s[%r]' % (self.a, self.b)
        if self.op in _EXPR_FUNCS:
            return '%s(%r, %r)' % (self.op, self.a, self.b)
        return '(%r %s %r)' % (self.a, self.op, self.b)


def _fmt_float(v):
    if v == int(v) and abs(v) < 1e15:
        return '%d' % int(v)
    return repr(round(v, 6))


def _to_float(text):
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return float(int(text, 16))
    except ValueError:
        return 0.0


class _ParseState(object):
    """Per-material parse context: the table registry plus error latching."""
    __slots__ = ('tables', 'diags', 'filename', 'material', 'failed',
                 'arg_keyword', 'arg_line', 'arg_start')

    def __init__(self, tables, diags, filename, material):
        self.tables = tables
        self.diags = diags
        self.filename = filename
        self.material = material
        self.failed = False
        # Which keyword's argument the expression parser is currently inside,
        # and the token index its argument started at. Set by begin_argument()
        # as each keyword is dispatched; _parse_term needs it to tell "this
        # expression contains a bogus token" from "this keyword has no
        # expression at all, so the parser took the next line's first word".
        self.arg_keyword = ''
        self.arg_line = 0
        self.arg_start = -1

    def begin_argument(self, keyword, line, start):
        self.arg_keyword = keyword
        self.arg_line = line
        self.arg_start = start

    def note(self, level, kind, message, line):
        self.diags.append(MtrDiagnostic(level, kind, message, self.filename,
                                        line, self.material))

    def fail(self, kind, message, line):
        """Latch a parse error. The engine calls SetMaterialFlag(MF_DEFAULTED)
        here and then MakeDefault()s the whole material, so we do the same -
        a definition this broken is broken in the game too."""
        self.note(DIAG_PARSE_ERROR, kind, message, line)
        self.failed = True


def _expect(src, state, want, fatal=True):
    """idMaterial::MatchToken.

    The engine's MatchToken sets MF_DEFAULTED when the token is missing, which
    makes the whole material fall back to the default checkerboard - so a
    missing comma or bracket really is fatal, and reproducing that is how we
    end up flagging the handful of definitions that are broken in the shipping
    games too. Image programs are the exception: MatchAndAppendToken just
    returns, so _parse_image() uses its own non-latching matcher.
    """
    tk = src.read()
    if tk is None or tk.val != want:
        if tk is not None:
            src.unread()
        if fatal:
            state.fail('expected-token', 'expected "%s"' % want, src.line())
        else:
            state.note(DIAG_PARSE_ERROR, 'expected-token',
                       'expected "%s"' % want, src.line())
        return False
    return True


def _parse_term(src, state):
    """idMaterial::ParseTerm."""
    term_at = src.i
    tk = src.read()
    if tk is None:
        state.fail('truncated', 'end of file inside an expression', src.line())
        return MtrExpr.const(0.0)

    if tk.val == '(':
        inner = _parse_expr(src, state)
        _expect(src, state, ')')
        return inner

    low = tk.val.lower()

    if low in _EXPR_VARS:
        return MtrExpr(EXPR_VAR, low, source=tk.val)

    if low in _EXPR_FUNCS:
        nxt = src.peek()
        if nxt is not None and nxt.val == '(':
            # TDM: min( a, b, ... ) / max( a, b, ... ), folded left to right
            src.read()
            node = _parse_expr(src, state)
            while True:
                nxt = src.peek()
                if nxt is None or nxt.val != ',':
                    break
                src.read()
                node = MtrExpr(EXPR_OP, node, _parse_expr(src, state), low)
            _expect(src, state, ')')
            return node

    if tk.val == '-':
        # The engine only accepts a negative *literal* here; `-parm0` is a
        # parse error in the game too.
        nxt = src.read()
        if nxt is not None and (nxt.type == TT_NUMBER or nxt.val == '.'):
            return MtrExpr.const(-_to_float(nxt.val))
        if nxt is not None:
            src.unread()
        state.fail('bad-negative', 'expected a number after "-"', tk.line)
        return MtrExpr.const(0.0)

    if tk.type == TT_NUMBER:
        return MtrExpr.const(_to_float(tk.val))

    if tk.val == '.':
        return MtrExpr.const(0.0)

    # Anything else must be a table lookup: name[ expr ]
    nxt = src.read()
    if nxt is None or nxt.val != '[':
        if nxt is not None:
            src.unread()
        # Report the CAUSE, not the symptom. A keyword whose expression is
        # missing entirely does not fail on its own line - idLexer crosses
        # newlines, so ParseTerm reads the first word of the NEXT line and
        # blames that. Doom 3's textures/sfx/flare_toggle is the case in
        # point: a bare `rgba` on one line, `red parm0 * parm7` on the next,
        # and the engine's own warning is "Bad term 'red'" - which sends you
        # looking at a line that is perfectly correct.
        #
        # The two are told apart exactly, not guessed: if the offending token
        # is the very first one of this keyword's argument AND it is on a
        # different line from the keyword, the argument is missing. Anything
        # else really is a bad token inside an expression that started fine.
        #
        # The kind carries the name either way, because the report groups by
        # kind and shows only the first few entries per group - with a fixed
        # slug every distinct case in a tree collapses into one unreadable row.
        if state.arg_keyword and term_at == state.arg_start                 and tk.line != state.arg_line:
            state.fail(
                # Lowercased for the kind, verbatim in the message: the kind
                # is a grouping key, and a .mtr writing RGBA would otherwise
                # get a report row of its own.
                'missing-expression:' + state.arg_keyword.lower(),
                '"%s" has no expression after it, so "%s" from line %d was '
                'read as its argument. The engine does the same and defaults '
                'the whole material.'
                % (state.arg_keyword, tk.val, tk.line), state.arg_line)
        else:
            state.fail(
                'bad-term:' + low,
                'unknown expression term "%s"%s' % (
                    tk.val,
                    ' in the expression after "%s"' % state.arg_keyword
                    if state.arg_keyword else ''),
                tk.line)
        return MtrExpr.const(0.0)
    index = _parse_expr(src, state)
    _expect(src, state, ']')
    # A table is a decl, so its name goes through MakeNameCanonical on both
    # sides - here and at registration in _parse_table. `low` is what the
    # rest of this branch already used and is right for every table any
    # shipped .mtr declares; canonical is what the engine actually indexes.
    low = engine_canonical_decl(tk.val)
    if low not in state.tables:
        # Noted unconditionally here and re-checked once the whole tree is
        # loaded - see _drop_resolved_table_notes(). A table declared in a file
        # that sorts later than this one is NOT undefined: idDeclManagerLocal
        # indexes every decl name at startup, so idMaterial::ParseTerm's
        # declManager->FindType( DECL_TABLE, ... ) finds a table wherever it
        # lives. Checking against a half-built db.tables reported 436 of
        # Doom 3's 436 table references as undefined when only one was.
        state.note(DIAG_CONTENT, TABLE_NOTE_PREFIX + low,
                   'table "%s" is not defined in any .mtr file' % tk.val,
                   tk.line)
    return MtrExpr(EXPR_TABLE, low, index, source=tk.val)


def _parse_priority(src, state, priority):
    if priority == 0:
        return _parse_term(src, state)
    a = _parse_priority(src, state, priority - 1)
    if state.failed:
        return a
    tk = src.read()
    if tk is None:
        return a
    if tk.val in _OPS_BY_PRIORITY[priority]:
        b = _parse_priority(src, state, priority)   # right-assoc, as the engine
        return MtrExpr(EXPR_OP, a, b, tk.val)
    src.unread()
    return a


def _parse_expr(src, state):
    """idMaterial::ParseExpression."""
    start = src.i
    node = _parse_priority(src, state, _TOP_PRIORITY)
    if not node.source:
        node.source = _joined_source(src, start, src.i)
    return node


def _joined_source(src, start, end):
    """Reconstruct readable source text for a token span (for node labels)."""
    parts = []
    for tk in src.toks[start:end]:
        val = tk.val
        if parts and val not in (',', ')', ']', '[') and parts[-1] not in ('(', '['):
            parts.append(' ')
        parts.append(val)
    return ''.join(parts).strip()


def expr_grouping_is_surprising(expr):
    """True when right-associativity changes this expression's value.

    Only `-` and `/` (and `%`) are order-sensitive, and only when they have
    another same-priority operator on their right. `a * b * c` is unaffected;
    `a - b - c` and `a - b + c` are. Used to report the handful of materials
    where engine-exact grouping differs from the obvious reading, instead of
    slapping a blanket disclaimer on everything.
    """
    for node in expr.walk():
        if node.kind != EXPR_OP or node.op not in ('-', '/', '%'):
            continue
        right = node.b
        if isinstance(right, MtrExpr) and right.kind == EXPR_OP:
            same = _OPS_BY_PRIORITY[1] if node.op in ('*', '/', '%') \
                else _OPS_BY_PRIORITY[2]
            if right.op in same:
                return True
    return False


# ---------------------------------------------------------------------------
# Image programs - renderer/Image_program.cpp, R_ParseImageProgram_r
# ---------------------------------------------------------------------------
# Image programs are a small recursive expression language over texture files
# that the engine evaluates once at load time and caches. They read from the
# same token stream as the surrounding material, so they have to be parsed,
# not regex-matched.

# op name -> (arity of image arguments, arity of trailing scalar arguments)
_IMAGE_OPS = {
    'heightmap':           (1, 1),
    'addnormals':          (2, 0),
    'add':                 (2, 0),
    'smoothnormals':       (1, 0),
    'invertalpha':         (1, 0),
    'invertcolor':         (1, 0),
    'makeintensity':       (1, 0),
    'makealpha':           (1, 0),
    'scale':               (1, 4),
    # Quake 4
    'downsize':            (1, 1),
    # The Dark Mod (renderer/resources/Image_program.cpp)
    'bakeambientdiffuse':  (1, 0),
    'bakeambientspecular': (1, 0),
    'nativelayout':        (1, 0),
    'cameralayout':        (1, 0),
}

# Images the engine generates internally; there is no file to load.
BUILTIN_IMAGES = frozenset((
    '_white', '_black', '_flat', '_default', '_quadratic', '_noflashlight',
    '_currentrender', '_currentdepth', '_scratch', '_fog', '_fogenter',
    '_xray', '_shadowatlas', '_videocapture', '_accum', '_ldr', '_smaaarea',
    '_smaasearch', '_bloomrender', '_envprobehdr', '_guirender',
))


class MtrImage(object):
    """A parsed image program. `op` is None for a plain texture reference."""
    __slots__ = ('op', 'args', 'scalars', 'path', 'canonical')

    def __init__(self, op=None, args=None, scalars=None, path=None,
                 canonical=''):
        self.op = op
        self.args = args or []       # list of MtrImage
        self.scalars = scalars or []  # list of float
        self.path = path             # set only when op is None
        self.canonical = canonical   # engine's canonical spelling

    def is_builtin(self):
        return self.op is None and self.path is not None \
            and self.path.lower() in BUILTIN_IMAGES

    def base_path(self):
        """The first real texture path inside this program, or None.

        Used for the fallback chain: even when an image program cannot be
        reproduced node-for-node, the texture underneath it usually can.
        """
        if self.op is None:
            return self.path
        for arg in self.args:
            found = arg.base_path()
            if found:
                return found
        return None

    def all_paths(self):
        if self.op is None:
            return [self.path] if self.path else []
        out = []
        for arg in self.args:
            out.extend(arg.all_paths())
        return out

    def __repr__(self):
        return self.canonical or (self.path or '<empty>')


def _parse_image(src, state, depth=0):
    """R_ParseImageProgram_r.

    The returned node also carries the engine's canonical spelling of the
    program, because that string is the key the generated/images/**.bimage
    cache is named after - see the .bimage fallback near the top of this
    file. The engine's AppendToken puts a space before every token except
    one matched by MatchAndAppendToken, which is why the canonical form of
    `addnormals(a,heightmap(b,3))` is `addnormals( a, heightmap( b, 3))`.
    """
    def match(want):
        tk = src.read()
        if tk is None or tk.val != want:
            if tk is not None:
                src.unread()
            state.note(DIAG_PARSE_ERROR, 'image-program',
                       'expected "%s" in image program' % want, src.line())
            return False
        return True

    tk = src.read()
    if tk is None:
        state.fail('truncated', 'end of file inside an image program',
                   src.line())
        return MtrImage(path='', canonical='')

    low = tk.val.lower()
    spec = _IMAGE_OPS.get(low) if depth < 16 else None
    if spec is None:
        # A plain texture reference. Quoted paths arrive with their quotes
        # already stripped by the lexer, which is one of the things the old
        # regex parser could not do.
        return MtrImage(path=tk.val, canonical=tk.val)

    n_images, n_scalars = spec
    match('(')
    args = []
    for k in range(n_images):
        if k:
            match(',')
        args.append(_parse_image(src, state, depth + 1))
    scalars = []
    for _ in range(n_scalars):
        match(',')
        t = src.read()
        if t is None:
            break
        if t.val == '-':
            t2 = src.read()
            scalars.append(-_to_float(t2.val) if t2 else 0.0)
        else:
            scalars.append(_to_float(t.val))
    match(')')
    node = MtrImage(op=low, args=args, scalars=scalars)
    node.canonical = _canonical_image(node)
    return node


def _canonical_image(img):
    """Rebuild R_ParsePastImageProgram's parseBuffer spelling."""
    if img.op is None:
        return img.path or ''
    parts = [img.op, '(']
    for i, arg in enumerate(img.args):
        if i:
            parts.append(',')
        parts.append(' ' + _canonical_image(arg))
    for s in img.scalars:
        parts.append(', ' + _fmt_float(s))
    parts.append(')')
    return ''.join(parts)


# ---------------------------------------------------------------------------
# Keyword tables
# ---------------------------------------------------------------------------
# One merged table covers Doom 3, Doom 3 BFG, Quake 4, Prey and The Dark Mod.
# There is deliberately no per-game mode: the union was validated against all
# 28456 material declarations in the five shipping bases and 5 of them fail,
# every one of which the real engine also rejects.
#
# Each entry names the *kind* of value the keyword consumes, which is the
# thing the old parser had to guess. Kinds:
#
#   flag            consumes nothing (a boolean)
#   tok             one token
#   tok_line        one token, only if on the same line
#   tok_line_rest   one token on the line, then skip the rest of the line
#   rest            the rest of the line
#   image           an image program
#   expr            one expression
#   expr2/3/4       N comma-separated expressions
#   expr_opt        an expression, but only if something follows on the line
#   float_opt       an optional float on the same line (defaults to 1)
#   num_or_vec3     one scalar or three (TDM)
#   ...plus the bespoke ones handled explicitly in _read_value()
#
# WHERE ENGINES DISAGREE, THE MERGED TABLE TAKES THE MOST TOLERANT ARITY THAT
# IS A SUPERSET OF BOTH. The clearest case is mirrorRenderMap: Doom 3 reads
# two integers, The Dark Mod reads nothing (it has a separate
# mirrorResolutionFactor keyword). 'rest' satisfies both, because in real
# content a keyword and its arguments always share a line even though the
# lexer does not require it.

MTR_MATERIAL_KEYWORDS = {
    # -- Doom 3 core -------------------------------------------------------
    'qer_editorimage': 'tok_line_rest',
    'description': 'tok',
    'polygonoffset': 'float_opt',
    'noshadows': 'flag', 'suppressinsubview': 'flag', 'portalsky': 'flag',
    'noselfshadow': 'flag', 'noportalfog': 'flag', 'forceshadows': 'flag',
    'nooverlays': 'flag', 'forceoverlays': 'flag', 'translucent': 'flag',
    'zeroclamp': 'flag', 'clamp': 'flag', 'alphazeroclamp': 'flag',
    'forceopaque': 'flag', 'twosided': 'flag', 'backsided': 'flag',
    'foglight': 'flag', 'blendlight': 'flag', 'ambientlight': 'flag',
    'mirror': 'flag', 'nofog': 'flag', 'unsmoothedtangents': 'flag',
    'lightfalloffimage': 'image', 'guisurf': 'tok', 'sort': 'sort',
    'spectrum': 'tok', 'deform': 'deform', 'decalinfo': 'decalinfo',
    'renderbump': 'rest',
    'diffusemap': 'image', 'specularmap': 'image', 'bumpmap': 'image',
    'decal_macro': 'flag',
    # -- surface parms (CheckSurfaceParm; all valueless) -------------------
    'solid': 'flag', 'water': 'flag', 'playerclip': 'flag',
    'monsterclip': 'flag', 'moveableclip': 'flag', 'ikclip': 'flag',
    'blood': 'flag', 'trigger': 'flag', 'aassolid': 'flag',
    'aasobstacle': 'flag', 'flashlight_trigger': 'flag', 'nonsolid': 'flag',
    'nullnormal': 'flag', 'areaportal': 'flag', 'qer_nocarve': 'flag',
    'discrete': 'flag', 'nofragment': 'flag', 'slick': 'flag',
    'collision': 'flag', 'noimpact': 'flag', 'nodamage': 'flag',
    'ladder': 'flag', 'nosteps': 'flag', 'nodrop': 'flag',
    'metal': 'flag', 'stone': 'flag', 'flesh': 'flag', 'wood': 'flag',
    'cardboard': 'flag', 'liquid': 'flag', 'glass': 'flag', 'plastic': 'flag',
    'ricochet': 'flag', 'surftype10': 'flag', 'surftype11': 'flag',
    'surftype12': 'flag', 'surftype13': 'flag', 'surftype14': 'flag',
    'surftype15': 'flag',
    # -- The Dark Mod ------------------------------------------------------
    'frobstage_texture': 'frob_texture',
    'frobstage_diffuse': 'frob_diffuse',
    'frobstage_none': 'flag',
    'islightgemsurf': 'tok', 'forceinteractions': 'flag',
    'interactionseparator': 'flag', 'cubiclight': 'flag', 'fogalpha': 'tok',
    'lightambientdiffuse': 'image', 'lightambientspecular': 'image',
    'parallaxmap': 'image', 'particle_macro': 'flag',
    'twosided_decal_macro': 'flag',
    # -- Quake 4 (_RAVEN) --------------------------------------------------
    # materialImage takes a full image program in Quake 4, e.g.
    #   materialImage downsize( .../bb_gd_lo_hit.tga, 2 )
    'materialimage': 'image', 'materialtype': 'tok',
    'needcurrentrender': 'flag', 'notfix': 'flag',
    'portaldistancefar': 'tok', 'portaldistancenear': 'tok',
    'portalimage': 'image', 'sightclip': 'flag', 'sky': 'flag',
    # Quake 4 game-side surface types. They carry no shading meaning, but
    # listing them keeps them out of the report as noise.
    'projectileclip': 'flag', 'shotclip': 'flag', 'largeshotclip': 'flag',
    'vehicleclip': 'flag', 'flyclip': 'flag', 'itemclip': 'flag',
    'notacticalfeatures': 'flag', 'bounce': 'flag',
    # -- Prey (_HUMANHEAD) -------------------------------------------------
    'decal_alphatest_macro': 'flag', 'directportal': 'expr_opt',
    'glass_macro': 'flag', 'lightwholemesh': 'flag', 'noseethru': 'flag',
    # Prey writes both `directPortal` and `directPortal parm5`.
    'seethru': 'flag', 'skipclip': 'flag', 'overlay_macro': 'flag',
    'scorch_macro': 'flag', 'skybox_macro': 'flag', 'skyboxportal': 'flag',
    'matter_cardboard': 'flag', 'matter_flesh': 'flag', 'matter_glass': 'flag',
    'matter_metal': 'flag', 'matter_pipe': 'flag', 'matter_stone': 'flag',
    'matter_tile': 'flag', 'matter_wood': 'flag',
    # Prey game-side surface types, absent from the ported renderer source
    # but present in the shipping content.
    'matter_altmetal': 'flag', 'matter_liquid': 'flag', 'wallwalk': 'flag',
    'forcefield': 'flag', 'forcefield_nobullets': 'flag', 'detail': 'flag',
    'hunterclip': 'flag',
    # -- Doom 3 BFG --------------------------------------------------------
    'basecolormap': 'image', 'normalmap': 'image', 'rmaomap': 'image',
    'pbrmap': 'image', 'reflectionmap': 'image', 'mikktspace': 'flag',
    'origin': 'flag', 'stereoeye': 'tok', 'persistentlod': 'flag',
    'lod1': 'flag', 'lod2': 'flag', 'lod3': 'flag', 'lod4': 'flag',
}

MTR_STAGE_KEYWORDS = {
    # -- Doom 3 core -------------------------------------------------------
    'name': 'rest', 'blend': 'blend', 'map': 'image',
    'remoterendermap': 'rest', 'mirrorrendermap': 'rest',
    'xrayrendermap': 'rest', 'guirendermap': 'rest', 'portalrendermap': 'rest',
    'screen': 'flag', 'screen2': 'flag', 'glasswarp': 'flag',
    'videomap': 'videomap', 'soundmap': 'tok',
    'cubemap': 'image', 'cameracubemap': 'image',
    'ignorealphatest': 'flag', 'nearest': 'flag', 'linear': 'flag',
    'clamp': 'flag', 'noclamp': 'flag', 'zeroclamp': 'flag',
    'alphazeroclamp': 'flag', 'uncompressed': 'flag', 'highquality': 'flag',
    'forcehighquality': 'flag', 'nopicmip': 'flag',
    'vertexcolor': 'flag', 'inversevertexcolor': 'flag',
    'privatepolygonoffset': 'float_opt', 'texgen': 'texgen',
    'scroll': 'expr2', 'translate': 'expr2', 'scale': 'expr2',
    'centerscale': 'expr2', 'shear': 'expr2', 'rotate': 'expr',
    'maskred': 'flag', 'maskgreen': 'flag', 'maskblue': 'flag',
    'maskalpha': 'flag', 'maskcolor': 'flag', 'maskdepth': 'flag',
    'alphatest': 'expr', 'colored': 'flag', 'color': 'expr4',
    'red': 'expr', 'green': 'expr', 'blue': 'expr', 'alpha': 'expr',
    'rgb': 'expr', 'rgba': 'expr', 'if': 'expr',
    'program': 'tok_line', 'fragmentprogram': 'tok_line',
    'vertexprogram': 'tok_line', 'megatexture': 'tok_line',
    'vertexparm': 'indexed_parm', 'fragmentmap': 'indexed_map',
    # -- The Dark Mod ------------------------------------------------------
    'grazingangle': 'tok', 'ignoredepth': 'flag', 'linearsteps': 'tok',
    'max': 'expr', 'min': 'expr', 'mirrorresolutionfactor': 'tok',
    'offsetexternalshadows': 'tok', 'refinesteps': 'tok',
    'remoteresolution': 'tok', 'shadowsoftness': 'tok', 'shadowsteps': 'tok',
    'withaudio': 'flag',
    # -- Prey --------------------------------------------------------------
    'glowstage': 'flag', 'growin': 'expr', 'growout': 'expr',
    'highres': 'flag', 'specularexp': 'expr2', 'fragmentparm': 'indexed_parm',
    'notscopeview': 'flag', 'notspiritwalk': 'flag', 'scopeview': 'flag',
    'shuttleview': 'flag', 'spiritwalk': 'flag',
    'shaderfallback1': 'flag', 'shaderfallback2': 'flag',
    'shaderfallback3': 'flag', 'shaderlevel1': 'flag',
    'shaderlevel2': 'flag', 'shaderlevel3': 'flag',
    # -- Quake 4 -----------------------------------------------------------
    'nomips': 'flag', 'glslprogram': 'tok_line',
    'shaderparm': 'named_parm', 'shadertexture': 'named_texture',
    # -- Doom 3 BFG --------------------------------------------------------
    'cubemapsingle': 'image', 'uncompressedcubemap': 'image',
    'stereomap': 'image', 'cubemapsize': 'tok', 'rendertargetmap': 'tok',
    'stencil': 'braced', 'vertexparm2': 'indexed_parm',
}

# Options that may precede the image in `fragmentMap <n> [opts] <image>`.
_FRAGMENTMAP_OPTIONS = frozenset((
    'cubemap', 'cameracubemap', 'nearest', 'linear', 'clamp', 'noclamp',
    'zeroclamp', 'alphazeroclamp', 'forcehighquality', 'uncompressed',
    'highquality', 'nopicmip', 'highres', 'nomips',
))

# `blend <shorthand>` values. The first four set a src/dst pair; the rest
# select an interaction lighting slot instead of a blend.
BLEND_SHORTHANDS = {
    'blend':        ('gl_src_alpha', 'gl_one_minus_src_alpha'),
    'add':          ('gl_one', 'gl_one'),
    'filter':       ('gl_dst_color', 'gl_zero'),
    'modulate':     ('gl_dst_color', 'gl_zero'),
    'none':         ('gl_zero', 'gl_one'),
}
BLEND_LIGHTING = {
    'bumpmap': 'bump', 'diffusemap': 'diffuse', 'specularmap': 'specular',
    'parallaxmap': 'parallax',                       # The Dark Mod
    'basecolormap': 'diffuse', 'normalmap': 'bump',  # Doom 3 BFG
    'rmaomap': 'specular', 'pbrmap': 'specular', 'reflectionmap': 'specular',
    'coverage': 'ambient',
    'shader': 'ambient',                             # Prey custom shader
}

_GL_BLEND_FACTORS = frozenset((
    'gl_one', 'gl_zero', 'gl_dst_color', 'gl_one_minus_dst_color',
    'gl_src_alpha', 'gl_one_minus_src_alpha', 'gl_dst_alpha',
    'gl_one_minus_dst_alpha', 'gl_src_alpha_saturate', 'gl_src_color',
    'gl_one_minus_src_color',
))

TEXGEN_VALUES = frozenset(('normal', 'reflect', 'skybox', 'wobblesky',
                           'glasswarp', 'screen', 'screen2'))

# idMaterial::ParseSort. Numeric values are also allowed.
SORT_VALUES = {
    'subview': -3.0, 'gui': -2.0, 'bad': -1.0, 'opaque': 0.0,
    'portalsky': 1.0, 'decal': 2.0, 'far': 3.0, 'medium': 4.0, 'close': 5.0,
    'almostnearest': 6.0, 'nearest': 7.0, 'afterfog': 8.0, 'postprocess': 100.0,
    'last': 101.0,
    # ETQW-derived names that also appear in a few Quake 4 files
    'opaquefirst': 0.0, 'opaquenearer': 0.0, 'opaquenearest': 0.0,
    'refractable': 4.0, 'refraction': 4.0,
}

# idMaterial::ParseDeform. The value is the number of expressions that
# follow; 'decl' means a decl name follows; 'rest' means the engine calls
# SkipRestOfLine (Prey's corona/jitter/beam and Quake 4's rectsprite).
DEFORM_VALUES = {
    'sprite': 0, 'tube': 0, 'eyeball': 0, 'flare': 1, 'expand': 1, 'move': 1,
    'turbulent': 'turbulent', 'particle': 'decl', 'particle2': 'decl',
    'corona': 'rest', 'jitter': 'rest', 'beam': 'rest', 'rectsprite': 'rest',
}

# Deforms that force two-sided rendering and disable shadows.
_DEFORM_TWOSIDED = frozenset(('sprite', 'tube', 'flare', 'corona', 'jitter',
                              'beam', 'rectsprite'))


# ---------------------------------------------------------------------------
# Stage and material objects
# ---------------------------------------------------------------------------

# How the stage's texture is sourced. Everything other than 'file' and
# 'builtin' needs a fallback texture and a report entry.
TEX_FILE      = 'file'        # an ordinary texture (possibly an image program)
TEX_BUILTIN   = 'builtin'     # _white, _black, _flat, _currentRender, ...
TEX_CUBE      = 'cube'        # cubeMap / cameraCubeMap
TEX_DYNAMIC   = 'dynamic'     # mirror/remote/xray/gui render targets
TEX_VIDEO     = 'video'       # videoMap / soundMap
TEX_NONE      = 'none'

# Texture wrap mode, from the trp/trpDefault machinery in ParseStage.
WRAP_REPEAT, WRAP_CLAMP, WRAP_ZERO, WRAP_ZERO_ALPHA = \
    'repeat', 'clamp', 'zeroclamp', 'alphazeroclamp'


class MtrTexTransform(object):
    """One texture-matrix keyword, kept in source order.

    Order matters and is not the obvious one. MultiplyTextureMatrix computes
    `new = old * reg`, so a keyword is composed on the *right* of everything
    already accumulated - meaning the LAST keyword in the file is applied to
    the texture coordinates FIRST. The builder walks this list in reverse.
    """
    __slots__ = ('op', 'x', 'y')

    def __init__(self, op, x, y=None):
        self.op = op            # translate|scroll|scale|centerscale|shear|rotate
        self.x = x              # MtrExpr
        self.y = y              # MtrExpr or None for rotate

    def is_dynamic(self):
        if self.x is not None and self.x.is_dynamic():
            return True
        return self.y is not None and self.y.is_dynamic()

    def __repr__(self):
        if self.y is None:
            return '%s %r' % (self.op, self.x)
        return '%s %r, %r' % (self.op, self.x, self.y)


class MtrStage(object):
    """One parsed material stage."""

    def __init__(self):
        self.lighting = 'ambient'          # ambient|bump|diffuse|specular|parallax
        self.blend_src = 'gl_one'
        self.blend_dst = 'gl_zero'
        self.blend_name = ''               # the shorthand as written, if any
        self.image = None                  # MtrImage
        self.tex_kind = TEX_NONE
        self.tex_detail = ''               # e.g. 'mirrorRenderMap'
        self.color = [None, None, None, None]   # MtrExpr per channel, or None
        self.colored = False               # `colored` == rgba from parm0..3
        self.alpha_test = None             # MtrExpr
        self.condition = None              # MtrExpr from `if`
        self.transforms = []               # list of MtrTexTransform
        self.texgen = None
        self.texgen_args = []
        self.vertex_color = None           # None|'modulate'|'inverse'
        self.wrap = WRAP_REPEAT
        self.filter = None                 # None|'nearest'|'linear'
        self.mask = set()                  # red/green/blue/alpha/color/depth
        self.private_polygon_offset = 0.0
        self.flags = set()                 # every valueless keyword seen
        self.programs = {}                 # vertex/fragment/glsl program names
        self.fragment_maps = {}            # index -> (options, MtrImage)
        self.parms = {}                    # vertexParm/fragmentParm index->exprs
        self.name = ''                     # `name` keyword (material editor)
        self.implicit = False              # generated, not present in source
        self.source_line = 0
        self.unsupported = []              # list of (kind, message)

    # -- derived properties --------------------------------------------------
    @property
    def is_interaction(self):
        return self.lighting != 'ambient'

    @property
    def has_custom_program(self):
        return bool(self.programs)

    def blend_pair(self):
        return (self.blend_src, self.blend_dst)

    def is_invisible(self):
        """`blend none` / gl_zero,gl_one draws nothing."""
        return (self.blend_src, self.blend_dst) == ('gl_zero', 'gl_one')

    @property
    def writes_no_color(self):
        """True for a `maskColor` stage, which draws no visible colour.

        maskColor sets GLS_COLORMASK, which tr_backend.cpp turns into
        glColorMask(0, 0, 0, 1): RGB writes off, alpha writes on. Such a
        stage exists only to deposit a mask in the framebuffer's alpha
        channel for a later gl_dst_alpha stage to read back - its own colour
        never reaches the screen. Drawing it produces a solid white blob,
        because makeAlpha (its usual source) sets RGB to 255 by definition.
        """
        return 'color' in self.mask or {'red', 'green', 'blue'} <= self.mask

    @property
    def writes_alpha(self):
        """False for a `maskAlpha` stage (GLS_ALPHAMASK), which leaves the
        framebuffer's alpha channel - and so any mask sitting in it - alone."""
        return 'alpha' not in self.mask

    @property
    def reads_dest_alpha(self):
        """True when this stage's source factor is the framebuffer alpha a
        preceding maskColor stage deposited."""
        return self.blend_src == 'gl_dst_alpha'

    def blends_with_destination(self):
        """Coverage test from idMaterial::Parse - a stage that reads the
        framebuffer makes the material translucent."""
        if self.blend_dst != 'gl_zero':
            return True
        return self.blend_src in ('gl_dst_color', 'gl_one_minus_dst_color',
                                  'gl_dst_alpha', 'gl_one_minus_dst_alpha')

    def is_dynamic(self):
        """True if anything about this stage varies at runtime."""
        if self.condition is not None:
            return True
        for c in self.color:
            if c is not None and c.is_dynamic():
                return True
        if self.alpha_test is not None and self.alpha_test.is_dynamic():
            return True
        for t in self.transforms:
            if t.is_dynamic():
                return True
        return self.colored

    def __repr__(self):
        return '<stage %s %s/%s %r>' % (self.lighting, self.blend_src,
                                        self.blend_dst, self.image)


# Material coverage, from idMaterial::Parse.
COVERAGE_OPAQUE, COVERAGE_PERFORATED, COVERAGE_TRANSLUCENT = \
    'opaque', 'perforated', 'translucent'

# What kind of thing this decl is. Light materials are not surface shaders and
# building them as such is meaningless, so they get classified here and given
# a flat unlit preview of their projection texture by the builder.
SURFACE_MATERIAL = 'surface'
LIGHT_PROJECTED  = 'light'
LIGHT_AMBIENT    = 'ambient-light'
LIGHT_FOG        = 'fog-light'
LIGHT_BLEND      = 'blend-light'


class MtrMaterial(object):
    """A fully parsed, semantically normalised material."""

    def __init__(self, name, filename='', line=0):
        self.name = name
        self.filename = filename
        self.line = line
        self.stages = []
        self.flags = set()             # every valueless material keyword
        self.editor_image = None       # str path from qer_editorimage
        self.description = ''
        self.sort = None               # float, once resolved
        self.sort_name = ''
        self.coverage = None
        self.coverage_writes = []      # every explicit write, in source order
        self.cull = 'front'            # front|two-sided|back
        self.polygon_offset = 0.0
        self.spectrum = 0
        self.gui_surf = ''
        self.light_falloff = None      # MtrImage
        self.light_ambient = {}        # TDM lightAmbientDiffuse/Specular
        self.deform = None             # (type, [MtrExpr], decl_name)
        self.decal_info = []
        self.material_type = ''        # Quake 4 materialType
        self.renderbump = ''
        self.kind = SURFACE_MATERIAL
        self.failed = False            # engine would MakeDefault() this
        self.diagnostics = []
        self.raw = {}                  # every keyword seen, for the report

    # -- convenience ---------------------------------------------------------
    def stage_by_lighting(self, lighting):
        for st in self.stages:
            if st.lighting == lighting:
                return st
        return None

    @property
    def is_light(self):
        return self.kind != SURFACE_MATERIAL

    @property
    def ambient_stages(self):
        return [s for s in self.stages if s.lighting == 'ambient']

    @property
    def interaction_stages(self):
        return [s for s in self.stages if s.lighting != 'ambient']

    def first_image_path(self):
        """Fallback chain: editor image, then the first real stage texture."""
        if self.editor_image:
            return self.editor_image
        for st in self.stages:
            if st.image is not None and st.tex_kind == TEX_FILE:
                path = st.image.base_path()
                if path:
                    return path
        for st in self.stages:
            for _idx, (_opts, img) in sorted(st.fragment_maps.items()):
                path = img.base_path()
                if path:
                    return path
        return None

    def __repr__(self):
        return '<MtrMaterial %s %d stages>' % (self.name, len(self.stages))


class MtrTable(object):
    __slots__ = ('name', 'snap', 'clamp', 'values')

    def __init__(self, name, snap, clamp, values):
        self.name = name
        self.snap = snap
        self.clamp = clamp
        self.values = values

    def lookup(self, index):
        """idDeclTable::TableLookup."""
        vals = self.values
        count = len(vals)
        if not count:
            return 0.0
        index = float(index)
        if self.clamp:
            if index < 0.0:
                index = 0.0
            elif index > 1.0:
                index = 1.0
            fi = index * (count - 1)
        else:
            fi = (index * count) % count
            if fi < 0.0:
                fi += count
        lo = int(fi)
        if self.snap:
            return vals[lo % count]
        frac = fi - lo
        return vals[lo % count] * (1.0 - frac) + vals[(lo + 1) % count] * frac

    def __repr__(self):
        return '<table %s n=%d%s%s>' % (self.name, len(self.values),
                                        ' snap' if self.snap else '',
                                        ' clamp' if self.clamp else '')


def _parse_table(toks, decl):
    """table <name> { [snap] [clamp] { v, v, ... } }"""
    src = _Cursor(toks, decl.start, decl.end)
    snap = clamp = False
    values = []
    while True:
        tk = src.read()
        if tk is None:
            break
        low = tk.val.lower()
        if low == 'snap':
            snap = True
        elif low == 'clamp':
            clamp = True
        elif tk.val == '-':
            nxt = src.read()
            values.append(-_to_float(nxt.val) if nxt else 0.0)
        elif tk.type == TT_NUMBER:
            values.append(_to_float(tk.val))
    # A table is a decl: registered under MakeNameCanonical, the same as a
    # material. See the table-lookup branch in _parse_term.
    return MtrTable(engine_canonical_decl(decl.name), snap, clamp, values)


# ---------------------------------------------------------------------------
# Value reading
# ---------------------------------------------------------------------------

def _read_num_or_vec3(src):
    """TDM ParseNumberOrVec3: one scalar, or three."""
    out = []
    while len(out) < 3:
        tk = src.peek()
        if tk is None:
            break
        if tk.val == '-':
            src.read()
            nxt = src.read()
            out.append(-_to_float(nxt.val) if nxt else 0.0)
        elif tk.type == TT_NUMBER:
            src.read()
            out.append(_to_float(tk.val))
        else:
            break
        if len(out) == 1:
            nxt = src.peek()
            if nxt is None or (nxt.type != TT_NUMBER and nxt.val != '-'):
                break
    return out


def _read_float(src):
    """idLexer::ParseFloat, which accepts a leading '-' as its own token."""
    tk = src.read()
    if tk is None:
        return 0.0
    if tk.val == '-':
        nxt = src.read()
        return -_to_float(nxt.val) if nxt else 0.0
    return _to_float(tk.val)


def _read_value(src, state, kind, keyword):
    """Consume one keyword's argument(s) and return a normalised value."""
    if kind == 'flag':
        return True

    if kind == 'tok':
        tk = src.read()
        return tk.val if tk else ''

    if kind == 'tok_line':
        tk = src.read_on_line()
        return tk.val if tk else ''

    if kind == 'tok_line_rest':
        tk = src.read_on_line()
        value = tk.val if tk else ''
        src.skip_rest_of_line()
        return value

    if kind == 'rest':
        parts = []
        while True:
            tk = src.read_on_line()
            if tk is None:
                break
            parts.append(tk.val)
        return ' '.join(parts)

    if kind == 'image':
        return _parse_image(src, state)

    if kind == 'expr':
        return _parse_expr(src, state)

    if kind in ('expr2', 'expr3', 'expr4'):
        want = int(kind[-1])
        vals = [_parse_expr(src, state)]
        while len(vals) < want:
            if not _expect(src, state, ','):
                break
            vals.append(_parse_expr(src, state))
        while len(vals) < want:
            vals.append(MtrExpr.const(0.0))
        return vals

    if kind == 'expr_opt':
        if src.read_on_line() is None:
            return None
        src.unread()
        return _parse_expr(src, state)

    if kind == 'float_opt':
        if src.read_on_line() is None:
            return 1.0
        src.unread()
        return _read_float(src)

    if kind == 'num_or_vec3':
        return _read_num_or_vec3(src)

    if kind == 'sort':
        tk = src.read_on_line()
        if tk is None:
            state.note(DIAG_CONTENT, 'sort',
                       'sort keyword with no value', src.line())
            return None
        if tk.type == TT_NUMBER:
            return _to_float(tk.val)
        low = tk.val.lower()
        if low not in SORT_VALUES:
            state.note(DIAG_CONTENT, 'sort',
                       'unknown sort value "%s"' % tk.val, tk.line)
            return None
        return low

    if kind == 'texgen':
        tk = src.read()
        if tk is None:
            return None
        low = tk.val.lower()
        if low not in TEXGEN_VALUES:
            state.fail('texgen', 'unknown texGen "%s"' % tk.val, tk.line)
            return None
        if low == 'wobblesky':
            return (low, [_parse_expr(src, state) for _ in range(3)])
        return (low, [])

    if kind == 'deform':
        tk = src.read()
        if tk is None:
            return None
        low = tk.val.lower()
        spec = DEFORM_VALUES.get(low)
        if spec is None:
            state.note(DIAG_UNSUPPORTED, 'deform',
                       'unknown deform type "%s"' % tk.val, tk.line)
            src.skip_rest_of_line()
            return (low, [], '')
        if spec == 'decl':
            d = src.read()
            return (low, [], d.val if d else '')
        if spec == 'turbulent':
            d = src.read()
            return (low, [_parse_expr(src, state) for _ in range(3)],
                    d.val if d else '')
        if spec == 'rest':
            src.skip_rest_of_line()
            return (low, [], '')
        return (low, [_parse_expr(src, state) for _ in range(spec)], '')

    if kind == 'decalinfo':
        # decalInfo <stayTime> <fadeTime> [ ( r,g,b,a ) ( r,g,b,a ) ]
        # Quake 4 writes the two times comma-separated ("decalinfo 10, 0.5"),
        # Doom 3 space-separated; accept either.
        vals = [_read_float(src)]
        nxt = src.peek()
        if nxt is not None and nxt.val == ',':
            src.read()
        vals.append(_read_float(src))
        for _ in range(2):
            tk = src.peek()
            if tk is None or tk.val != '(':
                break
            src.read()
            for _c in range(4):
                nxt = src.peek()
                if nxt is not None and nxt.val == ',':
                    src.read()
                vals.append(_read_float(src))
            _expect(src, state, ')', fatal=False)
        return vals

    if kind == 'blend':
        tk = src.read()
        if tk is None:
            return None
        low = tk.val.lower()
        if low in BLEND_SHORTHANDS or low in BLEND_LIGHTING:
            return (low, None, None)
        nxt = src.peek()
        if nxt is not None and nxt.val == ',':
            src.read()
            dst = src.read()
            dst_low = dst.val.lower() if dst else 'gl_zero'
            for factor, label in ((low, 'source'), (dst_low, 'destination')):
                if factor not in _GL_BLEND_FACTORS:
                    state.note(DIAG_CONTENT, 'blend',
                               'unknown GL %s blend factor "%s"'
                               % (label, factor), tk.line)
            return (None, low, dst_low)
        # A bare GL factor with no comma: the engine's NameToSrcBlendMode
        # warns and leaves the destination at its default.
        state.note(DIAG_CONTENT, 'blend',
                   'blend "%s" has no destination factor' % tk.val, tk.line)
        return (None, low, 'gl_zero')

    if kind == 'videomap':
        tk = src.read()
        if tk is None:
            return ('', False)
        if tk.val.lower() == 'loop':
            nxt = src.read()
            return (nxt.val if nxt else '', True)
        return (tk.val, False)

    if kind == 'indexed_parm':
        # vertexParm <n> <e> [, <e> [, <e> [, <e>]]]  -- commas only on-line
        idx = src.read()
        vals = [_parse_expr(src, state)]
        while len(vals) < 4:
            nxt = src.read_on_line()
            if nxt is None:
                break
            if nxt.val != ',':
                src.unread()
                break
            vals.append(_parse_expr(src, state))
        return (int(_to_float(idx.val)) if idx else 0, vals)

    if kind == 'named_parm':
        # Quake 4: shaderParm <name> <e> [, <e> [, <e> [, <e>]]]
        nm = src.read()
        vals = [_parse_expr(src, state)]
        while len(vals) < 4:
            nxt = src.read_on_line()
            if nxt is None:
                break
            if nxt.val != ',':
                src.unread()
                break
            vals.append(_parse_expr(src, state))
        return (nm.val if nm else '', vals)

    if kind in ('indexed_map', 'named_texture'):
        key_tok = src.read()
        opts = []
        while True:
            tk = src.peek()
            if tk is None or tk.val.lower() not in _FRAGMENTMAP_OPTIONS:
                break
            opts.append(src.read().val.lower())
        img = _parse_image(src, state)
        if kind == 'indexed_map':
            return (int(_to_float(key_tok.val)) if key_tok else 0, opts, img)
        return (key_tok.val if key_tok else '', opts, img)

    if kind == 'frob_texture':
        img = _parse_image(src, state)
        return (img, _read_num_or_vec3(src), _read_num_or_vec3(src))

    if kind == 'frob_diffuse':
        return (None, _read_num_or_vec3(src), _read_num_or_vec3(src))

    if kind == 'braced':                           # BFG `stencil { ... }`
        src.skip_braced()
        return True

    raise AssertionError('unhandled value kind %r for %r' % (kind, keyword))


# ---------------------------------------------------------------------------
# Stage parsing - idMaterial::ParseStage
# ---------------------------------------------------------------------------

_COLOR_CHANNEL = {'red': 0, 'green': 1, 'blue': 2, 'alpha': 3}
_MASK_KEYWORDS = {'maskred': 'red', 'maskgreen': 'green', 'maskblue': 'blue',
                  'maskalpha': 'alpha', 'maskcolor': 'color',
                  'maskdepth': 'depth'}
_WRAP_KEYWORDS = {'clamp': WRAP_CLAMP, 'noclamp': WRAP_REPEAT,
                  'zeroclamp': WRAP_ZERO, 'alphazeroclamp': WRAP_ZERO_ALPHA}
_DYNAMIC_MAPS = {
    'mirrorrendermap': 'mirror render target',
    'remoterendermap': 'remote camera render target',
    'xrayrendermap': 'x-ray render target',
    'guirendermap': 'in-game GUI render target',
    'portalrendermap': 'portal render target',
    'rendertargetmap': 'render target',
}
_SCREEN_TEXGENS = frozenset(('screen', 'screen2', 'glasswarp'))


def _parse_stage(src, state, default_wrap=WRAP_REPEAT):
    """Parse one `{ ... }` stage body. The opening brace is already consumed."""
    stage = MtrStage()
    stage.wrap = default_wrap
    stage.source_line = src.line()

    while True:
        if state.failed:
            return stage
        tk = src.read()
        if tk is None:
            state.fail('truncated', 'end of file inside a stage',
                       src.line())
            return stage
        if tk.val == '}':
            return stage

        low = tk.val.lower()
        kind = MTR_STAGE_KEYWORDS.get(low)

        if kind is None:
            # Unknown keyword: skip its line and keep going. Real .mtr content
            # is line-oriented even though the lexer is not, so this recovers
            # cleanly from both valueless flags and keywords with arguments -
            # and it is what lets one merged keyword table cope with engine
            # variants neither we nor the source trees know about.
            state.note(DIAG_UNKNOWN_KW, low,
                       'unknown stage keyword "%s"' % tk.val, tk.line)
            src.skip_rest_of_line()
            continue

        # Note what we are about to read an argument for, so a failure deep
        # inside the expression parser can name the keyword responsible.
        state.begin_argument(tk.val, tk.line, src.i)
        value = _read_value(src, state, kind, low)
        stage.flags.add(low)

        if low == 'blend':
            if value is None:
                continue
            shorthand, gl_src, gl_dst = value
            if shorthand is not None:
                stage.blend_name = shorthand
                if shorthand in BLEND_LIGHTING:
                    stage.lighting = BLEND_LIGHTING[shorthand]
                    stage.blend_src, stage.blend_dst = 'gl_one', 'gl_zero'
                    if shorthand == 'shader':
                        stage.unsupported.append(
                            ('blend shader',
                             'Prey custom-shader stage'))
                else:
                    stage.blend_src, stage.blend_dst = \
                        BLEND_SHORTHANDS[shorthand]
            else:
                stage.blend_src, stage.blend_dst = gl_src, gl_dst

        elif low in ('map', 'cubemapsingle', 'uncompressedcubemap',
                     'stereomap'):
            stage.image = value
            stage.tex_kind = (TEX_BUILTIN if value.is_builtin() else TEX_FILE)

        elif low in ('cubemap', 'cameracubemap'):
            stage.image = value
            stage.tex_kind = TEX_CUBE
            stage.tex_detail = low

        elif low in _DYNAMIC_MAPS:
            stage.tex_kind = TEX_DYNAMIC
            stage.tex_detail = low
            stage.unsupported.append((low, _DYNAMIC_MAPS[low]))
            if low in ('mirrorrendermap', 'xrayrendermap', 'portalrendermap'):
                stage.texgen = 'screen'

        elif low in ('videomap', 'soundmap'):
            stage.tex_kind = TEX_VIDEO
            stage.tex_detail = low
            stage.unsupported.append(
                (low, 'video texture' if low == 'videomap'
                 else 'sound-driven texture'))

        elif low in ('screen', 'screen2', 'glasswarp'):
            stage.texgen = low

        elif low == 'texgen':
            if value is not None:
                stage.texgen, stage.texgen_args = value

        elif low in ('scroll', 'translate', 'scale', 'centerscale', 'shear'):
            stage.transforms.append(MtrTexTransform(low, value[0], value[1]))

        elif low == 'rotate':
            stage.transforms.append(MtrTexTransform('rotate', value))

        elif low in _COLOR_CHANNEL:
            stage.color[_COLOR_CHANNEL[low]] = value

        elif low == 'rgb':
            stage.color[0] = stage.color[1] = stage.color[2] = value

        elif low == 'rgba':
            stage.color[0] = stage.color[1] = stage.color[2] = value
            stage.color[3] = value

        elif low == 'color':
            stage.color = list(value)

        elif low == 'colored':
            # Shorthand for rgba = parm0..parm3.
            stage.colored = True
            for i in range(4):
                stage.color[i] = MtrExpr(EXPR_VAR, 'parm%d' % i,
                                         source='parm%d' % i)

        elif low == 'alphatest':
            stage.alpha_test = value

        elif low == 'if':
            stage.condition = value

        elif low == 'vertexcolor':
            stage.vertex_color = 'modulate'

        elif low == 'inversevertexcolor':
            stage.vertex_color = 'inverse'

        elif low in _WRAP_KEYWORDS:
            stage.wrap = _WRAP_KEYWORDS[low]

        elif low in ('nearest', 'linear'):
            stage.filter = low

        elif low in _MASK_KEYWORDS:
            stage.mask.add(_MASK_KEYWORDS[low])

        elif low == 'privatepolygonoffset':
            stage.private_polygon_offset = value

        elif low in ('program', 'fragmentprogram', 'vertexprogram',
                     'glslprogram', 'megatexture'):
            stage.programs[low] = value
            stage.unsupported.append(
                (low, 'custom %s "%s"'
                 % ('shader program' if low != 'megatexture'
                    else 'megatexture', value)))

        elif low in ('vertexparm', 'vertexparm2', 'fragmentparm'):
            stage.parms[(low, value[0])] = value[1]

        elif low == 'shaderparm':
            stage.parms[(low, value[0])] = value[1]

        elif low in ('fragmentmap', 'shadertexture'):
            stage.fragment_maps[value[0]] = (value[1], value[2])

        elif low == 'name':
            stage.name = value

    return stage


# ---------------------------------------------------------------------------
# Material parsing - idMaterial::ParseMaterial
# ---------------------------------------------------------------------------

# Material keywords whose value is an image program and which the engine turns
# into a synthetic interaction stage (`blend <kw>\nmap <image>\n}`).
_IMPLICIT_STAGE_KEYWORDS = {
    'diffusemap': 'diffuse', 'bumpmap': 'bump', 'specularmap': 'specular',
    'parallaxmap': 'parallax',
    'basecolormap': 'diffuse', 'normalmap': 'bump',
    'rmaomap': 'specular', 'pbrmap': 'specular', 'reflectionmap': 'specular',
}

# Material-scope keywords that assign idMaterial::coverage as they are read.
#
# Only the ones BOTH engines agree on are listed. Raven's Material.cpp also
# writes coverage from DECAL_MACRO, decal_alphatest_macro and scorch_macro
# (translucent) and from skybox_macro and directportal (opaque); stock Doom 3
# writes none of those - its DECAL_MACRO sets polygonOffset, SURF_DISCRETE,
# sort and noShadows and stops. Adding them here would be a dialect guess
# applied to every base at once, so they stay out until the parser can tell
# which engine a .mtr was written for.
_COVERAGE_KEYWORDS = {
    'translucent': COVERAGE_TRANSLUCENT,
    'forceopaque': COVERAGE_OPAQUE,
    'mirror':      COVERAGE_OPAQUE,
}

# Macro keywords, expanded exactly as the engine does.
_MACROS = {
    'decal_macro':            dict(polygon_offset=1.0, sort='decal',
                                   flags=('discrete', 'nonsolid', 'noshadows')),
    'twosided_decal_macro':   dict(polygon_offset=1.0, sort='decal',
                                   flags=('discrete', 'noimpact', 'nonsolid',
                                          'noshadows'), cull='two-sided'),
    'decal_alphatest_macro':  dict(polygon_offset=1.0, sort='decal',
                                   flags=('discrete', 'nonsolid', 'noshadows')),
    'overlay_macro':          dict(polygon_offset=1.0, sort='decal',
                                   flags=('discrete', 'nonsolid', 'noshadows')),
    'scorch_macro':           dict(polygon_offset=1.0, sort='decal',
                                   flags=('discrete', 'nonsolid', 'noshadows')),
    'glass_macro':            dict(flags=('noshadows', 'translucent')),
    'particle_macro':         dict(flags=('noshadows', 'nonsolid',
                                          'translucent')),
    'skybox_macro':           dict(flags=('noshadows', 'nonsolid')),
}

_CULL_FLAGS = {'twosided': 'two-sided', 'backsided': 'back'}

# Keywords that mark this decl as a light shader rather than a surface shader.
_LIGHT_KEYWORDS = {
    'ambientlight': LIGHT_AMBIENT, 'cubiclight': LIGHT_AMBIENT,
    'foglight': LIGHT_FOG, 'blendlight': LIGHT_BLEND,
}


def _parse_material(toks, decl, tables, diags):
    """Parse one material decl body into an MtrMaterial."""
    mat = MtrMaterial(strip_material_extension(decl.name), decl.filename,
                      decl.line)
    state = _ParseState(tables, diags, decl.filename, mat.name)
    src = _Cursor(toks, decl.start, decl.end)
    default_wrap = WRAP_REPEAT

    while True:
        if state.failed:
            break
        tk = src.read()
        if tk is None:
            break

        if tk.val == '{':
            stage = _parse_stage(src, state, default_wrap)
            mat.stages.append(stage)
            if stage.alpha_test is not None:
                # ParseStage's `alphaTest` assigns coverage = MC_PERFORATED
                # then and there, so a stage block competes for coverage with
                # the material-scope keywords by POSITION, not by priority.
                mat.coverage_writes.append(COVERAGE_PERFORATED)
            continue

        low = tk.val.lower()
        kind = MTR_MATERIAL_KEYWORDS.get(low)

        if kind is None:
            state.note(DIAG_UNKNOWN_KW, low,
                       'unknown material keyword "%s"' % tk.val, tk.line)
            src.skip_rest_of_line()
            continue

        # Note what we are about to read an argument for, so a failure deep
        # inside the expression parser can name the keyword responsible.
        state.begin_argument(tk.val, tk.line, src.i)
        value = _read_value(src, state, kind, low)
        mat.raw[low] = value
        if kind == 'flag':
            mat.flags.add(low)
        if low in _COVERAGE_KEYWORDS:
            mat.coverage_writes.append(_COVERAGE_KEYWORDS[low])

        if low in _IMPLICIT_STAGE_KEYWORDS:
            stage = MtrStage()
            stage.lighting = _IMPLICIT_STAGE_KEYWORDS[low]
            stage.blend_name = low
            stage.image = value
            stage.tex_kind = TEX_BUILTIN if value.is_builtin() else TEX_FILE
            stage.wrap = default_wrap
            stage.source_line = tk.line
            mat.stages.append(stage)

        elif low == 'qer_editorimage':
            mat.editor_image = value

        elif low == 'description':
            mat.description = value

        elif low == 'sort':
            if value is not None:
                mat.sort_name = value if isinstance(value, str) else ''
                mat.sort = SORT_VALUES[value] if isinstance(value, str) \
                    else float(value)

        elif low == 'spectrum':
            try:
                mat.spectrum = int(float(value))
            except (TypeError, ValueError):
                state.note(DIAG_CONTENT, 'spectrum',
                           'spectrum value "%s" is not a number' % value,
                           tk.line)

        elif low == 'guisurf':
            mat.gui_surf = value

        elif low == 'lightfalloffimage':
            mat.light_falloff = value

        elif low in ('lightambientdiffuse', 'lightambientspecular'):
            mat.light_ambient[low] = value

        elif low == 'deform':
            mat.deform = value
            if value and value[0] in _DEFORM_TWOSIDED:
                mat.cull = 'two-sided'
                mat.flags.add('noshadows')

        elif low == 'decalinfo':
            mat.decal_info = value

        elif low == 'polygonoffset':
            mat.polygon_offset = value

        elif low == 'materialtype':
            mat.material_type = value

        elif low == 'renderbump':
            mat.renderbump = value

        elif low == 'mirror':
            # `mirror` is two assignments, not one: sort = SS_SUBVIEW as well
            # as coverage = MC_OPAQUE. The sort is what keeps the surface
            # visible - RB_T_FillDepthBuffer down-modulates a SS_SUBVIEW
            # surface instead of filling it black, so the reflection the
            # subview pass drew is still there when the ambient stages run.
            mat.sort_name, mat.sort = 'subview', SORT_VALUES['subview']

        elif low in _CULL_FLAGS:
            mat.cull = _CULL_FLAGS[low]

        elif low in _WRAP_KEYWORDS and low != 'noclamp':
            # Material-scope clamp keywords set the *default* for every stage
            # that does not override it (the trpDefault argument to ParseStage).
            # The global `alphazeroclamp` is not the same keyword as the
            # per-stage one: idMaterial::Parse sets trpDefault to
            # TR_CLAMP_TO_ZERO for it, while ParseStage's own alphazeroclamp
            # gives TR_CLAMP_TO_ZERO_ALPHA. Reproduce the engine, quirk and all.
            default_wrap = (WRAP_ZERO if low == 'alphazeroclamp'
                            else _WRAP_KEYWORDS[low])

        elif low in _MACROS:
            spec = _MACROS[low]
            mat.flags.update(spec.get('flags', ()))
            if 'sort' in spec and mat.sort is None:
                mat.sort_name = spec['sort']
                mat.sort = SORT_VALUES[spec['sort']]
            if 'polygon_offset' in spec:
                mat.polygon_offset = spec['polygon_offset']
            if 'cull' in spec:
                mat.cull = spec['cull']

        elif low in ('frobstage_texture', 'frobstage_diffuse'):
            mat.raw['frobstage'] = value

    mat.failed = state.failed
    return mat


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


def strip_material_extension(name):
    """The DISPLAY form of a material name: the last-dot rule, nothing else.

    Material names in models and .map files routinely carry a texture
    extension the .mtr does not, so both sides are truncated the same way.

    This is deliberately NOT the identity - engine_canonical_decl is, and it
    also folds case and backslashes. Around a thousand decls across the five
    corpora are declared with uppercase in them and 62 with backslashes;
    canonicalising the display name would churn every report line and every
    ALL_SOURCE datablock name to fix nothing. Identity canonical, display
    verbatim: the registry and every lookup go through the block above, and
    MtrMaterial.name keeps the spelling the .mtr used.
    """
    if not name:
        return name
    idx = name.rfind('.')
    return name[:idx] if idx != -1 else name


# ---------------------------------------------------------------------------
# Semantics pass - the tail of idMaterial::Parse
# ---------------------------------------------------------------------------
# Everything below is derived state the engine computes after parsing. It is
# what decides whether a material is a cutout or blended, what it sorts as,
# and whether a bump-only material renders lit-white or black - so it has to
# happen here rather than being guessed at by the node builder.

# stageLighting_t, with its enum VALUES, because SortInteractionStages sorts
# on them numerically and SL_AMBIENT is 0 - ahead of every lit stage.
_LIGHTING_ORDER = {'ambient': 0, 'bump': 1, 'parallax': 1, 'diffuse': 2,
                   'specular': 3}


def _add_implicit_stages(mat):
    """idMaterial::AddImplicitStages.

    A material with any interaction stage gets the missing ones filled in
    with _flat / _white. Without this, a bump-only material would render with
    no diffuse contribution at all, which is not what the game shows.
    """
    has = {st.lighting for st in mat.stages}
    if not (has & {'bump', 'diffuse', 'specular', 'parallax'}):
        return
    has_reflection = any(st.texgen == 'reflect' for st in mat.stages)

    def implicit(lighting, image):
        st = MtrStage()
        st.lighting = lighting
        st.blend_name = lighting + 'map'
        st.image = MtrImage(path=image, canonical=image)
        st.tex_kind = TEX_BUILTIN
        st.implicit = True
        return st

    # Exactly what the engine adds, and no more: a flat normal when there is
    # no bump stage, and a white diffuse when there is nothing else to light.
    # There is deliberately no implicit specular - inventing one would make
    # every unspecular material build a glossy lobe out of black.
    if 'bump' not in has and 'parallax' not in has:
        mat.stages.append(implicit('bump', '_flat'))
    if 'diffuse' not in has and 'specular' not in has and not has_reflection:
        mat.stages.append(implicit('diffuse', '_white'))


def _sort_interaction_stages(mat):
    """idMaterial::SortInteractionStages, ported literally.

    This used to be "split the stage list on ambient stages and sort each run
    into bump/diffuse/specular order", which is neither of the two things the
    engine does:

      * Ambient stages are NOT boundaries. The engine bubble-sorts on the raw
        stageLighting_t enum, and SL_AMBIENT is 0, so an ambient stage caught
        inside a group is moved to the FRONT of it, ahead of the bump. The
        group boundary is the next SL_BUMP and nothing else.
      * That boundary has an exception: "if the very first stage wasn't a
        bumpmap, this bumpmap is part of the first group". A group that opens
        on a diffuse swallows the next bump instead of ending at it.

    Both matter, because RB_CreateSingleDrawInteractions walks the sorted
    list and its accumulator is order-sensitive. models/mapobjects/hellcages/
    HellCagechain_ is `diffusemap, <ambient>, <implicit _flat bump>`: the old
    split left the implicit bump stranded after the diffuse, so the only
    interaction the engine flushes had no bump image and RB_SubmittInteraction
    would have dropped it. The engine's own sort puts the ambient first and
    the bump ahead of the diffuse, and the surface lights normally.

    3335 materials across the five game bases are ordered differently by the
    two versions, and 135 of them end up with a different number of
    interaction passes.

    The relative order of the ambient stages themselves is untouched: a
    bubble sort only swaps on a strict >, so two SL_AMBIENT stages never pass
    each other, and the ambient pass draws them in the same sequence as
    before.
    """
    stages = list(mat.stages)
    count = len(stages)

    def rank(stage):
        return _LIGHTING_ORDER.get(stage.lighting, 0)

    i = 0
    while i < count:
        # Find the next bumpmap, which opens the following group - unless
        # this group did not itself open on a bumpmap, in which case that
        # bumpmap belongs to this one.
        j = i + 1
        while j < count:
            if rank(stages[j]) == 1:
                if rank(stages[i]) != 1:
                    j += 1
                    continue
                break
            j += 1
        for length in range(1, j - i):
            for k in range(i, j - length):
                if rank(stages[k]) > rank(stages[k + 1]):
                    stages[k], stages[k + 1] = stages[k + 1], stages[k]
        i = j
    mat.stages = stages


def _classify(mat):
    """Decide surface vs light material, and record why."""
    for flag, kind in _LIGHT_KEYWORDS.items():
        if flag in mat.flags:
            mat.kind = kind
            return
    # Canonical, not .lower(): the prefix test below is a decl-name test,
    # and a decl written "lights\foo" is in the lights/ tree.
    lowered = engine_canonical_decl(mat.name)
    if mat.light_falloff is not None or mat.light_ambient:
        mat.kind = LIGHT_AMBIENT if mat.light_ambient else LIGHT_PROJECTED
        return
    if lowered.startswith('lights/') or lowered.startswith('fogs/'):
        mat.kind = LIGHT_PROJECTED


def _resolve_coverage(mat):
    """idMaterial::Parse - coverage, sort defaults and the flags they imply."""
    ambient = mat.ambient_stages
    drawn = [st for st in mat.stages if not st.is_invisible()]

    has_alpha_test = any(st.alpha_test is not None for st in mat.stages)

    if mat.coverage_writes:
        # coverage starts MC_BAD and every writer assigns it directly, so the
        # LAST write in the decl wins and the block below never runs - it is
        # guarded by `if (coverage == MC_BAD)`. Priority order gets this
        # wrong whenever an alphaTest stage follows `translucent`, which is
        # how hangingwires2sided, the Prey cacti and 90 others are written:
        # the engine cuts them out, and reading `translucent` as the winner
        # drew the whole quad, background texels and all.
        mat.coverage = mat.coverage_writes[-1]
    elif 'translucent' in mat.flags:
        # No keyword wrote coverage, so this is a macro-injected flag
        # (glass_macro, particle_macro) rather than the `translucent` keyword.
        mat.coverage = COVERAGE_TRANSLUCENT
    elif not mat.stages:
        mat.coverage = COVERAGE_TRANSLUCENT          # non-visible
    elif len(ambient) != len(mat.stages):
        # There is an interaction draw, so the surface is lit and solid -
        # unless a diffuse stage carries an alphaTest, which makes it a
        # perforated cutout rather than a blended surface.
        mat.coverage = COVERAGE_PERFORATED if has_alpha_test \
            else COVERAGE_OPAQUE
    elif drawn and drawn[0].blends_with_destination():
        mat.coverage = COVERAGE_TRANSLUCENT
    elif has_alpha_test:
        mat.coverage = COVERAGE_PERFORATED
    else:
        mat.coverage = COVERAGE_OPAQUE

    # Anything sampling _currentRender is forced to post-process/translucent.
    for st in mat.stages:
        paths = []
        if st.image is not None:
            paths.extend(p.lower() for p in st.image.all_paths() if p)
        for _o, img in st.fragment_maps.values():
            paths.extend(p.lower() for p in img.all_paths() if p)
        if '_currentrender' in paths:
            if mat.sort_name != 'portalsky':
                mat.sort = SORT_VALUES['postprocess']
                mat.sort_name = 'postprocess'
                mat.coverage = COVERAGE_TRANSLUCENT
            break

    if mat.coverage == COVERAGE_TRANSLUCENT:
        mat.flags.add('noshadows')      # translucent automatically implies it

    if mat.sort is None:
        if mat.polygon_offset:
            mat.sort_name, mat.sort = 'decal', SORT_VALUES['decal']
        elif mat.coverage == COVERAGE_TRANSLUCENT:
            mat.sort_name, mat.sort = 'medium', SORT_VALUES['medium']
        else:
            mat.sort_name, mat.sort = 'opaque', SORT_VALUES['opaque']


def _collect_stage_diagnostics(mat, diags):
    """Promote each stage's unsupported-feature notes into diagnostics."""
    for index, st in enumerate(mat.stages):
        for kind, message in st.unsupported:
            diags.append(MtrDiagnostic(
                DIAG_UNSUPPORTED, kind,
                'stage %d: %s' % (index, message),
                mat.filename, st.source_line or mat.line, mat.name))
        if st.texgen in _SCREEN_TEXGENS:
            diags.append(MtrDiagnostic(
                DIAG_APPROXIMATED, 'texgen ' + st.texgen,
                'stage %d: screen-space texture coordinates approximated with '
                'window coordinates' % index,
                mat.filename, st.source_line or mat.line, mat.name))
        elif st.texgen in ('reflect', 'skybox', 'wobblesky'):
            diags.append(MtrDiagnostic(
                DIAG_APPROXIMATED, 'texgen ' + st.texgen,
                'stage %d: environment-mapped texture coordinates '
                'approximated' % index,
                mat.filename, st.source_line or mat.line, mat.name))
        for expr in _stage_expressions(st):
            if expr_grouping_is_surprising(expr):
                diags.append(MtrDiagnostic(
                    DIAG_APPROXIMATED, 'expression-grouping',
                    'stage %d: "%s" groups right-to-left in idTech 4 '
                    '(engine-exact, not a typo)' % (index, expr.source),
                    mat.filename, st.source_line or mat.line, mat.name))
                break
    if mat.gui_surf:
        diags.append(MtrDiagnostic(
            DIAG_UNSUPPORTED, 'guisurf',
            'GUI surface "%s" - interactive GUIs are not reproducible'
            % mat.gui_surf, mat.filename, mat.line, mat.name))
    if mat.deform:
        diags.append(MtrDiagnostic(
            DIAG_UNSUPPORTED, 'deform ' + mat.deform[0],
            'geometry deform "%s" is not a shading effect' % mat.deform[0],
            mat.filename, mat.line, mat.name))


def _stage_expressions(st):
    for c in st.color:
        if c is not None:
            yield c
    if st.alpha_test is not None:
        yield st.alpha_test
    if st.condition is not None:
        yield st.condition
    for t in st.transforms:
        if t.x is not None:
            yield t.x
        if t.y is not None:
            yield t.y


def finish_material(mat, diags):
    """Run the engine's post-parse pass over a freshly parsed material."""
    if mat.failed:
        return mat
    _add_implicit_stages(mat)
    _sort_interaction_stages(mat)
    _classify(mat)
    _resolve_coverage(mat)
    _collect_stage_diagnostics(mat, diags)
    return mat


# ---------------------------------------------------------------------------
# Database - one parsed .mtr tree
# ---------------------------------------------------------------------------

class MtrDatabase(object):
    """Everything parsed from a materials tree: materials, tables, guides,
    and the diagnostics produced along the way."""

    def __init__(self):
        self.materials = {}          # lowercased name -> MtrMaterial
        self.material_order = []     # names in first-seen order
        self.tables = {}             # lowercased name -> MtrTable
        self.guides = {}             # lowercased name -> MtrGuide
        self.diagnostics = []
        self.files = []
        self.skipped_redefinitions = 0
        # Of those, the ones that are a higher-priority root deliberately
        # replacing a lower one rather than a genuine duplicate - see
        # add_material. Counted, never warned about.
        self.mod_overrides = 0
        # The asset roots this database was parsed under, highest priority
        # first - one entry normally, two when a Mod Base is configured.
        # Reports quote it so the search order is stated up front rather
        # than inferred from which material happened to win.
        self.roots = []

    # -- lookup --------------------------------------------------------------
    def find(self, name):
        """The engine's own lookup: canonicalise, then index.

        The registry is keyed on engine_canonical_decl, so this is the whole
        of it. There used to be a second, case-only attempt here; with
        canonical keys it is unreachable - a canonical key never holds a dot,
        a backslash or an uppercase letter, so anything `name.lower()` could
        match, the canonical form matches first.
        """
        if not name:
            return None
        return self.materials.get(engine_canonical_decl(name))

    def __len__(self):
        return len(self.materials)

    def __contains__(self, name):
        return self.find(name) is not None

    # -- construction --------------------------------------------------------
    def _root_index(self, path):
        """Which configured root `path` lives under, or None.

        Used only to tell a redefinition apart from a mod override; both
        look identical from inside one parse.
        """
        if not path or not self.roots:
            return None
        norm = _os.path.normcase(_os.path.normpath(path))
        for i, root in enumerate(self.roots):
            prefix = _os.path.normcase(_os.path.normpath(root))
            if norm == prefix or norm.startswith(prefix + _os.sep):
                return i
        return None

    def add_material(self, mat):
        # Registration and lookup both canonicalise, which is the only way
        # either helps: idDeclManagerLocal::CreateNewDecl canonicalises the
        # name it registers under and every FindType canonicalises the name
        # it asks for. 62 decls across the corpora are declared with
        # backslashes and are unreachable by their forward-slash spelling
        # without this - and zero of the 29,609 collide once canonicalised,
        # so it costs nothing.
        key = engine_canonical_decl(mat.name)
        if key in self.materials:
            # idDeclFile::LoadAndParse warns "previously defined at ..." and
            # KEEPS THE FIRST definition. The old importer kept the last one,
            # which silently changed 225 Doom 3 materials.
            first = self.materials[key]
            self.skipped_redefinitions += 1
            # A Mod Base makes "the same material declared twice" the normal
            # case rather than a fault: the mod's tree is parsed first
            # precisely so its copy wins, which is what the engine's fs_game
            # order does too. Reporting each one would bury the real
            # redefinitions - a mod overriding two hundred materials would
            # emit two hundred warnings about working correctly - so a
            # cross-root override is counted and passed over in silence. The
            # search order is stated once, up front, in the report instead.
            if self._root_index(first.filename) == self._root_index(mat.filename):
                self.diagnostics.append(MtrDiagnostic(
                    DIAG_CONTENT, 'redefinition',
                    'redefined; keeping the first definition from %s:%d'
                    % (first.filename, first.line),
                    mat.filename, mat.line, mat.name))
            else:
                self.mod_overrides += 1
            return False
        self.materials[key] = mat
        self.material_order.append(key)
        return True

    # -- reporting -----------------------------------------------------------
    def diagnostics_by_level(self):
        out = {}
        for d in self.diagnostics:
            out.setdefault(d.level, []).append(d)
        return out

    def summary(self):
        counts = {}
        for d in self.diagnostics:
            counts[d.level] = counts.get(d.level, 0) + 1
        failed = sum(1 for m in self.materials.values() if m.failed)
        return {
            'files': len(self.files),
            'materials': len(self.materials),
            'tables': len(self.tables),
            'guides': len(self.guides),
            'failed': failed,
            'redefinitions': self.skipped_redefinitions,
            'diagnostics': counts,
        }


def parse_mtr_text(text, filename, db):
    """Parse one .mtr source into db. Tables are registered before materials
    so a table declared later in the same file still resolves."""
    file_mark = len(db.diagnostics)
    line_map = _LineMap()
    if db.guides:
        text, line_map = _expand_guides(text, filename, db.guides,
                                        db.diagnostics)
    toks = _lex(text)
    decls = _scan_decls(toks, filename, db.diagnostics)

    for decl in decls:
        if decl.type == 'table':
            table = _parse_table(toks, decl)
            db.tables.setdefault(table.name, table)

    for decl in decls:
        if decl.type != 'material':
            continue
        mark = len(db.diagnostics)
        mat = _parse_material(toks, decl, db.tables, db.diagnostics)
        finish_material(mat, db.diagnostics)
        mat.diagnostics = db.diagnostics[mark:]
        if not db.add_material(mat):
            # Redefinition: drop this parse and its diagnostics, keeping only
            # the redefinition note add_material() just appended.
            del db.diagnostics[mark:-1]

    if line_map:
        # Guide expansion moved everything after each invocation; put the
        # reported lines back on the source the user can actually open.
        for d in db.diagnostics[file_mark:]:
            d.line = line_map.original(d.line)
        for key in db.material_order:
            mat = db.materials[key]
            if mat.filename == filename:
                mat.line = line_map.original(mat.line)
    return db


def _mtr_roots(base_dir, mod_dir=''):
    """Asset roots, highest priority first - the bpy-free half of _abs_roots.

    This lives inside the parser block, which has to stay importable with no
    Blender at all (tests/mtr_corpus.py execs it standalone), so it does the
    ordering and de-duplication and nothing else. Paths are returned exactly
    as handed in; the addon-side _abs_roots expands Blender's "//" form first
    and then delegates here, which is where that expansion always happened.

    base_dir may be one path or several. A blank mod_dir - the normal case -
    just gives back the base roots.
    """
    if not base_dir:
        roots = ()
    elif isinstance(base_dir, str):
        roots = (base_dir,)
    else:
        roots = tuple(r for r in base_dir if r)
    if mod_dir:
        roots = (mod_dir,) + roots
    out, seen = [], set()
    for r in roots:
        key = _os.path.normcase(_os.path.normpath(r))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return tuple(out)


def collect_mtr_files(path):
    """Every .mtr at or under `path`, sorted for deterministic ordering.

    `path` may be several paths rather than one - a Mod Base makes the
    derived materials source one folder per root. They are concatenated in
    the order given, NOT merged or sorted together, because that order is
    the override order: the parser keeps the first declaration of a name it
    sees (MtrDatabase.add_material), so the mod's .mtr files have to be
    handed over first for the mod's version of a material to win.
    """
    if isinstance(path, (list, tuple)):
        out, seen = [], set()
        for one in path:
            for f in collect_mtr_files(one):
                key = _os.path.normcase(_os.path.normpath(f))
                if key not in seen:
                    seen.add(key)
                    out.append(f)
        return out
    if not path:
        return []
    if _os.path.isfile(path):
        return [path] if path.lower().endswith('.mtr') else []
    found = []
    for root, dirs, files in _os.walk(path):
        dirs.sort()
        for name in sorted(files):
            if name.lower().endswith('.mtr'):
                found.append(_os.path.join(root, name))
    return found


def collect_guide_files(base_dir, mod_dir=''):
    """Quake 4 keeps its guide templates in <base>/guides/*.guide.

    Every root's guides/ folder is read, mod first, not just the first one
    that exists: guides are a preprocessing layer that the .mtr files then
    reference by name, so a mod supplying two of its own would otherwise
    hide every guide the base game defines and take most of its materials
    down with them. Load order is the override order - the caller keeps the
    first definition of a guide name it is handed.
    """
    out = []
    for root in _mtr_roots(base_dir, mod_dir):
        guide_dir = _os.path.join(root, 'guides')
        if not _os.path.isdir(guide_dir):
            continue
        out.extend(_os.path.join(guide_dir, n)
                   for n in sorted(_os.listdir(guide_dir))
                   if n.lower().endswith('.guide'))
    return out


def _drop_resolved_table_notes(db):
    """Retract the "undefined table" notes whose table exists after all.

    The parser sees one file at a time and cannot answer the question the note
    asks, because a .mtr tree is not ordered: Doom 3 declares sinTable in
    shaderDemo.mtr, fireballtable in monsters.mtr and blamptable in
    washroom.mtr, and every one of them is referenced from files that sort
    earlier. The engine has no such problem - idDeclManagerLocal scans every
    decl file's names before anything is parsed, so a table lookup resolves
    wherever the table lives.

    Deferring the verdict to here rather than pre-scanning the tree for table
    names costs nothing: the note is already tagged with the name it is about
    (TABLE_NOTE_PREFIX), so answering it later is one pass over a list, where
    a pre-scan would mean lexing every file twice. Comment handling stays
    exactly the real lexer's, which a cheap regex pre-scan would not - and
    that matters, because Doom 3's one genuinely undefined table is
    cdplayertable, which is undefined precisely because its declaration at
    senetemp.mtr:1 is commented out.
    """
    def keep(diag):
        if not diag.kind.startswith(TABLE_NOTE_PREFIX):
            return True
        return diag.kind[len(TABLE_NOTE_PREFIX):] not in db.tables

    # The same predicate over both lists - the per-material slices hold the
    # same objects as db.diagnostics, so filtering by identity would work too,
    # but a pure test cannot get the two out of step.
    if any(not keep(d) for d in db.diagnostics):
        for key in db.material_order:
            mat = db.materials[key]
            if mat.diagnostics:
                mat.diagnostics = [d for d in mat.diagnostics if keep(d)]
        db.diagnostics[:] = [d for d in db.diagnostics if keep(d)]


def load_mtr_database(source_path, base_dir=None, mod_dir=''):
    """Parse a .mtr file or tree, plus any Quake 4 guides under base_dir.

    Guides must be loaded first: they are a preprocessing layer, and in
    Quake 4 the majority of materials only exist after expansion.

    source_path may be several paths (a Mod Base makes the derived source
    one materials/ folder per root). They are read in priority order and
    MtrDatabase.add_material keeps the FIRST definition of a name, which is
    exactly the override the engine's own fs_game search order produces: the
    mod's copy of a material wins and the base game's is skipped.
    """
    db = MtrDatabase()
    # Recorded so a report can state the search order it was built under -
    # with two roots, "material X came from the mod, Y from the base game"
    # is the first thing anyone needs to know when the result surprises them.
    db.roots = list(_mtr_roots(base_dir, mod_dir))

    for guide_path in collect_guide_files(base_dir, mod_dir):
        try:
            with open(guide_path, 'rb') as fh:
                text = decode_mtr_bytes(fh.read())
        except (IOError, OSError) as exc:
            db.diagnostics.append(MtrDiagnostic(
                DIAG_PARSE_ERROR, 'io', 'cannot read guide file: %s' % exc,
                guide_path, 0))
            continue
        for guide in parse_guide_file(text, guide_path, db.diagnostics):
            # setdefault, so with several roots the highest-priority one's
            # guide keeps the name - same rule add_material applies.
            db.guides.setdefault(engine_canonical_decl(guide.name), guide)

    for path in collect_mtr_files(source_path):
        try:
            with open(path, 'rb') as fh:
                text = decode_mtr_bytes(fh.read())
        except (IOError, OSError) as exc:
            db.diagnostics.append(MtrDiagnostic(
                DIAG_PARSE_ERROR, 'io', 'cannot read file: %s' % exc, path, 0))
            continue
        db.files.append(path)
        parse_mtr_text(text, path, db)

    _drop_resolved_table_notes(db)
    return db


# ---------------------------------------------------------------------------
# Expression evaluation
# ---------------------------------------------------------------------------
# Two consumers: the constant folder used by Simple mode and by Standard
# mode's initial socket values, and the driver emitter used by Standard mode
# for anything that varies. Both walk the same AST, so a driven socket and the
# static value it was seeded with can never disagree - which they could in the
# old importer, where the static path used the declared sinTable and the
# driver path substituted an analytic (sin+1)/2 remap instead.

class MtrEvalContext(object):
    """Values the predefined expression terms resolve to."""

    def __init__(self, tables=None, time=0.0, parms=None, globals_=None,
                 sound=0.0, distance=0.0):
        self.tables = tables or {}
        self.time = time
        self.parms = list(parms) if parms else [0.0] * 12
        while len(self.parms) < 12:
            self.parms.append(0.0)
        self.globals = list(globals_) if globals_ else [0.0] * 8
        while len(self.globals) < 8:
            self.globals.append(0.0)
        self.sound = sound
        self.distance = distance

    def var(self, name):
        if name == 'time':
            return self.time
        if name == 'sound':
            return self.sound
        if name == 'distance':
            return self.distance
        if name.startswith('parm'):
            return self.parms[int(name[4:])]
        if name.startswith('global'):
            return self.globals[int(name[6:])]
        if name in ('fragmentprograms', 'glslprograms'):
            return 1.0
        return 0.0


# idTech 4's default entity shader parms: parm0..3 are the colour (white,
# opaque) and the rest start at zero. Simple mode folds against these, which
# is what an untouched entity looks like on frame 0.
DEFAULT_PARMS = [1.0, 1.0, 1.0, 1.0] + [0.0] * 8


def eval_expr(expr, ctx):
    """Evaluate an MtrExpr. Engine-exact: right-associativity is already in
    the tree shape, and % truncates both operands to integers."""
    kind = expr.kind
    if kind == EXPR_CONST:
        return expr.a
    if kind == EXPR_VAR:
        return ctx.var(expr.a)
    if kind == EXPR_TABLE:
        table = ctx.tables.get(expr.a)
        index = eval_expr(expr.b, ctx)
        return table.lookup(index) if table is not None else 0.0

    a = eval_expr(expr.a, ctx)
    b = eval_expr(expr.b, ctx)
    op = expr.op
    if op == '+':
        return a + b
    if op == '-':
        return a - b
    if op == '*':
        return a * b
    if op == '/':
        return a / b if b else 0.0
    if op == '%':
        ib = int(b)
        return float(int(a) % ib) if ib else 0.0
    if op == '>':
        return 1.0 if a > b else 0.0
    if op == '>=':
        return 1.0 if a >= b else 0.0
    if op == '<':
        return 1.0 if a < b else 0.0
    if op == '<=':
        return 1.0 if a <= b else 0.0
    if op == '==':
        return 1.0 if a == b else 0.0
    if op == '!=':
        return 1.0 if a != b else 0.0
    if op == '&&':
        return 1.0 if (a and b) else 0.0
    if op == '||':
        return 1.0 if (a or b) else 0.0
    if op == 'min':
        return a if a < b else b
    if op == 'max':
        return a if a > b else b
    return 0.0


def eval_or(expr, ctx, default=0.0):
    if expr is None:
        return default
    try:
        return eval_expr(expr, ctx)
    except (ValueError, OverflowError, ZeroDivisionError, IndexError):
        return default


# ---------------------------------------------------------------------------
# Driver emission
# ---------------------------------------------------------------------------
# Blender driver expressions, fully parenthesised so the engine's grouping is
# visible in the Drivers editor rather than merely implied.

DRIVER_TABLE_FN = 'idtech4_tbl'
DRIVER_PARM_FN = 'idtech4_parm'
DRIVER_GLOBAL_FN = 'idtech4_global'
DRIVER_SOUND_FN = 'idtech4_sound'
DRIVER_SPECTRUM_FN = 'idtech4_spectrum'

_DRIVER_BINOPS = {
    '+': '%s + %s', '-': '%s - %s', '*': '%s * %s',
    '>': 'float(%s > %s)', '>=': 'float(%s >= %s)',
    '<': 'float(%s < %s)', '<=': 'float(%s <= %s)',
    '==': 'float(%s == %s)', '!=': 'float(%s != %s)',
    '&&': 'float(bool(%s) and bool(%s))',
    '||': 'float(bool(%s) or bool(%s))',
    'min': 'min(%s, %s)', 'max': 'max(%s, %s)',
}


def expr_to_driver(expr, fps=24.0):
    """Translate an MtrExpr into a Blender scripted-expression driver."""
    kind = expr.kind
    if kind == EXPR_CONST:
        return repr(float(expr.a))
    if kind == EXPR_VAR:
        name = expr.a
        if name == 'time':
            return '(frame / %.6f)' % (fps if fps > 0 else 24.0)
        if name.startswith('parm'):
            return '%s(%s)' % (DRIVER_PARM_FN, name[4:])
        if name.startswith('global'):
            return '%s(%s)' % (DRIVER_GLOBAL_FN, name[6:])
        if name == 'sound':
            return '%s()' % DRIVER_SOUND_FN
        if name in ('fragmentprograms', 'glslprograms'):
            return '1.0'
        return '0.0'
    if kind == EXPR_TABLE:
        return "%s('%s', %s)" % (DRIVER_TABLE_FN, expr.a,
                                 expr_to_driver(expr.b, fps))

    a = expr_to_driver(expr.a, fps)
    b = expr_to_driver(expr.b, fps)
    op = expr.op
    if op == '/':
        # The engine returns 0 rather than raising on a divide by zero.
        return '(%s / %s if %s else 0.0)' % (a, b, b)
    if op == '%':
        # OP_TYPE_MOD truncates both operands to integers first.
        return '(float(int(%s) %% int(%s)) if int(%s) else 0.0)' % (a, b, b)
    fmt = _DRIVER_BINOPS.get(op)
    if fmt is None:
        return '0.0'
    return '(' + (fmt % (a, b)) + ')'

# ===========================================================================
# END MTR PARSER
# ===========================================================================


# ===========================================================================
# BEGIN ASSET RESOLUTION
#
# Everything between a name in a .mtr and a Blender image datablock: where the
# file actually is on disk, the compiled .bimage cache the engine would have
# loaded instead, the six-faces-to-equirectangular cube-map bake, and the
# colorspace guard that keeps one shared datablock from being claimed by
# whichever stage happened to wire it last.
#
# The module-level caches here are deliberately module-level, not per-resolver:
# bpy.data.images is one namespace for the whole session, so "have I already
# loaded this file?" has exactly one right answer regardless of how many
# imports are in flight.
# ===========================================================================


# Image datablocks are shared across materials (load_or_find_image reuses
# any existing bpy.data.images entry with the same filepath), but idTech4
# content routinely reuses one diffuse/albedo file as the greyscale source
# for a heightmap or specular stage too (e.g. "heightmap(foo_d.tga, 3)"
# pulling extra bump detail straight out of the diffuse texture). Since
# colorspace_settings is a property of the Image datablock itself, not of
# any one texture node's usage, whichever call sets it last would otherwise
# win for every usage of that image. Track every image ever wired as
# genuine color data (sRGB) so a later Non-Color wiring (bump/height/spec)
# of that same file can never silently downgrade it back.
_SRGB_MARKED_IMAGES = set()


def _set_image_colorspace(img, colorspace):
    """Set img's colorspace, refusing to downgrade an image already known
    to be color data (sRGB) to Non-Color — see _SRGB_MARKED_IMAGES above."""
    if colorspace == 'sRGB':
        _SRGB_MARKED_IMAGES.add(img.name_full)
        img.colorspace_settings.name = 'sRGB'
    elif img.name_full not in _SRGB_MARKED_IMAGES:
        img.colorspace_settings.name = colorspace


def _set_image_alpha_mode(img):
    """Keep an image's RGB and alpha independent, the way GL samples them.

    Blender's default for a loaded file is alpha_mode='STRAIGHT', which means
    "the RGB is unassociated, associate it for me": the buffer read back
    through Image.pixels still holds the file's own RGB, but the copy handed
    to Cycles and to EEVEE's GPU textures is premultiplied, so the Image
    Texture node's COLOR output evaluates to black wherever alpha is 0.

    No idTech 4 image works that way. LoadTGA (Image_files.cpp) copies red,
    green, blue and alphabyte into the texel one after another and never
    multiplies them together, and the fragment programs read .rgb and .a as
    two unrelated channels - so a stage whose blend equation ignores source
    alpha, which is most of them (_ALPHA_BLIND_BLENDS), must still see the
    file's colour where the artist left the alpha at 0.

    models/characters/scientist/head02/glasses2_fx is the case that found
    this: `blend filter` over a 32-bit .tga that is near-white with alpha 0
    across 77% of its texels. The multiply is exact - a Transparent BSDF
    tinted with the source colour - but the tint arrived premultiplied, so
    over three quarters of the lens it was Transparent(black), which is
    opaque black. 1,464 Doom 3 materials, 1,276 Quake 4, 772 Prey and 186
    Dark Mod ones sample an image that can bite this way.

    CHANNEL_PACKED is the mode that says "these channels are separate images
    and must not affect each other". It leaves the Alpha output bit-identical
    to STRAIGHT, so alphaTest, the alpha-blended stages and the maskColor
    idiom are untouched; only the premultiply goes away. NONE would also fix
    the colour, and would break every perforated material by forcing Alpha
    to 1.

    Reading the enum first is not a micro-optimisation: assigning it fires
    Blender's colour-management invalidation whether or not the value
    changed, and a whole-map import asks for the same datablock thousands of
    times. The read itself is free - measured at the noise floor over 52,009
    image references - so all the cost is the one assignment per datablock,
    ~250us of colour-management invalidation, which is +3.0s on a 6,265
    material / 11,892 image Doom 3 build (47.6s -> 50.6s, five interleaved
    pairs). Two ways of ducking it were measured and are worse:

      * memoising in a Python set instead of reading the enum saves nothing,
        because the read was never the cost;
      * skipping 24-bit images, which cannot be premultiplied at all, would
        cut 11,892 assignments to 1,872 - but reading img.depth forces the
        file off disk, and that costs more than the assignments it avoids
        (57.9s).

    Returns img so callers can `return _set_image_alpha_mode(x)`.
    """
    try:
        if img.alpha_mode != 'CHANNEL_PACKED':
            img.alpha_mode = 'CHANNEL_PACKED'
    except (AttributeError, TypeError):
        pass
    return img


# ---------------------------------------------------------------------------
# .bimage fallback
#
# The engine never loads a source .tga/.png straight off disk at runtime -
# every image it ever touches gets compiled once into a
# "generated/images/**.bimage" cache (idImage::ActuallyLoadImage /
# idBinaryImage::WriteGeneratedFile in renderer/BinaryImage.cpp) and loaded
# from there on subsequent runs. Materials never reference that cache
# directly, so when a .mtr points at a source image that isn't actually
# present in the working directory (only shipped as a compiled resource),
# we can still recover it by reconstructing the cache's own filename
# convention and loading the compiled .bimage instead. Decoding is handled
# by the sibling idTech4_bimage.py script - see that file for the format
# reverse-engineering notes.
# ---------------------------------------------------------------------------

# textureUsage_t (renderer/Image.h) - only the values that affect the
# generated-name suffix for ordinary material-stage textures.
TD_SPECULAR, TD_DIFFUSE, TD_DEFAULT, TD_BUMP, TD_FONT, TD_LIGHT = range(6)

_bimage_module = None
_bimage_module_load_attempted = False


def _get_bimage_module():
    """Lazily load the sibling idTech4_bimage.py script (same folder as this
    file) if present. Returns None - disabling the .bimage fallback - if it
    can't be found or fails to load, so this addon works standalone too."""
    global _bimage_module, _bimage_module_load_attempted
    if _bimage_module_load_attempted:
        return _bimage_module
    _bimage_module_load_attempted = True
    try:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        bimage_path = os.path.join(this_dir, "idTech4_bimage.py")
        if os.path.isfile(bimage_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("idTech4_bimage", bimage_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _bimage_module = mod
    except Exception as e:
        print(f"idTech4 Materials: couldn't load idTech4_bimage.py ({e}); "
              f".bimage fallback disabled")
        _bimage_module = None
    return _bimage_module


def _bimage_usage_for_label(rel_path, usage_label):
    """Mirror idImageManager::ImageFromFile's path-prefix overrides
    (renderer/ImageManager.cpp) and Material.cpp's per-stage-type usage
    assignment (SL_DIFFUSE/SL_BUMP/SL_SPECULAR -> TD_DIFFUSE/TD_BUMP/
    TD_SPECULAR, everything else -> TD_DEFAULT)."""
    lower = rel_path.replace('\\', '/').lower()
    if lower.startswith('fonts/') or lower.startswith('newfonts/'):
        return TD_FONT
    if lower.startswith('lights/'):
        return TD_LIGHT
    if usage_label == 'Diffuse':
        return TD_DIFFUSE
    if usage_label == 'Specular':
        return TD_SPECULAR
    if usage_label == 'Normal Map':
        return TD_BUMP
    return TD_DEFAULT


def _abs_roots(base_dir, mod_dir=''):
    """Absolute asset roots, highest priority first.

    Every path resolver below takes this rather than a single directory, so
    an optional Mod Base can sit in front of the Base Directory: the mod is
    searched first and anything it does not ship falls through to the base
    game. *base_dir* may itself be a sequence, for the callers that already
    hold a resolved root list and have no separate mod to add.

    Blender's "//" blend-relative form is expanded here and the ordering and
    de-duplication are then _mtr_roots' job - that half lives inside the MTR
    parser block, which has to stay importable with no bpy at all.

    Returns a tuple - possibly empty, which every caller reads as "nothing
    is configured" the same way a blank base_dir used to.
    """
    if not base_dir:
        bases = ()
    elif isinstance(base_dir, str):
        bases = (base_dir,)
    else:
        bases = tuple(r for r in base_dir if r)
    return _mtr_roots(tuple(bpy.path.abspath(b) for b in bases),
                      bpy.path.abspath(mod_dir) if mod_dir else '')


def _dds_cache_name(image_prog):
    """idImage::ImageProgramStringToCompressedFileName, verbatim.

    Doom 3 renderer/Image_load.cpp:980, The Dark Mod
    renderer/resources/Image_load.cpp:723 - every idTech4 renderer maps an
    image program string to its precompressed-cache path this way, and
    prefers that file over the source image when it exists.

    Prefix 'dds/', then per character:
        /  \\  (        -> '/'   (see the depth cap below)
        < > : | " .     -> '_'
        )  ,            -> dropped
        ' ' after a '/' -> dropped
    then append '.dds'.

    For a plain texture path this collapses to 'dds/<path>.dds', which is the
    case this function exists to serve and the only one it is verified
    against.

    It does NOT reproduce Doom 3's and Prey's cached IMAGE PROGRAM names, and
    is not expected to. Their dds/ trees do carry mangled program entries -
    top-level addnormals/, heightmap/, makealpha/, makeintensity/ and
    smoothnormals/ directories, e.g.

        dds/addnormals/models/characters/blood_stump/blood_stump_local heightmap  models characters blood_stump blood_stump_h 7.dds

    - but the engine mangles its own canonically re-serialized program string
    (built by R_ParsePastImageProgram, with .tga stripped throughout), not the
    raw .mtr text, and that serialization's exact spacing was not recovered
    from the available sources: feeding this function the real .mtr text
    yields the right shape but differs in whitespace around '(' and ')'.

    That gap costs nothing here, because image programs never reach this
    lookup - resolve_image_program_bimage handles them, and it prefers the
    program's plain source files, which are present for every program-based
    material in the corpus. If that ever changes, this is the function to
    finish, starting from R_ParsePastImageProgram.

    The depth cap is a real divergence between the games: Doom 3 and Prey
    stop substituting '/' after 4 levels and emit a space instead, while The
    Dark Mod removed the cap and always emits '/'. That bites plain paths
    too - 'textures/darkmod/wood/panels/x' already sits at the cap - so both
    spellings are generated, uncapped first because it is the one that
    resolves real texture paths.
    """
    out = []
    for capped in (False, True):
        f = ['dds/']
        depth = 0
        for c in image_prog:
            if c in '/\\(':
                if capped and depth >= 4:
                    f.append(' ')
                else:
                    f.append('/')
                    depth += 1
            elif c in '<>:|".':
                f.append('_')
            elif c == ' ' and f and f[-1].endswith('/'):
                pass                      # ignore a space right after a slash
            elif c in '),':
                pass                      # always ignored
            else:
                f.append(c)
        name = ''.join(f) + '.dds'
        if name not in out:
            out.append(name)
    return out


def _dds_cache_path(roots_abs, rel_path):
    """The precompressed dds/ image for rel_path under the first root that
    has it, or None.

    This is the lookup that makes The Dark Mod work at all. Measured over
    each game's whole dds/ tree - files with NO plain-source counterpart
    anywhere under the root:

        doom3     762 of 8116   ( 9%)
        prey      799 of 5300   (15%)
        darkmod  4922 of 4951   (99%)
        quake4      no dds/ tree at all

    Doom 3 and Prey ship dds/ as a genuine cache beside the sources, so
    missing it only loses a few hundred files. The Dark Mod ships DDS as the
    PRIMARY asset - its plain textures/ tree is mostly _local.tga normal maps
    and _ed.jpg editor images, and the diffuse/specular exist nowhere else.
    Without this, most Dark Mod surfaces resolve to nothing and render pink.

    Both the engine's mangled name and a plain extension-stripped form are
    tried: a .mtr may reference '..._s.tga' where the shipped file is
    '..._s.dds', not the '..._s_tga.dds' the mangling rule alone produces.
    """
    rel = rel_path.replace('\\', '/')
    names = list(_dds_cache_name(rel))
    stripped = strip_material_extension(rel)
    if stripped and stripped != rel:
        for n in _dds_cache_name(stripped):
            if n not in names:
                names.append(n)
    for root_abs in roots_abs:
        for name in names:
            cand = os.path.normpath(os.path.join(root_abs, name))
            if os.path.isfile(cand):
                return cand
    return None


def _dds_cache_candidates(root_abs, rel_path):
    """Every dds/ path _dds_cache_path would try under one root, in order.

    Split out so image_search_paths can quote them - a "texture not found"
    report that omits the dds/ tree sends anyone reading it to the wrong
    place, which is exactly how this bug survived so long.
    """
    rel = rel_path.replace('\\', '/')
    names = list(_dds_cache_name(rel))
    stripped = strip_material_extension(rel)
    if stripped and stripped != rel:
        for n in _dds_cache_name(stripped):
            if n not in names:
                names.append(n)
    return [os.path.normpath(os.path.join(root_abs, n)) for n in names]


def _bimage_generated_path(base_dir_abs, rel_path, usage, cube=0):
    """Mirror idImage::GetGeneratedName + idBinaryImage::GetGeneratedFileName
    (renderer/Image_load.cpp, renderer/BinaryImage.cpp): strip a literal
    '.tga' extension (that's the only one ImageFromFile ever strips), splice
    in '#__<usage><cube>' before any extension that's left, then the whole
    thing lands under generated/images/ with a .bimage extension."""
    name = rel_path.replace('\\', '/').replace('.tga', '').replace('.TGA', '')
    root, ext = os.path.splitext(name)
    suffixed = f"{root}#__{usage:02d}{cube:02d}{ext}"
    gfn = f"generated/images/{suffixed}.bimage"
    gfn = gfn.replace('(', '/').replace(',', '/').replace(')', '').replace(' ', '')
    return os.path.normpath(os.path.join(base_dir_abs, gfn))


def _find_bimage_fallback(roots_abs, rel_path, usage_label):
    """Look for a compiled .bimage cache matching rel_path. Tries the usage
    code this call site would actually get first, then the other common
    usages in case the same source was only ever cached under a different
    one - decoding is driven by the file's own header either way, so any
    match found this way still decodes correctly."""
    mod = _get_bimage_module()
    if mod is None:
        return None
    if isinstance(roots_abs, str):
        roots_abs = (roots_abs,)
    primary = _bimage_usage_for_label(rel_path, usage_label)
    candidates = [primary] + [u for u in (TD_DEFAULT, TD_DIFFUSE, TD_SPECULAR, TD_BUMP) if u != primary]
    # Roots outer, usages inner: a root that has ANY cached variant of this
    # texture is the root that owns it, and reaching past it to the base
    # game for a different usage code would decode the wrong image.
    for root_abs in roots_abs:
        for usage in candidates:
            path = _bimage_generated_path(root_abs, rel_path, usage)
            if os.path.isfile(path):
                return path
    return None


# --- composite bumpmap() / addnormals(bumpmap(), heightmap()) fallback ----
#
# heightmap(path, scale) and addnormals(a, heightmap(b, scale)) aren't
# simple per-file textures - the engine's offline image-program interpreter
# (renderer/Image_program.cpp: R_ParseImageProgram_r) bakes the WHOLE
# expression into one composite normal map and caches THAT under a path
# built from the expression's own canonical text, e.g. for
# addnormals(models/x/foo_local.tga, heightmap(models/x/foo_h.tga, 10)):
#
#   generated/images/addnormals/models/x/foo_local/heightmap/models/x/foo_h/10#__0300.bimage
#
# (usage is always forced to TD_BUMP=3 for these - see the heightmap branch
# of R_ParseImageProgram_r). The individual foo_local.tga/foo_h.tga source
# files referenced *inside* the expression are never cached under their own
# name, so if they're missing from disk (binary-only distribution) the only
# way back to a usable normal map is this composite cache - which is
# already the fully-baked result, equivalent to a plain normal map texture,
# not a height field needing Blender's own Bump-node conversion.

def _bimage_composite_path(base_dir_abs, joined_expr, usage=TD_BUMP, cube=0):
    """joined_expr: the image-program text as the engine's parseBuffer would
    hold it (parens/commas intact, e.g. 'addnormals(a,heightmap(b,10))').
    Mirrors ImageManager::ImageFromFile's blanket '.tga' strip (applied to
    the whole string, including embedded params) followed by
    idBinaryImage::GetGeneratedFileName's '(' '/' ',' '/' ')' '' ' ' ''
    substitutions - there's no remaining extension on a composite
    expression, so the usage/cube suffix lands at the very end."""
    stripped = joined_expr.replace('.tga', '').replace('.TGA', '')
    suffixed = f"{stripped}#__{usage:02d}{cube:02d}"
    gfn = f"generated/images/{suffixed}.bimage"
    gfn = gfn.replace('(', '/').replace(',', '/').replace(')', '').replace(' ', '')
    return os.path.normpath(os.path.join(base_dir_abs, gfn))



# --- image-program .bimage composite fallback ------------------------------
#
# An image program is not a per-file texture: the engine's offline
# interpreter (R_ParseImageProgram_r) evaluates the WHOLE expression once and
# caches the finished result under a path built from the expression's own
# canonical text, e.g.
#
#   generated/images/addnormals/x/foo_local/heightmap/x/foo_h/10#__0300.bimage
#
# The individual files named inside the expression are never cached under
# their own names, so for a binary-only distribution this composite is the
# only way back to a usable texture - and it is already the baked result, so
# it should be wired like an ordinary texture rather than rebuilt from parts.

_IMGPROG_FORCED_BUMP = frozenset(('heightmap', 'addnormals', 'smoothnormals'))


def resolve_image_program_bimage(base_dir, img_expr, usage_label=None,
                                 mod_dir=''):
    """Absolute path of the pre-baked .bimage for an image program, or None.

    Returns None when the program's own source files are present on disk -
    those are preferred, because they keep the conversion editable in the
    node tree - or when no matching cache exists either.
    """
    if img_expr is None or img_expr.op is None:
        return None
    if _get_bimage_module() is None:
        return None

    roots_abs = _abs_roots(base_dir, mod_dir)
    inner = img_expr.base_path()
    if inner and _plain_source_path(roots_abs, inner):
        return None

    if img_expr.op in _IMGPROG_FORCED_BUMP:
        candidates = [TD_BUMP]
    else:
        primary = _bimage_usage_for_label(inner or '', usage_label)
        candidates = [primary] + [u for u in (TD_DEFAULT, TD_DIFFUSE,
                                              TD_SPECULAR, TD_BUMP)
                                  if u != primary]
    for root_abs in roots_abs:
        for usage in candidates:
            path = _bimage_composite_path(root_abs, img_expr.canonical,
                                          usage=usage)
            if os.path.isfile(path):
                return path
    return None


def _plain_source_candidates(root_abs, rel_path):
    """Every plain source path tried for rel_path under one root, in order.

    Split out from _plain_source_path so a "missing texture" report can say
    where it looked - see _searched_paths_for.
    """
    candidate = os.path.normpath(os.path.join(root_abs, rel_path))
    return [candidate] + [candidate + ext for ext in
                          ('.tga', '.png', '.jpg', '.jpeg', '.dds', '.bmp')]


def _plain_source_path(roots_abs, rel_path):
    """Return the plain on-disk source image for rel_path (with or without
    common image extensions) under the first root that has it, or None.

    Roots are tried in priority order and a root is exhausted before the
    next is tried, so a mod's own .tga wins over the base game's - but a
    mod that ships no copy at all falls straight through.
    """
    if isinstance(roots_abs, str):
        roots_abs = (roots_abs,)
    for root_abs in roots_abs:
        for cand in _plain_source_candidates(root_abs, rel_path):
            if os.path.isfile(cand):
                return cand
    return None


def resolve_image_path(base_dir, rel_path, usage_label=None, mod_dir=''):
    """Return an absolute filesystem path for rel_path under base_dir.
    Tries with and without common image extensions, then - if the source
    image isn't actually present in the working directory - falls back to
    the compiled generated/images/**.bimage cache the engine would have
    loaded instead (see the ".bimage fallback" section above). usage_label
    ('Diffuse'/'Specular'/'Normal Map'/None) should match whatever this call
    site would tell the engine, so the right compressed variant is found.

    With a Mod Base configured every root is tried in turn, mod first. The
    not-found return is built under the LAST root - the base game's - because
    that is where a genuinely missing texture was supposed to live, and a
    broken image path pointing into the mod would send anyone tracking it
    down to the wrong tree."""
    roots_abs = _abs_roots(base_dir, mod_dir)
    if not roots_abs:
        return os.path.normpath(rel_path)

    plain = _plain_source_path(roots_abs, rel_path)
    if plain:
        return plain

    # The engine's precompressed dds/ cache, which it prefers over the source
    # image whenever it exists. Tried AFTER plain source rather than before,
    # for two reasons: an uncompressed .tga/.png is better input to Blender
    # than a DXT-compressed copy of the same image, and searching it second
    # makes this addition strictly additive - every lookup that already
    # succeeded resolves to the byte-identical path it did before, and only
    # what used to fail can now be found. See _dds_cache_path for why The
    # Dark Mod cannot be imported at all without this.
    dds_path = _dds_cache_path(roots_abs, rel_path)
    if dds_path:
        return dds_path

    bimage_path = _find_bimage_fallback(roots_abs, rel_path, usage_label)
    if bimage_path:
        return bimage_path

    candidate = os.path.normpath(os.path.join(roots_abs[-1], rel_path))
    # Return the original candidate even if not found (Blender will show missing)
    return candidate


def image_search_paths(base_dir, rel_path, usage_label=None, mod_dir=''):
    """Every path resolve_image_path would try for rel_path, in order.

    This is what a "texture not found" report quotes. With one root it is
    already more useful than the bare name (six extensions and four .bimage
    usage codes are a lot of places to have looked); with a Mod Base it is
    the only way to tell "the mod doesn't ship it and neither does the base"
    apart from "one of these two roots is pointing at the wrong tree".
    """
    out = []
    roots_abs = _abs_roots(base_dir, mod_dir)
    for root_abs in roots_abs:
        out.extend(_plain_source_candidates(root_abs, rel_path))
    for root_abs in roots_abs:
        out.extend(_dds_cache_candidates(root_abs, rel_path))
    if _get_bimage_module() is not None:
        primary = _bimage_usage_for_label(rel_path, usage_label)
        usages = [primary] + [u for u in (TD_DEFAULT, TD_DIFFUSE, TD_SPECULAR,
                                          TD_BUMP) if u != primary]
        for root_abs in roots_abs:
            for usage in usages:
                out.append(_bimage_generated_path(root_abs, rel_path, usage))
    return out



# Whole-map imports resolve tens of thousands of textures, so the "have I
# already loaded this file?" lookup cannot be a linear scan of bpy.data.images
# - that turns generation into an O(n^2) crawl (it was 70% of the build time
# before this cache existed). Keyed by normalised absolute path, seeded once,
# and rebuilt only when something outside this addon changes the image count.
_IMAGE_BY_PATH = {}
_IMAGE_CACHE_COUNT = -1


def _image_lookup():
    global _IMAGE_CACHE_COUNT
    if len(bpy.data.images) != _IMAGE_CACHE_COUNT:
        _IMAGE_BY_PATH.clear()
        for img in bpy.data.images:
            try:
                key = os.path.normpath(bpy.path.abspath(img.filepath))
            except (AttributeError, TypeError, ValueError):
                continue
            _IMAGE_BY_PATH.setdefault(key, img)
        _IMAGE_CACHE_COUNT = len(bpy.data.images)
    return _IMAGE_BY_PATH


def _remember_image(abs_path, img):
    global _IMAGE_CACHE_COUNT
    _IMAGE_BY_PATH[abs_path] = img
    _IMAGE_CACHE_COUNT = len(bpy.data.images)
    return _set_image_alpha_mode(img)


def load_or_find_image(image_path):
    """
    Return a Blender image data-block for image_path without creating duplicates.
    Searches bpy.data.images by normalised filepath before calling load().
    If image_path is a compiled .bimage cache (see resolve_image_path's
    fallback), decodes it via the sibling idTech4_bimage.py module instead
    of bpy.data.images.load(), which can't read that format.
    If the file does not exist on disk a placeholder image is created so the
    node tree is still valid (shows pink/missing in the viewport).
    """
    abs_path = os.path.normpath(bpy.path.abspath(image_path))

    cached = _image_lookup().get(abs_path)
    if cached is not None:
        try:
            cached.name          # touch it; a removed datablock raises here
            return _set_image_alpha_mode(cached)
        except ReferenceError:
            _IMAGE_BY_PATH.pop(abs_path, None)

    if abs_path.lower().endswith('.bimage'):
        mod = _get_bimage_module()
        if mod is not None:
            try:
                img, _info = mod.load_bimage_as_image(abs_path)
                return _remember_image(abs_path, img)
            except Exception as e:
                print(f"idTech4 Materials: failed to decode '{abs_path}': {e}")
        # fall through to the placeholder below if decoding wasn't possible

    if os.path.isfile(abs_path):
        try:
            # check_existing=False on purpose: Blender's own duplicate check
            # is a linear scan of bpy.data.images, and _image_lookup() has
            # already answered that question from a dict. With several
            # thousand textures loaded, letting Blender rescan on every load
            # is most of the import time.
            return _remember_image(
                abs_path, bpy.data.images.load(abs_path, check_existing=False))
        except RuntimeError:
            pass

    # Placeholder for missing files
    img = bpy.data.images.new(os.path.basename(abs_path), 4, 4)
    img.filepath = abs_path
    img.source = 'FILE'
    return _remember_image(abs_path, img)


def _generate_glass_streak_pixels(size, base_alpha, seed):
    """Procedural pane-of-glass alpha pattern: a mostly-clear field at
    base_alpha with a handful of thin diagonal streaks (like light catching
    scratches/smears) rising toward near-opaque, brightening toward white
    at their peak the way a glint does. Returns a flat RGBA list, bottom-to
    -top row order (matches Image.pixels)."""
    rnd = random.Random(seed)
    base_rgb = (0.86, 0.92, 0.95)
    glint_rgb = (0.98, 0.99, 1.0)
    diag = size + size
    streaks = [{
        'offset': rnd.uniform(0, diag),
        'thickness': rnd.uniform(1.2, 3.0),
        'strength': rnd.uniform(0.35, 0.75),
    } for _ in range(rnd.randint(3, 6))]

    pixels = [0.0] * (size * size * 4)
    for y in range(size):
        for x in range(size):
            d = x + y
            glint = 0.0
            for s in streaks:
                dist = abs(d - s['offset']) % diag
                dist = min(dist, diag - dist)
                if dist < s['thickness']:
                    falloff = 1.0 - (dist / s['thickness'])
                    glint = max(glint, s['strength'] * falloff)
            a = min(base_alpha + glint, 1.0)
            idx = (y * size + x) * 4
            pixels[idx + 0] = base_rgb[0] + (glint_rgb[0] - base_rgb[0]) * glint
            pixels[idx + 1] = base_rgb[1] + (glint_rgb[1] - base_rgb[1]) * glint
            pixels[idx + 2] = base_rgb[2] + (glint_rgb[2] - base_rgb[2]) * glint
            pixels[idx + 3] = a
    return pixels


def get_or_create_glass_placeholder_image(name_hint, base_alpha=0.15):
    """A stand-in for the GUARANTEED FALLBACK's qer_editorimage when even
    THAT can't be found on disk (no plain source, no .bimage in any form).

    load_or_find_image()'s usual placeholder sets source='FILE' pointing at
    a path that doesn't exist, which Blender flags with its built-in
    magenta/pink "missing image" checker — glaring in the viewport and, for
    a glass material, actively misleading (a real qer_editorimage here only
    ever tints the Diffuse BSDF; the translucent look comes entirely from
    the fallback's Mix Shader Fac, so a pink block reads as "broken" when
    the material would otherwise render as perfectly plausible pale glass).

    This builds an ordinary GENERATED-source image instead: a light, cool-
    tinted swatch that's mostly at base_alpha (clear pane) with a few thin
    diagonal streaks rising toward opaque and brightening toward white -
    glints, the way real light catches smears/scratches on glass - rather
    than flat, uniform coverage. Its own Alpha is meant to drive the
    fallback's Mix Shader Fac (see ensure_visible) so those streaks
    actually show up as more solid/more reflective, not just as a static
    color. Named '<hint>_missing_temp' so it stays obviously synthetic in
    the Outliner/Image list rather than being mistaken for a real asset.
    """
    name = f"{name_hint}_missing_temp"
    tag = f"{name_hint}|{base_alpha:.3f}"
    for img in bpy.data.images:
        if img.get("idtech4_glass_placeholder_tag") == tag:
            return img
    size = 64
    img = bpy.data.images.new(name, size, size, alpha=True, float_buffer=False)
    img.pixels.foreach_set(_generate_glass_streak_pixels(size, base_alpha, seed=name_hint))
    _pack_written_pixels(img)
    img["idtech4_glass_placeholder_tag"] = tag
    return img


def _pack_written_pixels(img):
    """Make pixels written into a new image survive.

    bpy.data.images.new() gives a source='GENERATED' image whose buffer
    Blender regenerates from generated_color whenever something invalidates
    it - and simply assigning colorspace_settings.name is enough to do that.
    Everything foreach_set() wrote is then silently replaced by flat black,
    which is exactly how the baked cube maps came out empty. Packing turns
    the buffer into real owned image data (source becomes 'FILE' with a
    packed_file) and it survives.

    Which is also why the alpha mode is set here and not at bpy.data.images
    .new(): it is one more property whose assignment invalidates the buffer,
    so it has to wait until pack() has made the pixels real.
    """
    try:
        img.pack()
    except RuntimeError:
        pass
    _set_image_alpha_mode(img)


# ---------------------------------------------------------------------------
# Cube maps
# ---------------------------------------------------------------------------
# R_LoadCubeImages loads six separate files and hands pics[i] to
# GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, so these orders are the GL face order:
# +X, -X, +Y, -Y, +Z, -Z. `cubeMap` names the faces by axis; `cameraCubeMap`
# names them camera-relative and reorients each one on load so it ends up in
# that same axis order. R_RotatePic is a transpose (dst[i][j] = src[j][i]),
# not a rotation, which is why "forward" needs nothing more than that.
_CUBE_AXIS_SIDES = ('_px', '_nx', '_py', '_ny', '_pz', '_nz')
_CUBE_CAMERA_SIDES = ('_forward', '_back', '_left', '_right', '_up', '_down')
# (transpose, horizontal flip, vertical flip) per face, CF_CAMERA only.
_CUBE_CAMERA_FIXUP = ((True, False, False), (True, True, True),
                      (False, False, True), (False, True, False),
                      (True, False, False), (True, False, False))


def resolve_cube_faces(base_dir, rel_path, camera, mod_dir=''):
    """The six on-disk faces of a cubeMap/cameraCubeMap, in GL order, or None.

    Blender has no cube-map texture node, so a stage reading
    `cubeMap env/gen2` used to resolve as though `env/gen2` were one ordinary
    file. Nothing is at that path - only env/gen2_px.tga and its five
    siblings - so the stage fell through to load_or_find_image()'s
    missing-file placeholder, which Blender draws as its magenta "no image"
    checker. On the reflection stage of a glass material, which is additive,
    that painted the entire surface pink.
    """
    roots_abs = _abs_roots(base_dir, mod_dir)
    sides = _CUBE_CAMERA_SIDES if camera else _CUBE_AXIS_SIDES
    faces = []
    for side in sides:
        # Per face, not per root: a mod that replaces only two of the six
        # faces still needs the base game's other four, and the engine
        # resolves each face's filename independently too.
        found = _plain_source_path(roots_abs, rel_path + side)
        if not found:
            return None
        faces.append(found)
    return faces


def _read_face_topdown(path):
    """One cube face as (size, pixels), pixels[0] being the TOP row.

    Blender stores pixels bottom-up while the engine's cube faces are
    top-down. The face is read through a Non-Color colorspace so the bytes
    arrive raw and can be written straight back out without picking up a
    second sRGB transform on the way.

    Returns a (size, size, 4) float32 array. It used to be a list of lists
    of Python floats, which is what capped the bake at 256x128 - see
    get_or_create_cubemap_equirect().
    """
    img = bpy.data.images.load(path, check_existing=False)
    try:
        width, height = img.size
        if width < 1 or height < 1 or width != height:
            return 0, None
        img.colorspace_settings.name = 'Non-Color'
        buf = np.empty(width * height * 4, dtype=np.float32)
        img.pixels.foreach_get(buf)
    finally:
        bpy.data.images.remove(img)
    # Bottom-up rows to top-down.
    return width, buf.reshape(height, width, 4)[::-1]


def _orient_face(pixels, size, fixup):
    """Apply R_LoadCubeImages' CF_CAMERA fixups to one face.

    R_RotatePic is a transpose (dst[i][j] = src[j][i]), which on a
    (row, column, channel) array is a swap of the first two axes.
    """
    transpose, hflip, vflip = fixup
    if transpose:
        pixels = pixels.transpose(1, 0, 2)
    if hflip:
        pixels = pixels[:, ::-1]
    if vflip:
        pixels = pixels[::-1]
    return pixels


def get_or_create_cubemap_equirect(base_dir, img_expr, camera, note=None,
                                   mod_dir=''):
    """Bake a cubeMap's six faces into one equirectangular image.

    Blender's Environment Texture node samples an equirectangular map by
    direction, which is the nearest thing it has to a cube map, so the faces
    are resampled into that layout once and cached on the image datablock.
    Direction <-> pixel uses Blender's own convention
    (u = -atan2(y, x) / 2pi + 0.5, v = atan2(z, hypot(x, y)) / pi + 0.5) and
    the standard GL cube-face (sc, tc, ma) rules. Both engines are Z-up, so
    no axis swap is needed.

    Returns a flat black image when the faces are not on disk: on an additive
    reflection stage black contributes nothing, which is a far better wrong
    answer than a magenta checker.
    """
    rel = img_expr.base_path() if img_expr is not None else None
    tag = '%s|%s' % (rel, 'camera' if camera else 'axis')
    for existing in bpy.data.images:
        if existing.get('idtech4_cubemap_tag') == tag:
            return existing

    faces = resolve_cube_faces(base_dir, rel, camera, mod_dir=mod_dir) if rel else None
    loaded = []
    size = 0
    if faces:
        for index, path in enumerate(faces):
            face_size, pixels = _read_face_topdown(path)
            # `pixels is None` on purpose: it is an array now, and a bare
            # truth test on one raises rather than returning False.
            if pixels is None or (size and face_size != size):
                loaded = []
                break
            size = face_size
            if camera:
                pixels = _orient_face(pixels, size, _CUBE_CAMERA_FIXUP[index])
            loaded.append(pixels)

    name = '%s_cubemap' % (os.path.basename(rel) if rel else 'cubemap')
    if len(loaded) != 6:
        img = bpy.data.images.new(name, 8, 4, alpha=True, float_buffer=False)
        img.pixels.foreach_set([0.0, 0.0, 0.0, 1.0] * 32)
        _pack_written_pixels(img)
        img['idtech4_cubemap_tag'] = tag
        img['idtech4_cubemap_missing'] = True
        if note is not None:
            note(rel)
        return img

    width, height = _equirect_size(size)
    out = _resample_equirect(loaded, size, width, height)

    img = bpy.data.images.new(name, width, height, alpha=True,
                              float_buffer=False)
    img.pixels.foreach_set(out)
    _pack_written_pixels(img)
    img['idtech4_cubemap_tag'] = tag
    img['idtech4_cubemap_face_size'] = size
    return img


# The equirectangular image spans 360 degrees of azimuth, and one cube face
# spans 90 of them, so a face keeps its own resolution only at width == 4 *
# size. The old bake used `min(256, max(64, size * 2))`, which is a quarter
# of that AND capped: Doom 3's env/cloudy is six 512x512 faces and came out
# 256x128, so each face was reduced to 64 pixels of azimuth - an 8x linear
# loss that reads as a smeared blur on any reflective surface.
#
# The cap existed because the resample was a Python loop over every output
# texel: 0.72s for 256x128, and quadratic, so 2048x1024 would have been ~45
# SECONDS per cube map at import time. _resample_equirect() is vectorised, so
# the 1:1 size is now the cheap option and the cap is only a memory guard
# (4096x2048 RGBA float32 is 128MB in flight).
_EQUIRECT_MAX_WIDTH = 4096
_EQUIRECT_MIN_WIDTH = 64


def _equirect_size(face_size):
    """(width, height) of the equirect bake for a cube of this face size."""
    width = min(_EQUIRECT_MAX_WIDTH,
                max(_EQUIRECT_MIN_WIDTH, int(face_size) * 4))
    return width, width // 2


def _resample_equirect(loaded, size, width, height):
    """The six oriented faces as one flat equirectangular RGBA buffer.

    Direction <-> pixel uses Blender's own convention
    (u = -atan2(y, x) / 2pi + 0.5, v = atan2(z, hypot(x, y)) / pi + 0.5) and
    the standard GL cube-face (sc, tc, ma) rules, exactly as the scalar
    version did - this is the same arithmetic evaluated over whole arrays
    instead of one texel at a time.
    """
    row = np.arange(height, dtype=np.float64)
    col = np.arange(width, dtype=np.float64)
    elev = ((row + 0.5) / height - 0.5) * _math.pi
    azim = (0.5 - (col + 0.5) / width) * (_math.pi * 2.0)

    z = np.sin(elev)[:, None] * np.ones((1, width))
    radius = np.cos(elev)[:, None]
    x = radius * np.cos(azim)[None, :]
    y = radius * np.sin(azim)[None, :]

    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)
    # Same tie-breaking as the scalar version: X wins ties, then Y, then Z.
    x_major = (ax >= ay) & (ax >= az)
    y_major = ~x_major & (ay >= az)
    z_major = ~x_major & ~y_major

    face = np.empty((height, width), dtype=np.intp)
    sc = np.empty((height, width), dtype=np.float64)
    tc = np.empty((height, width), dtype=np.float64)
    major = np.empty((height, width), dtype=np.float64)

    for mask, positive, index, s, t, m in (
            (x_major, x > 0, 0, -z, -y, ax),
            (x_major, x <= 0, 1, z, -y, ax),
            (y_major, y > 0, 2, x, z, ay),
            (y_major, y <= 0, 3, x, -z, ay),
            (z_major, z > 0, 4, x, -y, az),
            (z_major, z <= 0, 5, -x, -y, az)):
        sel = mask & positive
        face[sel] = index
        sc[sel] = s[sel]
        tc[sel] = t[sel]
        major[sel] = m[sel]

    # A zero major axis is the degenerate direction the scalar loop skipped
    # (it left the texel black); dividing by 1 and masking to black at the
    # end is the same result without a warning.
    degenerate = major <= 0.0
    safe = np.where(degenerate, 1.0, major)
    column = (0.5 * (sc / safe + 1.0) * size).astype(np.intp)
    line = (0.5 * (tc / safe + 1.0) * size).astype(np.intp)
    np.clip(column, 0, size - 1, out=column)
    np.clip(line, 0, size - 1, out=line)

    faces = np.stack(loaded).astype(np.float32)     # (6, size, size, 4)
    out = faces[face, line, column]
    out[degenerate] = 0.0
    return out.reshape(-1)


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


def resolve_shared_paths():
    """(base_directory, mod_base_directory, materials source), source defaulted.

    An empty Materials Source is not the same as "no materials": every
    idTech4 tree keeps its .mtr files in <base>/materials, and the three
    sibling importers have always defaulted to that when only a Base
    Directory is set. This addon did not, so a perfectly configured Base
    Directory produced "Set a Materials Source in the idTech4 tab's Sources
    panel first" and refused to build anything.

    With a Mod Base configured the derived source is a TUPLE - one
    "materials" folder per root, mod first - not just the mod's. A mod ships
    a handful of .mtr files and inherits the rest, so stopping at the mod's
    own materials folder would hide the base game's whole tree behind them.
    An explicitly configured Materials Source is still used exactly as given:
    that field is how a user says "look here and nowhere else", and a Mod
    Base must not quietly widen it.

    The stored config is left alone - get_shared_paths() still returns
    literally what is in it - because the fallback is derived, not chosen,
    and writing it back would turn a default into a setting the user then has
    to maintain.
    """
    base, mod, source = get_shared_paths()
    if source:
        return base, mod, source
    derived = tuple(
        d for d in (os.path.join(bpy.path.abspath(r), 'materials')
                    for r in shared_search_roots(base, mod))
        if os.path.isdir(d))
    if not derived:
        return base, mod, source
    # One root is the overwhelmingly common case; hand back the plain
    # string there so nothing downstream has to care that the multi-root
    # form exists at all.
    return base, mod, (derived[0] if len(derived) == 1 else derived)


class AssetResolver(object):
    """One import's view of the game base: name in, image datablock out.

    Holds the base directory and a memo of what has already been looked up.
    That memo is the point of the class. Resolving one texture is up to eleven
    os.path.isfile() calls (six extensions, then four .bimage usage codes),
    and a whole-map import asks for the same handful of shared textures
    thousands of times; across a game base the same question is asked tens of
    thousands of times with the same answer. The image datablock cache below
    is separate and stays module-level, because bpy.data.images is one
    namespace for the whole session.
    """

    __slots__ = ('base_dir', 'mod_dir', 'roots_abs', '_paths', '_missing')

    def __init__(self, base_dir, mod_dir=''):
        self.base_dir = base_dir or ''
        self.mod_dir = mod_dir or ''
        # Every lookup goes through this tuple rather than a single
        # directory: an optional Mod Base is searched first, with the Base
        # Directory still searched behind it. Resolved once here because it
        # is the same answer for every one of the tens of thousands of
        # lookups this object memoises.
        self.roots_abs = _abs_roots(self.base_dir, self.mod_dir)
        self._paths = {}
        self._missing = set()

    # -- paths --------------------------------------------------------------

    def resolve(self, rel_path, usage_label=None):
        """Absolute path for rel_path, real or not.

        Mirrors the engine: source file first, compiled .bimage second, and
        failing both the path the file would have had, so the node tree stays
        valid and Blender shows its own missing-image marker.
        """
        key = (rel_path, usage_label)
        hit = self._paths.get(key)
        if hit is None:
            hit = resolve_image_path(self.base_dir, rel_path, usage_label,
                                     mod_dir=self.mod_dir)
            self._paths[key] = hit
        return hit

    def exists(self, rel_path):
        """True if rel_path resolves to something actually on disk."""
        if not rel_path:
            return False
        if rel_path in self._missing:
            return False
        if _plain_source_path(self.roots_abs, rel_path):
            return True
        if _find_bimage_fallback(self.roots_abs, rel_path, None):
            return True
        self._missing.add(rel_path)
        return False

    def program_bimage(self, img_expr, usage_label=None):
        """The pre-baked .bimage for a whole image program, or None.

        None means "build the program out of its parts" - either because the
        parts are on disk, which keeps the conversion editable in the node
        tree, or because no cache exists either.
        """
        return resolve_image_program_bimage(self.base_dir, img_expr,
                                            usage_label, mod_dir=self.mod_dir)

    # -- images -------------------------------------------------------------

    def image(self, rel_path, usage_label=None, colorspace=None):
        """Load (or find) the datablock for rel_path, with its colorspace set.

        colorspace goes through _set_image_colorspace, which refuses to
        downgrade an image already known to be colour data - idTech4 content
        routinely feeds one diffuse .tga to a heightmap stage as well, and
        the setting belongs to the datablock, not to the usage.
        """
        img = load_or_find_image(self.resolve(rel_path, usage_label))
        if colorspace:
            _set_image_colorspace(img, colorspace)
        return img

    def image_at(self, abs_path, colorspace=None):
        """As image(), for a path that has already been resolved."""
        img = load_or_find_image(abs_path)
        if colorspace:
            _set_image_colorspace(img, colorspace)
        return img

    def cubemap(self, img_expr, camera, note=None):
        """Six cube faces baked to one equirectangular image, or None.

        Blender has no cube-map texture node, so the faces are resampled into
        an equirect at face_size * 4 and read back with an Environment Texture
        node. note, if given, is called with a diagnostic string.
        """
        return get_or_create_cubemap_equirect(self.base_dir, img_expr, camera,
                                              note=note, mod_dir=self.mod_dir)

    def cube_faces(self, rel_path, camera):
        return resolve_cube_faces(self.base_dir, rel_path, camera,
                                  mod_dir=self.mod_dir)

    # -- reporting ----------------------------------------------------------

    def search_paths(self, rel_path, usage_label=None):
        """Every path resolve() would have tried, for a not-found report."""
        return image_search_paths(self.base_dir, rel_path, usage_label,
                                  mod_dir=self.mod_dir)

    # -- generated stand-ins ------------------------------------------------

    def placeholder(self):
        return get_or_create_placeholder_image()

    def glass_placeholder(self, name_hint, base_alpha=0.15):
        return get_or_create_glass_placeholder_image(name_hint, base_alpha)


# ===========================================================================
# END ASSET RESOLUTION
# ===========================================================================


# ===========================================================================
# BEGIN PARAMETER POLICY
#
# idTech4 material expressions read `time`, `parm0..11`, `global0..7`, `sound`
# and declared tables. What this addon does with those is an axis of its own,
# orthogonal to the fidelity mode, and it is on its own a bigger performance
# lever than a whole mode step: on mars_city1, 219 drivers cost 74ms of a 91ms
# frame while evaluating all 219 expressions was 0.3ms of it. The rest was
# EEVEE rebuilding each material's GPU shader because a driver touched it.
#
# Three policies, one interface. Every expression in the entire builder goes
# through resolve() and every stage through stage_verdict(), so the twelve
# mode x policy combinations need no cross-product code anywhere.
# ===========================================================================

PARAMS_DYNAMIC = 'DYNAMIC'
PARAMS_BAKED = 'BAKED'
PARAMS_SKIP = 'SKIP'

VERDICT_BUILD = 'BUILD'     # build the stage unconditionally
VERDICT_DROP = 'DROP'       # do not create the stage's nodes at all
VERDICT_GATE = 'GATE'       # build it, gated on a live condition


class Resolution(object):
    """What one policy decided about one expression.

    value    the float to write into the socket right now
    driver   a driver-syntax expression string, or None
    record   the .mtr expression text to store for slider-refold, or None
    dropped  True: the term is refused outright and the caller must use its
             own neutral default (rgb 1, alpha 1, no transform, no alphaTest)
             rather than this value
    """

    __slots__ = ('value', 'driver', 'record', 'dropped')

    def __init__(self, value=0.0, driver=None, record=None, dropped=False):
        self.value = value
        self.driver = driver
        self.record = record
        self.dropped = dropped

    @property
    def is_live(self):
        """True if this resolution needs anything beyond a plain socket value."""
        return self.driver is not None or self.record is not None

    def __repr__(self):
        return 'Resolution(%r, driver=%r, record=%r, dropped=%r)' % (
            self.value, self.driver, self.record, self.dropped)


class TransformPlan(object):
    """What a policy decided about a stage's whole texture-matrix chain.

    kind is one of:
      'none'    no transform at all - the stage uses the default UV directly,
                which for an Image Texture node means no node at all
      'matrix'  the whole chain folded to one constant 2x3, expressible as a
                single Mapping node's Location/Rotation/Scale
      'chain'   one Mapping (or shear group) per transform, some of them live

    signature is what the ResourcePlan keys UV chains on: two stages whose
    signatures match share one chain of nodes.
    """

    __slots__ = ('kind', 'matrix', 'transforms', 'signature')

    def __init__(self, kind, matrix=None, transforms=(), signature=()):
        self.kind = kind
        self.matrix = matrix
        self.transforms = tuple(transforms)
        self.signature = signature


TRANSFORM_NONE = TransformPlan('none', signature=())


class ParameterPolicy(object):
    """Base class; also the DYNAMIC implementation's shared machinery."""

    name = PARAMS_DYNAMIC

    def __init__(self, ctx, fps=24.0):
        self.ctx = ctx              # MtrEvalContext
        self.fps = fps if fps > 0 else 24.0

    # -- expressions --------------------------------------------------------

    def fold(self, expr, default=0.0):
        """The expression's value right now, with no policy attached."""
        if expr is None:
            return default
        try:
            return eval_expr(expr, self.ctx)
        except (ValueError, TypeError, ZeroDivisionError, AttributeError):
            return default

    def resolve(self, expr, default=1.0):
        raise NotImplementedError

    # -- stages -------------------------------------------------------------

    def stage_verdict(self, stage):
        raise NotImplementedError

    # -- texture matrices ---------------------------------------------------

    def transform_chain(self, stage):
        raise NotImplementedError

    # -- shared helpers -----------------------------------------------------

    def _fold_matrix(self, transforms):
        """Compose a list of transforms into one row-major 2x3, or None.

        MultiplyTextureMatrix computes new = old * reg, so the LAST keyword
        written in the .mtr is applied to the texture coordinates FIRST. The
        composition order here is the .mtr's own order and _mat_mul carries
        the right-multiplication; walking the built chain is what has to run
        in reverse.
        """
        m = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        for tform in transforms:
            m = _mat_mul(m, _transform_matrix(tform, self.ctx))
        return m

    def _static_chain(self, stage):
        """Fold a wholly-constant transform chain, or None if it cannot be.

        A shear is not expressible as a Mapping node, and a surviving rotation
        or shear term (m[1]/m[3]) needs the Rotation input, which the folded
        form does not carry - both fall back to the node chain.
        """
        if any(t.op == 'shear' for t in stage.transforms):
            return None
        m = self._fold_matrix(stage.transforms)
        if abs(m[1]) > 1e-6 or abs(m[3]) > 1e-6:
            return None
        if _mat_is_identity(m):
            return TRANSFORM_NONE
        return TransformPlan('matrix', matrix=tuple(m),
                             signature=('m',) + tuple(round(v, 6) for v in m))


class DynamicParameters(ParameterPolicy):
    """Drivers for what actually moves; plain values plus a record for the rest.

    The split is the whole point. Only `time` advances on its own; parm0..11,
    global0..7, sound and spectrum change when the user moves a panel slider
    and at no other moment. An expression that reaches `time` gets a scripted
    driver. One that is dynamic but not time-varying gets a folded value and a
    record on the node tree, refolded the instant a slider moves - which is the
    only instant it can change. On mars_city1 that took 219 drivers to 51 and
    playback from ~15 fps to ~23.
    """

    name = PARAMS_DYNAMIC

    def resolve(self, expr, default=1.0):
        if expr is None:
            return Resolution(default)
        value = self.fold(expr, default)
        if expr.is_time_varying():
            return Resolution(value, driver=expr_to_driver(expr, self.fps),
                              record=None)
        if expr.is_dynamic():
            # Driver syntax, NOT expr.source. A record is re-evaluated by
            # refresh_parameters() against bpy.app.driver_namespace - the same
            # namespace a driver resolves through. That namespace defines
            # idtech4_parm/_global/_sound/_tbl and has never defined a bare
            # `parm6`, so storing the .mtr text made every record raise
            # NameError on refold. refresh_parameters() swallows a raising
            # record exactly as Blender swallows a raising driver, which is why
            # the whole static half of this policy failed silently: 794 of the
            # 795 records the Quake 4 corpus emits could not be evaluated.
            # The .mtr text is wrong twice over - `source` keeps the original
            # casing (`Parm0`, `decalFade[ ... ]`) while the parser lowercases
            # EXPR_VAR names, so even injecting `parm0..11` as plain names
            # would not have rescued it.
            return Resolution(value, driver=None,
                              record=expr_to_driver(expr, self.fps))
        return Resolution(value)

    def stage_verdict(self, stage):
        if stage.condition is None:
            return VERDICT_BUILD
        return VERDICT_GATE

    def transform_chain(self, stage):
        if not stage.transforms:
            return TRANSFORM_NONE
        if not any(t.is_dynamic() for t in stage.transforms):
            folded = self._static_chain(stage)
            if folded is not None:
                return folded
        sig = ['c']
        for tform in stage.transforms:
            sig.append((tform.op,
                        tform.x.source if tform.x is not None else '',
                        tform.y.source if tform.y is not None else ''))
        return TransformPlan('chain', transforms=stage.transforms,
                             signature=tuple(sig))


class BakedParameters(ParameterPolicy):
    """Everything folded once, at build time, against the current panel values.

    No drivers, no static records, and - the part that matters - no dependency
    on bpy.app.driver_namespace at all, which is why this is the default and
    the mode to recommend for a whole-map import. A material built this way
    cannot be broken by the namespace being wiped on file load.

    The cost is that a slider move cannot refold it: changing parm3 means
    rebuilding the material.
    """

    name = PARAMS_BAKED

    def resolve(self, expr, default=1.0):
        if expr is None:
            return Resolution(default)
        return Resolution(self.fold(expr, default))

    def stage_verdict(self, stage):
        """Evaluate the condition now: a false one means the stage is not built.

        The engine skips a conditioned-off stage entirely (idMaterial's
        registers drive ParseStage's conditionRegister), so dropping its nodes
        is what the frame would have looked like - not a hidden branch.
        """
        if stage.condition is None:
            return VERDICT_BUILD
        return VERDICT_BUILD if self.fold(stage.condition, 1.0) != 0.0 \
            else VERDICT_DROP

    def transform_chain(self, stage):
        if not stage.transforms:
            return TRANSFORM_NONE
        folded = self._static_chain(stage)
        if folded is not None:
            return folded
        # A shear or a surviving rotation still needs its nodes, but with
        # every term already a number.
        sig = ['b']
        for tform in stage.transforms:
            sig.append((tform.op,
                        round(self.fold(tform.x, 0.0), 6),
                        round(self.fold(tform.y, 0.0), 6)))
        return TransformPlan('chain', transforms=stage.transforms,
                             signature=tuple(sig))


class SkipParameters(ParameterPolicy):
    """Refuse every parameter. Nothing is looked up, nothing is evaluated.

    A conditional stage is dropped without asking what its condition says, and
    any term that is dynamic at all comes back `dropped`, which tells the
    caller to use its own neutral default instead of a number. That is a real
    simplification, not a cheaper evaluation: the resulting material has no
    expression machinery in it whatsoever.

    This empties some materials outright. That is what the visibility
    guarantee exists for.
    """

    name = PARAMS_SKIP

    def resolve(self, expr, default=1.0):
        if expr is None:
            return Resolution(default)
        if expr.is_dynamic():
            return Resolution(default, dropped=True)
        return Resolution(self.fold(expr, default))

    def stage_verdict(self, stage):
        return VERDICT_DROP if stage.condition is not None else VERDICT_BUILD

    def transform_chain(self, stage):
        if not stage.transforms:
            return TRANSFORM_NONE
        if any(t.is_dynamic() for t in stage.transforms):
            return TRANSFORM_NONE
        folded = self._static_chain(stage)
        return folded if folded is not None else TRANSFORM_NONE


_POLICIES = {
    PARAMS_DYNAMIC: DynamicParameters,
    PARAMS_BAKED: BakedParameters,
    PARAMS_SKIP: SkipParameters,
}


def make_policy(name, ctx, fps=24.0):
    return _POLICIES.get(name, BakedParameters)(ctx, fps)


# ---------------------------------------------------------------------------
# Texture-matrix arithmetic, shared by every policy.
# ---------------------------------------------------------------------------

def _transform_matrix(tform, ctx):
    """One texture-matrix keyword as a row-major 2x3 [m00,m01,m02,m10,m11,m12].

    These are idMaterial::ParseStage's own matrices. `rotate` is in CYCLES,
    not degrees - the engine indexes sinTable/cosTable directly with it - and
    it rotates about (0.5, 0.5), which is where the extra translation terms
    come from. The per-op default matters: an omitted scale term is 1, an
    omitted translation term is 0.
    """
    op = tform.op
    a = eval_or(tform.x, ctx, 0.0 if op in ('translate', 'scroll', 'shear',
                                            'rotate') else 1.0)
    b = eval_or(tform.y, ctx, 0.0 if op in ('translate', 'scroll', 'shear')
                else 1.0)
    if op in ('translate', 'scroll'):
        return [1.0, 0.0, a, 0.0, 1.0, b]
    if op == 'scale':
        return [a, 0.0, 0.0, 0.0, b, 0.0]
    if op == 'centerscale':
        return [a, 0.0, 0.5 - 0.5 * a, 0.0, b, 0.5 - 0.5 * b]
    if op == 'shear':
        return [1.0, a, -0.5 * a, b, 1.0, -0.5 * b]
    if op == 'rotate':
        angle = a * 2.0 * math.pi
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return [cos_a, -sin_a, -0.5 * cos_a + 0.5 * sin_a + 0.5,
                sin_a, cos_a, -0.5 * sin_a - 0.5 * cos_a + 0.5]
    return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def _mat_mul(old, reg):
    """idMaterial::MultiplyTextureMatrix - new = old composed with reg, where
    reg is applied to the texture coordinates first."""
    return [
        old[0] * reg[0] + old[1] * reg[3],
        old[0] * reg[1] + old[1] * reg[4],
        old[0] * reg[2] + old[1] * reg[5] + old[2],
        old[3] * reg[0] + old[4] * reg[3],
        old[3] * reg[1] + old[4] * reg[4],
        old[3] * reg[2] + old[4] * reg[5] + old[5],
    ]


def _mat_is_identity(m):
    ident = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    return all(abs(m[i] - ident[i]) < 1e-6 for i in range(6))


# ===========================================================================
# END PARAMETER POLICY
# ===========================================================================


# ===========================================================================
# BEGIN DRIVER INFRASTRUCTURE
#
# What the DYNAMIC policy's drivers and static records resolve through at
# runtime. BAKED needs none of this, which is the point of BAKED: a material
# built that way cannot be broken by any of it going missing.
# ===========================================================================

# The live table registry drivers look up through idtech4_tbl(). Populated
# from the parsed .mtr tree at generation time, and rehydrated from the .blend
# on file load - see _idtech4_load_post below.
_TABLE_REGISTRY = {}    # {lowercased name: MtrTable}

_DRIVER_TABLE_RE = re.compile(r"idtech4_tbl\('([^']+)'")


def _load_tables_into_registry(db):
    """Publish a parsed database's tables to the driver namespace.

    Everything the tree declares, because at this point nothing has been built
    yet and there is no way to know which tables the materials about to be
    built will reach for. _prune_table_registry() cuts it back to the ones
    actually in use once they have been.
    """
    _TABLE_REGISTRY.clear()
    _TABLE_REGISTRY.update(db.tables)


def _tables_referenced_by_expressions():
    """Lowercased names of every table this .blend looks up, live.

    BOTH halves of the DYNAMIC policy, which is the whole point: a table
    indexed by `time` is carried by a driver, but one indexed by a parm -
    `decalFade[ parm7 ]` - is dynamic without being time-varying, so it is
    carried by a static record on the node tree instead and never appears in
    animation_data.drivers at all. Scanning only the drivers would prune such
    a table out of the registry, and _table_lookup_fn returns 0.0 for a table
    it cannot find rather than raising - so the socket would refold to a
    silently wrong number instead of failing loudly. The pruned list is also
    what gets persisted into created_tables and rehydrated on file load, so
    the mistake would survive the save.

    Walks all of bpy.data.materials, not just the ones a given pass built, so
    a second run over a different set of materials cannot strand the first
    run's expressions on a table that is no longer registered. Everything this
    addon writes lands on a material's own node tree, so there is nowhere else
    to look.
    """
    used = set()
    for mat in bpy.data.materials:
        if not (mat.use_nodes and mat.node_tree):
            continue
        nt = mat.node_tree
        anim = nt.animation_data
        for fcurve in (anim.drivers if anim else ()):
            for match in _DRIVER_TABLE_RE.finditer(fcurve.driver.expression):
                used.add(match.group(1).lower())
        for _path, _index, expression in _read_static_expressions(nt):
            for match in _DRIVER_TABLE_RE.finditer(expression):
                used.add(match.group(1).lower())
    return used


def _prune_table_registry(used):
    """Drop registered tables nothing refers to.

    Doom 3 declares 198 tables and Quake 4 321, of which a map typically
    reaches for a couple of dozen. Keeping the rest serves nothing: a table
    nothing looks up cannot change what is drawn, and it is one more row
    between the user and the tables that do matter.
    """
    for name in [n for n in _TABLE_REGISTRY if n not in used]:
        del _TABLE_REGISTRY[name]


def _settings():
    try:
        return bpy.context.scene.idtech4_settings
    except (AttributeError, KeyError):
        return None


def _table_lookup_fn(table_name, index):
    """bpy.app.driver_namespace['idtech4_tbl'].

    Consults the DECLARED table, including sinTable and cosTable. Doom 3's
    sinTable is a 256-entry table over one full period returning -1..1;
    substituting an analytic (sin(2*pi*x)+1)/2 remap silently halves and
    biases every rotation, flicker and pulse driven through it.
    """
    table = _TABLE_REGISTRY.get(str(table_name).lower())
    if table is None:
        return 0.0
    return table.lookup(index)


def _parm_fn(parm_index):
    settings = _settings()
    if settings is None:
        return 0.0
    try:
        return float(settings.shader_parms[int(parm_index)])
    except (TypeError, ValueError, IndexError, AttributeError):
        return 0.0


def _global_fn(global_index):
    """global0..7 are renderer-wide shader parms, distinct from the per-entity
    parm0..11. There is no way to know their live in-game values from a static
    import, so they are backed by scene properties for preview purposes."""
    settings = _settings()
    if settings is None:
        return 0.0
    try:
        return float(settings.global_parms[int(global_index)])
    except (TypeError, ValueError, IndexError, AttributeError):
        return 0.0


def _sound_fn():
    """OP_TYPE_SOUND - the current sound amplitude on the entity. Roughly 500
    light materials modulate their brightness with it. Exposed as a scene
    slider so the effect can at least be previewed."""
    settings = _settings()
    if settings is None:
        return 0.0
    return float(getattr(settings, 'sound_amplitude', 0.0))


def _spectrum_fn():
    """The scene's current Spectrum. A material declaring `spectrum N` is
    visible only when this matches."""
    settings = _settings()
    if settings is None:
        return 0
    return int(getattr(settings, 'spectrum', 0))


def _register_driver_namespace():
    ns = bpy.app.driver_namespace
    ns[DRIVER_TABLE_FN] = _table_lookup_fn
    ns[DRIVER_PARM_FN] = _parm_fn
    ns[DRIVER_GLOBAL_FN] = _global_fn
    ns[DRIVER_SOUND_FN] = _sound_fn
    ns[DRIVER_SPECTRUM_FN] = _spectrum_fn


def _unregister_driver_namespace():
    ns = bpy.app.driver_namespace
    for name in (DRIVER_TABLE_FN, DRIVER_PARM_FN, DRIVER_GLOBAL_FN,
                 DRIVER_SOUND_FN, DRIVER_SPECTRUM_FN):
        ns.pop(name, None)


def _rehydrate_table_registry(scene=None):
    """Rebuild _TABLE_REGISTRY from what the .blend already saved.

    settings.created_tables stores each table's name, values, clamp and snap,
    and always has - it was written on every generation run and never once
    read back. Without this, reopening a file leaves idtech4_tbl() with an
    empty registry, and every table-driven expression returns 0.0.
    """
    if scene is None:
        scene = getattr(bpy.context, 'scene', None)
    settings = getattr(scene, 'idtech4_settings', None) if scene else None
    if settings is None:
        return 0
    restored = 0
    for item in getattr(settings, 'created_tables', ()):
        try:
            values = json.loads(item.entries_json)
        except (TypeError, ValueError):
            continue
        if not isinstance(values, list):
            continue
        table = MtrTable(item.table_name, bool(item.is_snap),
                         bool(item.is_clamp), [float(v) for v in values])
        _TABLE_REGISTRY[item.table_name.lower()] = table
        restored += 1
    return restored


@bpy.app.handlers.persistent
def _idtech4_load_post(_dummy):
    """Put the driver namespace back after a file load.

    bpy.app.driver_namespace survives a SAVE but is wiped by wm.open_mainfile
    and wm.read_homefile, and an already-enabled addon does not get register()
    called again on file load - so idtech4_tbl / _parm / _global / _sound /
    _spectrum were simply gone for the rest of the session. Blender then marks
    each driver is_valid == False and the socket reads 0.0, NOT the last good
    value. On an additive stage that is black, so the surface disappears
    outright (textures/sfx/bioscanbeam in mars_city1 is the reference case).

    It never reproduces in-session, which is exactly why it read as "it used
    to work".

    Both halves are needed: the namespace functions, and the table data they
    look up. BAKED-mode materials need neither, and that is worth knowing when
    choosing a policy.
    """
    _register_driver_namespace()
    try:
        _rehydrate_table_registry()
    except Exception as exc:                                # noqa: BLE001
        print('idTech4 Materials: table registry not restored (%s)' % exc)


# ---------------------------------------------------------------------------
# Static shader-parm expressions
# ---------------------------------------------------------------------------
# The other half of the DYNAMIC policy. A socket whose value depends on
# parm0..11, global0..7, sound or spectrum - but not on time - carries a plain
# number plus a record of the expression that produced it, instead of a driver
# that would re-run on every frame to produce the same number.
#
# The record lives on the node TREE as a JSON list of
# [socket path, array index or -1, driver-syntax expression], so it survives
# save/load with the material it belongs to.

_STATIC_EXPR_PROP = 'idtech4_static_exprs'


def _store_static_expressions(nt, records):
    """Write the records whose sockets survived the orphan pass."""
    kept = []
    for path, index, expression in records:
        try:
            nt.path_resolve(path)
        except (ValueError, AttributeError, TypeError):
            continue                       # its node was pruned as an orphan
        kept.append([path, index, expression])
    if kept:
        try:
            nt[_STATIC_EXPR_PROP] = json.dumps(kept)
        except (TypeError, ValueError):
            pass
    else:
        nt.pop(_STATIC_EXPR_PROP, None)


def _read_static_expressions(nt):
    raw = nt.get(_STATIC_EXPR_PROP) if nt is not None else None
    if not raw:
        return []
    try:
        records = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return records if isinstance(records, list) else []


def refresh_parameters(context=None):
    """Re-fold every recorded parm/global/sound/spectrum socket.

    Returns (materials touched, sockets written). Cheap enough to hang off a
    slider's update callback: it visits only materials that carry a record,
    and evaluates the same expression text a driver would have evaluated - so
    a refolded socket and a driven one cannot disagree.
    """
    _register_driver_namespace()
    scene = getattr(context, 'scene', None) if context is not None else None
    if scene is None:
        scene = getattr(bpy.context, 'scene', None)
    namespace = dict(bpy.app.driver_namespace)
    namespace['frame'] = float(scene.frame_current) if scene else 0.0
    materials = 0
    sockets = 0
    for material in bpy.data.materials:
        nt = material.node_tree
        if nt is None:
            continue
        records = _read_static_expressions(nt)
        if not records:
            continue
        touched = False
        for record in records:
            try:
                path, index, expression = record
                socket = nt.path_resolve(path)
            except (ValueError, AttributeError, TypeError):
                continue
            try:
                value = float(eval(expression, namespace))   # noqa: S307
            except Exception:                                # noqa: BLE001
                # Exactly what a driver does when its expression throws: leave
                # the socket alone rather than poison it with a zero.
                continue
            try:
                if index is None or index < 0:
                    socket.default_value = value
                else:
                    socket.default_value[index] = value
            except (TypeError, AttributeError, IndexError, ValueError):
                continue
            touched = True
            sockets += 1
        if touched:
            materials += 1
            nt.update_tag()
    return materials, sockets


# ===========================================================================
# END DRIVER INFRASTRUCTURE
# ===========================================================================


# ===========================================================================
# BEGIN RESOURCE PLAN
#
# Scan the material for every image and every UV chain it will need, compile
# the list, then build one node for each. The old builder created a fresh
# ShaderNodeTexImage on every call, so a material that reads one texture from
# four stages carried four copies of it - 74,897 stage image references across
# the corpus resolve to 71,190 distinct images, and 2,074 materials reuse one.
#
# The cache keys have to be honest or dedup becomes a bug. A ShaderNodeTexImage
# owns its own Vector input AND its own `extension` and `interpolation`, so two
# stages reading the same file with different `clamp` or `nearest`, or with
# different texture matrices, genuinely cannot share one node.
# ===========================================================================

# What a stage's texture is for. Drives the .bimage usage code the engine
# would have compiled the file under, and the colorspace.
USE_DIFFUSE = 'Diffuse'
USE_SPECULAR = 'Specular'
USE_NORMAL = 'Normal Map'
USE_DEFAULT = None


class Sampling(object):
    """The two properties an Image Texture node owns about how it reads.

    Both come from the stage, and both are part of the dedup key because they
    live on the node rather than on the image datablock.

    The repeat mode belongs to the idImage, fixed when the stage's map is
    loaded (the textureRepeat_t argument to ImageFromFile), so it applies to a
    bumpmap stage exactly as it does to a diffuse one. All three clamp modes
    bind GL_CLAMP_TO_EDGE; what separates them is that zeroclamp and
    alphazeroclamp additionally overwrite the outermost ring of texels through
    R_SetBorderTexels, so a clamped edge reads as nothing rather than as a
    smear of the border pixel. Blender has no per-image border colour, and
    CLIP is the closest single setting to that intent.
    """

    __slots__ = ('interpolation', 'extension')

    def __init__(self, interpolation='Linear', extension='REPEAT'):
        self.interpolation = interpolation
        self.extension = extension

    @classmethod
    def for_stage(cls, stage):
        interp = 'Closest' if stage.filter == 'nearest' else 'Linear'
        ext = 'REPEAT'
        if stage.wrap == WRAP_CLAMP:
            ext = 'EXTEND'
        elif stage.wrap in (WRAP_ZERO, WRAP_ZERO_ALPHA):
            ext = 'CLIP'
        return cls(interp, ext)


DEFAULT_SAMPLING = Sampling()


class ImageRequest(object):
    """One Image Texture node the material is going to need."""

    __slots__ = ('key', 'path', 'colorspace', 'sampling', 'uv_key', 'label',
                 'order')

    def __init__(self, key, path, colorspace, sampling, uv_key, label, order):
        self.key = key
        self.path = path
        self.colorspace = colorspace
        self.sampling = sampling
        self.uv_key = uv_key
        self.label = label
        self.order = order


class UVRequest(object):
    """One texture-coordinate chain the material is going to need."""

    __slots__ = ('key', 'transform', 'texgen', 'texgen_args', 'order')

    def __init__(self, key, transform, texgen, texgen_args, order):
        self.key = key
        self.transform = transform
        self.texgen = texgen
        self.texgen_args = texgen_args
        self.order = order

    @property
    def is_default(self):
        """True when this chain needs no nodes at all.

        An Image Texture node with an unconnected Vector input already reads
        the active UV map, which is exactly what "no texgen, no transform"
        means - so the cheapest correct chain is no chain.
        """
        return self.texgen in (None, '', 'base') and \
            self.transform.kind == 'none'


class ResourcePlan(object):
    """Every image and UV chain one material needs, keyed and ordered.

    Ordered because the layout depends on it: images go in one column and UV
    chains in the column to their left, in plan order, so the graph is
    readable in the shader editor without a post-hoc pass that tries to infer
    structure from an arbitrary graph.
    """

    __slots__ = ('images', 'uv_chains', 'groups', 'stages',
                 'dropped_stages', 'notes')

    def __init__(self):
        self.images = {}            # ImageKey -> ImageRequest
        self.uv_chains = {}         # UVKey    -> UVRequest
        self.groups = set()         # shared node-group names needed
        self.stages = []            # (stage, verdict) in engine draw order
        # (stage, reason, level, kind) for the report - see drop() below.
        # Every stage the engine would have drawn and this build does not
        # goes here, whichever of the three reasons it was: the parameter
        # policy refused it, the fidelity mode cannot or will not build it,
        # or the engine itself would not have drawn it (an invisible blend,
        # a shaderLevel duplicate, a render-target map).
        self.dropped_stages = []
        self.notes = []

    def drop(self, stage, reason, level=DIAG_APPROXIMATED,
             kind='stage-dropped'):
        """Record a stage as not built, with why - once per stage.

        _engine_stage runs a second time inside _rescue_empty_ambient, so
        the same stage can reach here twice for one material; the report
        should still name it once.

        `kind` is what the grouped report buckets by, so a drop that is a
        real capability gap keeps the kind it has always had (`program
        refraction`, `renderMap`) rather than being flattened into the
        generic one - that is the string tests/maximum_fidelity.py matches
        a mismatch against to decide it was declared rather than wrong.
        """
        for entry in self.dropped_stages:
            if entry[0] is stage:
                return
        self.dropped_stages.append((stage, reason, level, kind))

    def add_note(self, level, kind, message):
        """A material-level note, recorded once however often it is raised.

        Same reason drop() dedups: the ambient rescue walks the stages a
        second time, so a note raised from inside that walk - a video stage
        standing in for itself, say - was landing in the report twice for
        one material with nothing to tell the two copies apart.
        """
        entry = (level, kind, message)
        if entry not in self.notes:
            self.notes.append(entry)

    # -- lookup used by the graph -------------------------------------------

    def image_key(self, path, colorspace, sampling, uv_key):
        return (path, colorspace, sampling.interpolation, sampling.extension,
                uv_key)

    def want_image(self, path, colorspace, sampling, uv_key, label=''):
        key = self.image_key(path, colorspace, sampling, uv_key)
        req = self.images.get(key)
        if req is None:
            req = ImageRequest(key, path, colorspace, sampling, uv_key, label,
                               len(self.images))
            self.images[key] = req
        return req

    def want_uv(self, transform, texgen, texgen_args):
        key = (texgen or '', tuple(texgen_args or ()), transform.signature)
        req = self.uv_chains.get(key)
        if req is None:
            req = UVRequest(key, transform, texgen, texgen_args,
                            len(self.uv_chains))
            self.uv_chains[key] = req
        return req

    def want_group(self, name):
        self.groups.add(name)

    # -- the pre-pass -------------------------------------------------------

    @classmethod
    def build(cls, mat_ir, profile, params, assets):
        """Walk the material once, deciding what will be needed.

        Nothing is created here - this only answers "what nodes, and how many
        of each". The builder then creates exactly that set, which is what
        makes the layout deterministic and the orphan sweep a sanity check
        rather than a load-bearing pass.
        """
        plan = cls()
        if mat_ir is None:
            return plan

        has_fallback = _material_has_shader_fallback(mat_ir)
        for stage in mat_ir.stages:
            verdict = params.stage_verdict(stage)
            if verdict == VERDICT_DROP:
                plan.drop(stage, _params_drop_reason(stage, params))
                continue
            # What the engine would draw comes first; the mode only gets an
            # opinion about stages that would have reached the framebuffer.
            drawn = _engine_stage(mat_ir, stage, plan, has_fallback,
                                  params)
            if drawn is None:
                continue
            if not _profile_wants_stage(drawn, profile):
                plan.drop(drawn, _profile_drop_reason(drawn, profile))
                continue
            plan.stages.append((drawn, verdict))

        # The ambient cap runs here, before anything is planned, so a capped
        # stage costs nothing at all - not even a resolved path. It runs after
        # the profile filter so that "keep 4" means four stages that would
        # actually have been built.
        _apply_ambient_cap(plan, profile)
        _rescue_empty_ambient(plan, mat_ir, profile, params)

        for stage, _verdict in plan.stages:
            _plan_stage(plan, stage, profile, params, assets)

        if profile.specular == 'roughness_only':
            plan.want_group(ROUGHNESS_GROUP_NAME)
        return plan


def _profile_can_build_stage(stage, profile):
    """Whether this mode can build this stage AT ALL.

    Capability, not thinning. A mode that declares it loads no specular
    textures or bakes no cube maps genuinely cannot draw those stages, and
    nothing - including the empty-material rescue below - may build one
    behind that declaration. Doing so made Basic and Simple bake the 2048x1024
    equirect that Good drops precisely to save it, so the lower modes used
    MORE video memory than the higher one on all 105 of Doom 3's cube-mapped
    materials.
    """
    lighting = stage.lighting
    if lighting == 'bump' and not profile.normals:
        return False
    if lighting == 'specular' and not profile.wants_specular_texture:
        return False
    if lighting == 'parallax' and not profile.heightmaps:
        return False
    if stage.tex_kind == TEX_CUBE and not profile.cubemaps:
        return False
    if lighting == 'ambient' and profile.ambient == 'none':
        return False
    return True


def _profile_wants_stage(stage, profile):
    """Whether this mode builds this stage, capability AND thinning."""
    if not _profile_can_build_stage(stage, profile):
        return False
    lighting = stage.lighting
    if lighting == 'ambient' and profile.ambient == 'alpha_only':
        # Basic and Simple build only the ambient stages that carry alpha -
        # decals, overlays and cutouts. That is where their rung is earned:
        # four additive stages measured +31%, and the lit surface alone
        # measured the same as Good's.
        if not _stage_carries_alpha(stage):
            return False
    return True


def _rescue_empty_ambient(plan, mat_ir, profile, params):
    """Thin the ambient stack; never delete the surface with it.

    The alpha_only filter exists to shed the additive glow stacks that cost
    31% for four stages. But a material whose ONLY stages are additive ambient
    ones - an sfx glow, a sky - IS that stack, and dropping all of it left 214
    materials in the first 1,200 of Doom 3 falling through to a placeholder
    that told the user less than one stage would have.

    So the thinning filters keep their saving on the stack and give it up on
    the last stage: if nothing survived, the first ambient stage the engine
    would have drawn comes back. That is one stage, not a stack, so the
    measured cost of the rung is unaffected.

    It runs for EVERY mode, not only the thinning ones. Gated on alpha_only it
    left Good - which does no thinning but does drop cube maps - falling back
    on the 105 cube-mapped Doom 3 materials while Basic and Simple drew them,
    so a lower rung produced a better result than a higher one.

    And it restores only what the mode CAN build: a capability the profile has
    declared it does not have is not something to fall back on.
    """
    if plan.stages or not mat_ir.stages:
        return
    has_fallback = _material_has_shader_fallback(mat_ir)
    for stage in mat_ir.stages:
        if stage.lighting != 'ambient':
            continue
        if params.stage_verdict(stage) == VERDICT_DROP:
            continue
        if not _profile_can_build_stage(stage, profile):
            continue
        drawn = _engine_stage(mat_ir, stage, plan, has_fallback, params)
        if drawn is None:
            continue
        plan.stages.append((drawn, VERDICT_BUILD))
        plan.dropped_stages = [entry for entry in plan.dropped_stages
                               if entry[0] is not drawn]
        plan.add_note(
            DIAG_APPROXIMATED, 'ambient-rescued',
            'this mode had dropped every ambient stage, which would have left '
            'nothing to draw; the first one it can build is kept')
        return


def _stage_carries_alpha(stage):
    """The ambient stages the alpha_only modes still build.

    Note this is NOT stage.writes_alpha, which asks whether the stage's alpha
    WRITES are masked and is true of almost every stage there is. The question
    here is whether the stage's own contribution is gated by an alpha channel,
    which is what makes it a decal, an overlay or a cutout rather than one
    more layer of glow on the additive stack that these modes exist to shed.

    A stage that does not blend with the destination at all is kept whatever
    its alpha: `blend gl_one, gl_zero` overwrites, so it is the surface, and
    it costs nothing to stack because there is no stack.
    """
    if stage.alpha_test is not None:
        return True
    if stage.reads_dest_alpha:
        return True
    if not stage.blends_with_destination():
        return True
    src, dst = stage.blend_pair()
    if src == 'gl_src_alpha' or dst == 'gl_one_minus_src_alpha':
        return True
    # A multiplicative blend is a decal by construction - a skid mark, a
    # stain, a logo - and it is reproduced EXACTLY by one tinted Transparent
    # BSDF, so it costs two nodes and no shader stack. These modes shed the
    # additive glow stacks, not the decals.
    return _BLEND_STRATEGY.get(stage.blend_pair()) in ('filter', 'invfilter')


def _params_drop_reason(stage, params):
    """Why the parameter policy refused this stage, in its own terms.

    Only a stage carrying an `if` is ever refused (see each policy's
    stage_verdict), and the two policies that can refuse one do it for
    opposite reasons: Skip will not look at a condition at all, Baked looked
    and it came out false. The third case is unreachable today and is worded
    so it stays honest if a future policy adds one.
    """
    if stage.condition is None:
        return 'refused by the %s parameter policy' % params.name.lower()
    if params.name == PARAMS_SKIP:
        return ('the Skip parameter policy drops every conditional stage '
                'without evaluating it')
    return ('its `if` condition folded to false at the current frame and '
            'shader parm values')


def _profile_drop_reason(stage, profile):
    lighting = stage.lighting
    if lighting == 'bump':
        return 'mode builds no normal maps'
    if lighting == 'specular':
        return 'mode loads no specular textures'
    if lighting == 'parallax':
        return 'mode builds no heightmaps'
    if stage.tex_kind == TEX_CUBE:
        return 'mode builds no cube maps'
    return ('mode builds only ambient stages that carry alpha; this one is '
            '%s,%s' % stage.blend_pair())


def _apply_ambient_cap(plan, profile):
    """Keep the first N ambient stages in engine draw order, drop the rest.

    Justified by measurement rather than by the original brief: four additive
    stages cost 31%, the corpus has 8,466 translucent materials, and Prey
    ships materials 200 additive stages deep. Maximum is uncapped by
    definition. Capping can empty a material outright, which is why it runs
    before the visibility audit rather than after it.
    """
    cap = profile.ambient_cap
    if not cap:
        return
    kept = []
    seen = 0
    for entry in plan.stages:
        stage = entry[0]
        if stage.lighting != 'ambient':
            kept.append(entry)
            continue
        seen += 1
        if seen <= cap:
            kept.append(entry)
        else:
            plan.drop(stage, 'ambient stage %d exceeds this mode\'s cap of %d'
                             % (seen, cap))
    plan.stages = kept


def _stage_usage(stage):
    return {'bump': USE_NORMAL, 'specular': USE_SPECULAR,
            'diffuse': USE_DIFFUSE}.get(stage.lighting, USE_DEFAULT)


def _stage_colorspace(stage):
    """sRGB only for what the engine treats as colour.

    Bump and specular maps are data. The guard in _set_image_colorspace stops
    a Non-Color wiring from claiming a datablock already known to be colour -
    idTech4 content routinely feeds one diffuse .tga to a heightmap stage too.
    """
    return 'Non-Color' if stage.lighting in ('bump', 'parallax', 'specular') \
        else 'sRGB'


def _plan_stage(plan, stage, profile, params, assets):
    uv = plan.want_uv(params.transform_chain(stage), stage.texgen,
                      stage.texgen_args)
    if stage.tex_kind == TEX_CUBE:
        # A cube map is one baked equirect image with its own coordinate
        # source; it shares nothing with the ordinary UV column.
        return
    if stage.tex_kind not in (TEX_FILE,):
        return
    sampling = Sampling.for_stage(stage)
    colorspace = _stage_colorspace(stage)
    usage = _stage_usage(stage)
    _plan_image_program(plan, stage.image, colorspace, sampling, uv.key,
                        usage, assets)


def _plan_image_program(plan, img_expr, colorspace, sampling, uv_key, usage,
                        assets):
    """Register the Image Texture nodes one image program will need.

    A program whose whole result is already in the .bimage cache is one node,
    not a tree: the engine baked it offline and cached it under the
    expression's own canonical text, so it is a finished texture. Otherwise
    every TEX_FILE leaf underneath the operators needs its own node.
    """
    if img_expr is None:
        return
    if img_expr.op is None:
        if img_expr.is_builtin() or not img_expr.path:
            # `map _flat` is the bumpmap AddImplicitStages hands every
            # interaction material that declares none. It is the engine's
            # flatNormalMap, not a file - there is nothing at base/_flat, and
            # resolving it as a path put Blender's magenta missing-image
            # placeholder on 225 Doom 3 materials.
            return
        plan.want_image(assets.resolve(img_expr.path, usage), colorspace,
                        sampling, uv_key, img_expr.path)
        return

    baked = assets.program_bimage(img_expr, usage)
    if baked is not None:
        plan.want_image(baked, colorspace, sampling, uv_key,
                        img_expr.canonical)
        return

    for arg in img_expr.args:
        _plan_image_program(plan, arg, colorspace, sampling, uv_key, usage,
                            assets)


# ===========================================================================
# END RESOURCE PLAN
# ===========================================================================


# ---------------------------------------------------------------------------
# What the ENGINE would draw, before any mode has an opinion
# ---------------------------------------------------------------------------
# These filters are not fidelity policy. They are the difference between the
# stage list the parser produced and the set of stages that actually reach the
# framebuffer, and they apply in every mode.

# The stock fragment programs in this family all read _currentRender and
# offset the lookup through a normal map, which no Blender shader can do:
# there is no way to sample what has already been rasterised behind the
# surface.
#
# This was once approximated with a Refraction BSDF driven by the stage's own
# distortion normal map. It reproduced the mechanism, but it does not look
# good - a real refracting surface reads as thick glass, while the engine's
# effect is a shimmer of at most a couple of percent of the screen
# (heatHaze.vfp clamps its offset to 0.02 * magnitude). Drawing nothing at all
# is both better looking and closer to the truth: the stage with its
# distortion removed is exactly "the background, unchanged", which is what an
# undrawn translucent stage gives you. 400 stages across the corpus.
_REFRACTION_PROGRAMS = ('heathaze', 'refraction', 'glasswarp', 'distort')


def _basename_stem(path):
    return os.path.splitext(os.path.basename((path or '').replace('\\', '/')))[0].lower()


def _refraction_program(stage):
    for value in stage.programs.values():
        name = _basename_stem(str(value))
        for known in _REFRACTION_PROGRAMS:
            if name.startswith(known):
                return known
    return None


def _best_fragment_map(mat_ir, stage):
    """The fragmentMap that is the surface's own colour, by name.

    A custom-program stage binds several fragmentMaps and only one of them is
    the colour - the rest are lookup tables, masks and normal maps. The map
    whose basename matches the material's own stem is the colour in practice
    (Prey's skin.vfp binds its diffuse at fragmentMap 4, behind an orenmap
    lookup at 0), falling back to the lowest-indexed non-builtin map.
    """
    stem = _basename_stem(mat_ir.name)
    best = None
    best_score = -1
    for index in sorted(stage.fragment_maps):
        _opts, img = stage.fragment_maps[index]
        path = img.base_path()
        if not path or img.is_builtin():
            continue
        candidate = _basename_stem(path)
        if candidate == stem:
            score = 3
        elif stem and candidate.startswith(stem):
            score = 2
        else:
            score = 1
        if score > best_score:
            best, best_score = img, score
    return best


def _substitute_stage(mat_ir, stage, plan):
    """A drawable clone of a stage we cannot reproduce, or None.

    Video and custom-program stages are understood but not representable, so
    the stage's own best still image stands in for it and the substitution is
    diagnosed. Nothing here is silent.
    """
    replacement = _best_fragment_map(mat_ir, stage)
    if replacement is None and stage.image is not None \
            and stage.image.base_path():
        replacement = stage.image
    if replacement is None and mat_ir.editor_image:
        replacement = MtrImage(path=mat_ir.editor_image,
                               canonical=mat_ir.editor_image)
    if replacement is None:
        # A drop, not just a note: this is one of the ways a stage the
        # engine draws ends up not built, so it belongs in the report
        # naming the stage it happened to, like every other one.
        plan.drop(stage, 'a %s stage has no still image to stand in for it, '
                         'so it is not drawn'
                         % (stage.tex_detail or stage.tex_kind),
                  DIAG_UNSUPPORTED, 'substituted')
        return None
    plan.add_note(DIAG_UNSUPPORTED, 'substituted',
                  'a %s stage cannot be reproduced; drawn with %s instead'
                  % (stage.tex_detail or stage.tex_kind,
                     replacement.canonical or replacement.path))
    clone = MtrStage()
    clone.__dict__.update(stage.__dict__)
    clone.image = replacement
    clone.tex_kind = TEX_FILE
    clone.programs = {}
    clone.fragment_maps = {}
    return clone


def _permanently_dark(stage, params):
    """True when this stage's colour registers are a constant zero.

    RB_STD_T_RenderShaderPasses skips such a stage on every frame - an
    additive stage adding black adds nothing, and an alpha-blended one at
    alpha zero covers nothing - so there is no reason to build it, and no
    reason to load its texture either. Quake 4's door frames carry a glow
    stage with `red 0 green 0 blue 0` behind an `if(Parm7 == 1)`, and
    building it cost a texture that could never appear.

    A register that merely happens to be zero right now is NOT this: a driver
    or a slider can put it back, so only constant expressions count.
    """
    def folded(expr, default):
        if expr is None:
            return default
        if expr.is_dynamic():
            return None
        return params.fold(expr, default)

    pair = stage.blend_pair()
    if pair == ('gl_one', 'gl_one'):
        values = [folded(c, 1.0) for c in stage.color[:3]]
        return all(v is not None and v <= 0.0 for v in values)
    if pair == ('gl_src_alpha', 'gl_one_minus_src_alpha'):
        alpha = folded(stage.color[3], 1.0)
        return alpha is not None and alpha <= 0.0
    return False


def _engine_stage(mat_ir, stage, plan, has_fallback, params):
    """The stage as it will actually be built, or None if it is not drawn."""
    if stage.lighting != 'ambient':
        return stage
    if stage.is_invisible():
        # `blend none` / gl_zero,gl_one draws nothing at all.
        plan.drop(stage, 'the engine draws nothing for it either - `blend '
                         'none` / gl_zero,gl_one leaves the destination '
                         'exactly as it was',
                  DIAG_APPROXIMATED, 'stage-not-drawn')
        return None
    if _permanently_dark(stage, params):
        plan.drop(stage, 'every colour term folds to black, so it adds '
                         'nothing to the frame at any parameter value',
                  DIAG_APPROXIMATED, 'stage-not-drawn')
        return None
    if has_fallback and any(f.startswith('shaderlevel') for f in stage.flags):
        # Prey ships both paths for its fragment-program materials: stages
        # flagged shaderFallback<N> are the plain-texture version and stages
        # flagged shaderLevel<N> are the shader version of the same surface.
        # Building both double-draws, so prefer the fallback - it is the one
        # that can be reproduced exactly.
        plan.drop(stage, 'the material ships a shaderFallback stage for the '
                         'same surface; building both would double-draw it, '
                         'and the fallback is the one reproducible exactly',
                  DIAG_APPROXIMATED, 'shaderLevel')
        return None
    if stage.has_custom_program:
        kind = _refraction_program(stage)
        if kind is not None:
            plan.drop(stage, 'screen-space %s program not drawn; the '
                             'background shows through undistorted' % kind,
                      DIAG_UNSUPPORTED, 'program ' + kind)
            return None
        return _substitute_stage(mat_ir, stage, plan)
    if stage.tex_kind == TEX_VIDEO:
        return _substitute_stage(mat_ir, stage, plan)
    if stage.tex_kind == TEX_DYNAMIC:
        plan.drop(stage, 'a %s stage renders the scene from another view; '
                         'not drawn' % (stage.tex_detail or 'render map'),
                  DIAG_UNSUPPORTED, stage.tex_detail or 'renderMap')
        return None
    return stage


def _material_has_shader_fallback(mat_ir):
    return any(any(f.startswith('shaderfallback') for f in st.flags)
               for st in mat_ir.ambient_stages)


# ===========================================================================
# BEGIN NODE GRAPH
#
# Node creation, dedup, linking and layout. Knows nothing about idTech4 beyond
# the shape of the keys the ResourcePlan hands it, and holds no mode policy at
# all - which is what lets one change to image wiring or texture matrices land
# in all four fidelity modes at once.
#
# Anything that needs to decide something (what a texture matrix folds to,
# whether a term gets a driver) lives in a primitive in the builder section
# and is plugged in here as uv_builder. The graph owns only the caching.
# ===========================================================================

# Where a node is put at the moment it is created. Roughly left to right,
# but only roughly: arrange_node_tree() rewrites every one of these once the
# tree is finished, and it is the finished wiring - not the order the builder
# happened to create things in - that decides what column a node belongs in.
#
# They are kept because a node has to be created somewhere, and because a
# caller that wires part of a graph without going through MaterialBuilder
# .build() (tests/node_dedup.py does) never reaches the layout pass and would
# otherwise get every node stacked on the origin.
COL_UV = -2400
COL_UV_STEP = 220
COL_IMAGE = -1250
ROW_STEP = -320
COL_WORK = -900
COL_OUTPUT = 1600


class NodeGraph(object):
    """One material's node tree, plus the caches that keep it from repeating."""

    def __init__(self, node_tree, plan, assets):
        self.nt = node_tree
        self.plan = plan
        self.assets = assets
        self.nodes = node_tree.nodes
        self.links = node_tree.links
        self.uv_builder = None  # set by MaterialBuilder to its TexCoords
        self._images = {}       # ImageKey    -> node
        self._uvs = {}          # UVKey       -> socket or None
        self._values = {}       # rounded float -> socket
        self._programs = {}     # program key -> (color, alpha)
        self._work_y = 0

    # -- primitives ---------------------------------------------------------

    def new(self, bl_idname, label='', x=0, y=0):
        node = self.nodes.new(bl_idname)
        if label:
            node.label = label
        node.location = (x, y)
        return node

    def work(self, bl_idname, label='', x=None, dy=-200):
        """A node in the working area, stacked down the column as it is made.

        For nodes whose position carries no meaning - the arithmetic between a
        texture and a shader. Anything structural gets an explicit x, y.
        """
        node = self.new(bl_idname, label, COL_WORK if x is None else x,
                        self._work_y)
        self._work_y += dy
        return node

    def link(self, from_socket, to_socket):
        if from_socket is None or to_socket is None:
            return None
        return self.links.new(from_socket, to_socket)

    # -- keyed resources ----------------------------------------------------

    def image(self, request):
        """The one Image Texture node for this ImageKey, created on first ask."""
        node = self._images.get(request.key)
        if node is not None:
            return node
        node = self.new('ShaderNodeTexImage', request.label, COL_IMAGE,
                        ROW_STEP * len(self._images))
        node.image = self.assets.image_at(request.path, request.colorspace)
        node.interpolation = request.sampling.interpolation
        node.extension = request.sampling.extension
        uv = self.uv(request.uv_key)
        if uv is not None:
            self.link(uv, node.inputs['Vector'])
        self._images[request.key] = node
        return node

    def uv(self, uv_key):
        """The texture-coordinate socket for a UVKey, or None for the default.

        None is not a failure: an Image Texture node with an unconnected
        Vector input already reads the active UV map, so the default chain is
        genuinely no nodes at all, which is what most stages want.
        """
        if uv_key in self._uvs:
            return self._uvs[uv_key]
        request = self.plan.uv_chains.get(uv_key)
        socket = None
        if request is not None and self.uv_builder is not None:
            socket = self.uv_builder.build(request)
        self._uvs[uv_key] = socket
        return socket

    def value(self, number, label=''):
        """A shared Value node for one constant.

        Only worth a node when something needs the number on a socket rather
        than in one; most callers write the float straight into a default_value
        and never come here.
        """
        key = round(float(number), 6)
        socket = self._values.get(key)
        if socket is None:
            node = self.new('ShaderNodeValue', label or ('%g' % key),
                            COL_WORK - 200, 200 - 120 * len(self._values))
            node.outputs[0].default_value = key
            socket = node.outputs[0]
            self._values[key] = socket
        return socket

    def group(self, tree, label=''):
        """Instance a shared node group into this tree."""
        node = self.work('ShaderNodeGroup', label or tree.name)
        node.node_tree = tree
        return node

    # -- image-program memo -------------------------------------------------
    # Two stages reading the same program with the same sampling and the same
    # UV chain get the same wired result, operators included - not just the
    # same leaf texture node.

    def program(self, key):
        return self._programs.get(key)

    def remember_program(self, key, result):
        self._programs[key] = result
        return result

    # -- housekeeping -------------------------------------------------------

    def prune_orphans(self):
        """Remove nodes that reach no output. A sanity check, not a pass.

        Nodes are only created when the plan says they are needed, so this
        should find nothing; it stays because "should" is not "does", and
        because a stage that turns out to contribute nothing can still leave
        its sampler behind.

        It reads node_tree.links ONCE. Do not reach for NodeSocket.links here:
        Blender builds that list by walking the whole tree per socket, which is
        quadratic, and it is the accessor behind the intermittent 'NodeLink
        object has no attribute to_socket' failures on large materials.
        """
        producers = {}
        for link in self.nt.links:
            producers.setdefault(link.to_node, []).append(link.from_node)

        keep = set()
        frontier = [n for n in self.nodes if n.type == 'OUTPUT_MATERIAL']
        while frontier:
            node = frontier.pop()
            if node in keep:
                continue
            keep.add(node)
            frontier.extend(producers.get(node, ()))

        removed = 0
        for node in list(self.nodes):
            if node not in keep:
                self.nodes.remove(node)
                removed += 1
        return removed

    def node_count(self):
        return len(self.nodes)


# ===========================================================================
# END NODE GRAPH
# ===========================================================================


# ===========================================================================
# BEGIN NODE LAYOUT
#
# One pass over a finished tree that puts it into readable left-to-right
# order: sources on the left, the Material Output on the right, nothing
# overlapping anything, and as few crossed links as a cheap heuristic can
# manage.
#
# It runs after the graph is built rather than during it, because which
# column a node belongs in is a property of the finished wiring - a Value
# node feeding the last mix is not a "source" even though it has no inputs,
# and the builder cannot know that at the moment it creates the node.
#
# This is layered (Sugiyama-style) drawing without the dummy-node machinery:
#   1. layer  - longest path from a source, so every link points right
#   2. order  - median-heuristic sweeps, to uncross what can be uncrossed
#   3. place  - barycentre pull under a hard non-overlap constraint
# Nothing here knows anything about idTech4.
# ===========================================================================

LAYOUT_COL_GAP = 70.0     # horizontal gap between one column and the next
LAYOUT_ROW_GAP = 34.0     # vertical gap between two nodes in a column
LAYOUT_SWEEPS = 6         # ordering passes; more buys very little past ~4
LAYOUT_PLACE_PASSES = 4   # barycentre passes over the columns

# How tall a node draws.
#
# node.dimensions is (0, 0) until the node editor has actually drawn the
# tree, and materials are built headless - by the importer under a modal
# operator, and by every test in tests/ under --background - so the height
# has to be computed rather than read.
#
# Blender draws a node as a base (header, dropdowns, the image or object
# selector, a checkbox) plus one row per visible socket. Rows are 21 units.
# An unconnected Vector or Rotation input that shows its value draws a
# labelled XYZ widget instead of a row, which is four rows; connected, or
# marked hide_value the way a Normal or a texture-coordinate input is, it is
# back to one. That model reproduces every drawn height measured across the
# five game corpora to within a couple of units - Vector Math alone was seen
# at 2, 3, 6 and 10 rows and lands on 60 + 21 * rows every time.
#
# The bases below were measured that way by tests/node_layout_gui.py, which
# opens a real window, draws each material and fails if any estimate comes
# in under what Blender drew. Re-run it after a Blender upgrade. Each carries
# a few units of headroom: too tall costs whitespace, too short costs the one
# thing this pass exists to guarantee.
_LAYOUT_ROW = 21.0        # one socket row
_LAYOUT_WIDE_ROWS = 4     # an unconnected Vector/Rotation value widget
_LAYOUT_BASE = 90.0       # header and widgets, for a node not listed below

_NODE_BASES = {
    'ShaderNodeAddShader':        40.0,
    'ShaderNodeAttribute':        92.0,
    'ShaderNodeBsdfDiffuse':      42.0,
    'ShaderNodeBsdfTransparent':  40.0,
    'ShaderNodeBump':             66.0,
    'ShaderNodeCombineColor':     64.0,
    'ShaderNodeCombineXYZ':       40.0,
    'ShaderNodeEmission':         42.0,
    'ShaderNodeGroup':            60.0,
    'ShaderNodeHoldout':          34.0,
    'ShaderNodeInvert':           40.0,
    'ShaderNodeMapping':          62.0,
    'ShaderNodeMath':             90.0,
    'ShaderNodeMix':             142.0,
    'ShaderNodeMixShader':        40.0,
    'ShaderNodeNewGeometry':      42.0,
    'ShaderNodeNormalMap':       138.0,
    'ShaderNodeOutputMaterial':   62.0,
    'ShaderNodeRGB':             136.0,
    'ShaderNodeSeparateColor':    64.0,
    'ShaderNodeSeparateXYZ':      40.0,
    'ShaderNodeTexChecker':       42.0,
    'ShaderNodeTexCoord':         94.0,
    'ShaderNodeTexEnvironment':  188.0,
    'ShaderNodeTexImage':        214.0,
    'ShaderNodeUVMap':            90.0,
    'ShaderNodeValue':            34.0,
    'ShaderNodeVectorMath':       66.0,
    'ShaderNodeVertexColor':      64.0,
    'ShaderNodeVolumeAbsorption': 66.0,
    'ShaderNodeVolumeScatter':    66.0,
    'NodeGroupInput':             60.0,
    'NodeGroupOutput':            60.0,
}

# Nodes whose sockets live in collapsed panels, so the row model does not
# describe them: Principled draws 31 enabled sockets in 362 units because
# most of them are inside closed panels. A flat number, with headroom for a
# panel someone opens by hand.
_NODE_FIXED_HEIGHTS = {
    'ShaderNodeBsdfPrincipled': 380.0,
    'NodeReroute': 20.0,
}

_LAYOUT_WIDE_SOCKETS = frozenset(('VECTOR', 'ROTATION'))

# Sourceless nodes that stay in the left source column even when the thing
# they feed sits far to the right. These are where texture data enters the
# tree, and reading the graph depends on finding them together.
#
# Every OTHER sourceless node - a Value, an RGB, a Transparent BSDF standing
# in for "hidden" - is pulled right to sit beside the node it feeds, which is
# where it is legible and where it stops dragging a link across the whole
# width of the tree.
_LAYOUT_SOURCE_NODES = frozenset((
    'ShaderNodeTexImage', 'ShaderNodeTexEnvironment', 'ShaderNodeTexChecker',
    'ShaderNodeTexCoord', 'ShaderNodeUVMap', 'ShaderNodeAttribute',
    'ShaderNodeVertexColor', 'ShaderNodeNewGeometry', 'NodeGroupInput',
))

_LAYOUT_SINK_NODES = frozenset((
    'ShaderNodeOutputMaterial', 'NodeGroupOutput',
))


def node_display_height(node):
    """How tall `node` draws, in node-editor units. See _NODE_BASES."""
    fixed = _NODE_FIXED_HEIGHTS.get(node.bl_idname)
    if fixed is not None:
        return fixed
    if node.hide:
        return 34.0
    rows = 0
    for socket in node.outputs:
        if not socket.hide and socket.enabled:
            rows += 1
    for socket in node.inputs:
        if socket.hide or not socket.enabled:
            continue
        if socket.is_linked or socket.hide_value or \
                socket.type not in _LAYOUT_WIDE_SOCKETS:
            rows += 1
        else:
            rows += _LAYOUT_WIDE_ROWS
    return _NODE_BASES.get(node.bl_idname, _LAYOUT_BASE) + _LAYOUT_ROW * rows


def _layout_edges(nt, nodes):
    """(preds, succs) over `nodes`, reading node_tree.links exactly once.

    Not NodeSocket.links: Blender rebuilds that by walking every link in the
    tree per socket, which is what makes it quadratic on big materials.
    """
    preds = dict((n, []) for n in nodes)
    succs = dict((n, []) for n in nodes)
    for link in nt.links:
        a, b = link.from_node, link.to_node
        if a is b or a not in preds or b not in preds:
            continue
        if b not in succs[a]:
            succs[a].append(b)
        if a not in preds[b]:
            preds[b].append(a)
    return preds, succs


def _layout_columns(nodes, preds, succs):
    """Column index per node: the longest path from any sourceless node.

    Longest-path layering is what makes every link point strictly right - a
    node always lands at least one column past everything feeding it, so
    there is no such thing as a backwards or vertical link in the result.
    """
    column = {}
    stack = set()

    def depth(node):
        got = column.get(node)
        if got is not None:
            return got
        if node in stack:
            # Shader trees are acyclic and Blender refuses to link a cycle,
            # but a tree that arrived here some other way must not recurse
            # forever.
            return 0
        stack.add(node)
        parents = preds[node]
        got = 1 + max([depth(p) for p in parents]) if parents else 0
        stack.discard(node)
        column[node] = got
        return got

    for node in nodes:
        depth(node)

    last = max(column.values()) if column else 0

    # The Material Output's position is a promise: it is the right-hand end
    # of the tree whether or not it happens to sit on the longest path.
    for node in nodes:
        if node.bl_idname in _LAYOUT_SINK_NODES:
            column[node] = last

    # Constants and stand-ins ride right to meet whatever consumes them.
    for node in nodes:
        if preds[node] or node.bl_idname in _LAYOUT_SOURCE_NODES:
            continue
        if succs[node]:
            column[node] = max(0, min(column[s] for s in succs[node]) - 1)
    return column


def _count_crossings(order, preds):
    """Links crossed between every adjacent pair of columns, as ordered."""
    total = 0
    for i in range(1, len(order)):
        rank = dict((n, j) for j, n in enumerate(order[i - 1]))
        pairs = []
        for j, node in enumerate(order[i]):
            for p in preds[node]:
                if p in rank:
                    pairs.append((j, rank[p]))
        for a in range(len(pairs)):
            ja, ra = pairs[a]
            for b in range(a + 1, len(pairs)):
                jb, rb = pairs[b]
                if (ja - jb) * (ra - rb) < 0:
                    total += 1
    return total


def _order_columns(columns, preds, succs):
    """Order each column top to bottom, uncrossing what a sweep can.

    The median heuristic: put each node beside the median position of what it
    connects to in the column just laid out, sweep in both directions a few
    times, and keep whichever pass crossed the fewest links. It is not
    optimal - minimising crossings is NP-hard - but it is cheap and it is
    what takes a stack of shader maths from unreadable to obvious.
    """
    def pass_over(order, forward):
        span = (range(1, len(order)) if forward
                else range(len(order) - 2, -1, -1))
        neighbours = preds if forward else succs
        for i in span:
            rank = dict((n, j) for j, n
                        in enumerate(order[i - 1 if forward else i + 1]))
            keyed = []
            for j, node in enumerate(order[i]):
                positions = sorted(rank[n] for n in neighbours[node]
                                   if n in rank)
                if positions:
                    mid = len(positions) // 2
                    key = (float(positions[mid]) if len(positions) % 2
                           else 0.5 * (positions[mid - 1] + positions[mid]))
                else:
                    # Nothing to be beside; hold the position it has, so a
                    # sweep never shuffles unconstrained nodes for free.
                    key = float(j)
                keyed.append((key, j, node))
            keyed.sort(key=lambda t: (t[0], t[1]))
            order[i] = [t[2] for t in keyed]
        return order

    order = [list(col) for col in columns]
    best = [list(col) for col in order]
    best_score = _count_crossings(order, preds)
    for step in range(LAYOUT_SWEEPS):
        if not best_score:
            break
        order = pass_over(order, forward=(step % 2 == 0))
        score = _count_crossings(order, preds)
        if score < best_score:
            best_score = score
            best = [list(col) for col in order]
    return best


def _stack(heights, wanted):
    """Place one ordered column as near `wanted` as its order allows.

    Blender's node location is the TOP-LEFT corner and y decreases downward,
    so node i occupies [top - height, top] and "i sits above i+1" is

        top[i + 1] <= top[i] - height[i] - LAYOUT_ROW_GAP

    Substituting z[i] = top[i] + sum of (height + gap) over everything above
    it turns that into "z is non-increasing", and the nearest non-increasing
    sequence to a given one is exactly antitonic regression - solved here by
    pooling adjacent violators into blocks that take their mean.

    Two clamping sweeps were tried first and are subtly wrong: the sweep that
    compacts a column back upwards can push a node into the one above it,
    which is how a Normal Map ended up drawn over the roughness group. This
    is O(n), needs no iteration to converge, and is optimal rather than
    merely legal.
    """
    if not wanted:
        return []
    offset = 0.0
    offsets = []
    for h in heights:
        offsets.append(offset)
        offset += h + LAYOUT_ROW_GAP

    blocks = []                       # [total, count], means strictly falling
    for i, want in enumerate(wanted):
        blocks.append([want + offsets[i], 1])
        while len(blocks) > 1 and \
                blocks[-2][0] * blocks[-1][1] < blocks[-1][0] * blocks[-2][1]:
            total, count = blocks.pop()
            blocks[-1][0] += total
            blocks[-1][1] += count

    tops = []
    for total, count in blocks:
        mean = total / float(count)
        for _ in range(count):
            tops.append(mean - offsets[len(tops)])
    return tops


def arrange_node_tree(nt):
    """Lay `nt` out left to right. Returns the number of nodes placed.

    Safe on any shader tree, built by this addon or not.
    """
    nodes = [n for n in nt.nodes if n.bl_idname != 'NodeFrame']
    frames = [n for n in nt.nodes if n.bl_idname == 'NodeFrame']
    if not nodes:
        return 0

    preds, succs = _layout_edges(nt, nodes)
    column = _layout_columns(nodes, preds, succs)

    columns = [[] for _ in range(max(column.values()) + 1)]
    for node in nodes:                    # creation order seeds the ordering
        columns[column[node]].append(node)
    columns = _order_columns(columns, preds, succs)

    heights = dict((n, node_display_height(n)) for n in nodes)

    # X: every column is as wide as its widest node, so nothing reaches into
    # the next one however wide an Image Texture turns out to be.
    xs = []
    x = 0.0
    for col in columns:
        xs.append(x)
        x += max([n.width for n in col] or [140.0]) + LAYOUT_COL_GAP

    # Y: stack once to get somewhere legal, then let the barycentres pull
    # each node towards what it is wired to, re-clamping every time.
    tops = {}
    for col in columns:
        y = 0.0
        for node in col:
            tops[node] = y
            y -= heights[node] + LAYOUT_ROW_GAP

    for step in range(LAYOUT_PLACE_PASSES):
        forward = (step % 2 == 0)
        span = (range(len(columns)) if forward
                else range(len(columns) - 1, -1, -1))
        neighbours = preds if forward else succs
        for i in span:
            col = columns[i]
            wanted = []
            for node in col:
                linked = neighbours[node]
                if linked:
                    mean = sum(tops[n] - heights[n] * 0.5
                               for n in linked) / float(len(linked))
                    wanted.append(mean + heights[node] * 0.5)
                else:
                    wanted.append(tops[node])
            for node, top in zip(col, _stack([heights[n] for n in col],
                                             wanted)):
                tops[node] = top

    # Normalise: left edge at x = 0, and the tree centred on y = 0 so that
    # opening the material lands the view on the middle of it.
    highest = max(tops.values())
    lowest = min(tops[n] - heights[n] for n in nodes)
    shift = -(highest + lowest) * 0.5
    for i, col in enumerate(columns):
        for node in col:
            node.location = (xs[i], tops[node] + shift)

    # Frames carry a warning label and own no children; park them in a stack
    # above the graph, where they are read once and never in the way.
    #
    # Nothing reaches this today: the two frames the builder makes - the
    # guiSurf banner and the "this material failed to parse" one - are
    # deleted by prune_orphans() before the layout runs, because a frame has
    # no links and so never reaches the Material Output. That is a bug in
    # prune_orphans, not here; this stays correct for when it is fixed, and
    # for any tree that arrives with frames already in it.
    top = highest + shift + 130.0
    for frame in frames:
        if frame.parent is not None:
            continue
        frame.location = (0.0, top)
        top += 90.0
    return len(nodes)


# ===========================================================================
# END NODE LAYOUT
# ===========================================================================


# ===========================================================================
# BEGIN SHADER LIBRARY
#
# Node groups and generated images shared by every material in the scene, so
# that editing one reaches all of them and so that 6,000 materials do not
# carry 6,000 copies of the same four maths nodes.
# ===========================================================================


def _strip_roughness_band(grp):
    """Remove a Map Range left behind by the roughness-band experiment.

    A .blend generated while that toggle existed carries a `range` node inside
    the shared group, and if it was left on the compressed setting the whole
    scene would keep rendering that way even though the code no longer offers
    it. Rewire straight through and delete the node - editing inside the group
    reaches every material that references it, and the datablock itself is
    kept so their node_tree links stay intact.
    """
    rng = grp.nodes.get('range')
    if rng is None:
        return grp
    src_socket = (rng.inputs['Value'].links[0].from_socket
                  if rng.inputs['Value'].is_linked else None)
    targets = [link.to_socket for link in rng.outputs['Result'].links]
    grp.nodes.remove(rng)
    if src_socket is not None:
        for socket in targets:
            grp.links.new(src_socket, socket)
    grp.pop('idtech4_roughness_version', None)
    grp.pop('idtech4_roughness_mode', None)
    return grp


def get_or_create_estimate_roughness_group():
    """
    Return (creating if needed) a node group that converts a specular map
    colour to a Blender Roughness value.
    Luminance → power(0.5) → 1 - result  (bright spec → low roughness).

    Left deliberately as-is. The output does sit high - across 140 of the 1783
    specular maps Doom 3 ships, the median is 0.634 with 94% above 0.40 - and
    a compressed 0.05..0.30 band measures closer to the source texture's
    saturation on paper. It was built as a toggle and compared side by side,
    and the tight band did not look like the game. Numbers lost to eyes; do
    not "fix" this again without looking at it in a viewport first.
    """
    GROUP_NAME = ROUGHNESS_GROUP_NAME
    if GROUP_NAME in bpy.data.node_groups:
        return _strip_roughness_band(bpy.data.node_groups[GROUP_NAME])

    grp = bpy.data.node_groups.new(GROUP_NAME, 'ShaderNodeTree')

    # Inputs / Outputs
    grp.interface.new_socket('Specular Color', in_out='INPUT',  socket_type='NodeSocketColor')
    grp.interface.new_socket('Roughness',      in_out='OUTPUT', socket_type='NodeSocketFloat')

    nodes = grp.nodes
    links = grp.links

    gi = nodes.new('NodeGroupInput');  gi.location = (-400, 0)
    go = nodes.new('NodeGroupOutput'); go.location  = ( 400, 0)

    # Luminance via dot product with BT.601 weights
    lum = nodes.new('ShaderNodeVectorMath')
    lum.operation = 'DOT_PRODUCT'
    lum.inputs[1].default_value = (0.299, 0.587, 0.114)
    lum.location = (-200, 0)

    # Power 0.5 ≈ perceptual mid-point
    pw = nodes.new('ShaderNodeMath')
    pw.operation = 'POWER'
    pw.inputs[1].default_value = 0.5
    pw.location = (0, 0)

    # Invert: bright spec → low roughness
    inv = nodes.new('ShaderNodeMath')
    inv.operation = 'SUBTRACT'
    inv.use_clamp = True
    inv.inputs[0].default_value = 1.0
    inv.location = (200, 0)

    links.new(gi.outputs['Specular Color'], lum.inputs[0])
    links.new(lum.outputs['Value'],         pw.inputs[0])
    links.new(pw.outputs['Value'],          inv.inputs[1])
    links.new(inv.outputs['Value'],         go.inputs['Roughness'])

    return grp


_PLACEHOLDER_IMAGE_NAME = 'idtech4_placeholder'
# Flat dark grey at alpha 0.22. Deliberately not 0.20: the swatch has to
# read as scaffolding through a wall of them without disappearing.
_PLACEHOLDER_RGBA = (0.045, 0.045, 0.05, 0.22)


def get_or_create_placeholder_image():
    """A flat dark-grey, mostly-transparent swatch, generated once and shared.

    Deliberately featureless. It marks "the engine draws something here that
    this importer will not pretend to reproduce" - a GUI, a light projection,
    a mirror - and reads as scaffolding rather than as a material that came
    out wrong.
    """
    for img in bpy.data.images:
        if img.get('idtech4_placeholder_image'):
            return img
    size = 8
    img = bpy.data.images.new(_PLACEHOLDER_IMAGE_NAME, size, size, alpha=True,
                              float_buffer=False)
    img.pixels.foreach_set(list(_PLACEHOLDER_RGBA) * (size * size))
    _pack_written_pixels(img)
    img['idtech4_placeholder_image'] = True
    return img


# ===========================================================================
# END SHADER LIBRARY
# ===========================================================================


# ===========================================================================
# BEGIN BUILD PROFILES
#
# The four fidelity modes, expressed as data rather than as `if` tests spread
# through the builder. A profile says what a mode wants; the builder and the
# surface strategies read it and never ask which mode they are in.
#
# The modes exist to make a scene renderable on a card that is not an RTX
# 4090. Every field below is justified by a measurement, and the acceptance
# criterion for the ladder is that each rung renders measurably faster than
# the one above it - see tests/tier_bench.py.
#
# TWO sets of numbers, and they say different things.
#
# The SYNTHETIC ones (tier_bench_run.ps1: hand-built graphs, one feature
# varied at a time, 300 materials on 300 cubes, 1280x720, 64 samples, GL cache
# wiped between variants) isolate what each graph feature costs: three
# interaction passes against one is 1.77x, Principled against Diffuse a
# further ~20%, four additive ambient stages +31%.
#
# The REAL ones (tier_bench_real.ps1: the same scene, every graph built by
# this addon from 300 Doom 3 materials that have their textures on disk) are
# what a user actually gets, and they are much smaller:
#
#     real_maximum  0.4672s  1.00x   9.9 nodes/mat  819 tex  338.5 MB
#     real_good     0.4088s  1.14x   9.7           819      338.5
#     real_basic    0.3592s  1.30x   7.4           584      247.1
#     real_simple   0.3115s  1.50x   4.5           281      114.9
#
#     noise floor, from an identical-graph control:  0.7%
#
# Note Good and Maximum are identical on resident texture. Good's rung is
# entirely shape - one interaction pass instead of N, Diffuse instead of
# Principled, no specular highlight - and none of that unloads a texture.
# The VRAM ladder starts at Basic, which stops loading specular maps
# (-91MB here), and Simple, which stops loading normals and heights
# (-132MB more).
#
# The gap is not a contradiction, it is the corpus: almost every real material
# flushes ONE drawInteraction_t, so Maximum's pass-collapsing lever - the
# biggest one in the synthetic run - barely fires. What the real run shows
# instead is that the ladder's payoff is mostly VRAM: 338 MB of resident
# texture down to 113 MB, a 67% cut, on exactly the low-memory cards these
# modes exist to serve. est_speedup below is the REAL figure, because that is
# the one the UI shows a user.
#
# One thing this table was originally costed wrongly. Good used to drop cube
# maps, justified as "saves a ~7.7MB equirect bake each". The bake is cached
# per distinct cube map and shared by every material referencing it (see
# get_or_create_cubemap_equirect's idtech4_cubemap_tag), so it is not "each"
# at all: Doom 3 references 40 distinct cube maps across 105 materials, about
# 20 of which have their faces on disk. The whole saving was a one-time ~150MB
# for an entire game base - and the real bench measured Good against Maximum
# at 336.2MB against 338.5MB of resident texture, so it did not show up at
# all. The cost was 36 materials going blank, every sky among them, in the
# DEFAULT mode. Every mode builds cube maps now.
# ===========================================================================

MODE_MAXIMUM = 'MAXIMUM'
MODE_GOOD = 'GOOD'
MODE_BASIC = 'BASIC'
MODE_SIMPLE = 'SIMPLE'


class BuildProfile(object):
    """What one fidelity mode wants built. Immutable, one instance per mode."""

    __slots__ = ('name', 'label', 'description', 'surface', 'interactions',
                 'lit_shader', 'specular', 'normals', 'heightmaps', 'cubemaps',
                 'ambient', 'ambient_cap', 'vertex_color',
                 'prefer_editor_image', 'est_speedup')

    def __init__(self, name, label, description, surface, interactions,
                 lit_shader, specular, normals, heightmaps, cubemaps,
                 ambient, ambient_cap, vertex_color, prefer_editor_image,
                 est_speedup):
        self.name = name
        self.label = label
        self.description = description
        self.surface = surface
        self.interactions = interactions
        self.lit_shader = lit_shader
        self.specular = specular
        self.normals = normals
        self.heightmaps = heightmaps
        self.cubemaps = cubemaps
        self.ambient = ambient
        self.ambient_cap = ambient_cap
        self.vertex_color = vertex_color
        self.prefer_editor_image = prefer_editor_image
        self.est_speedup = est_speedup

    def copy(self, **overrides):
        fields = {slot: getattr(self, slot) for slot in self.__slots__}
        fields.update(overrides)
        return BuildProfile(**fields)

    # Does this mode load a specular texture at all? Basic does not, and that
    # is a per-material VRAM saving that never shows up in a render time.
    @property
    def wants_specular_texture(self):
        return self.specular != 'none'

    def __repr__(self):
        return '<BuildProfile %s>' % self.name


PROFILES = {}


def _profile(**kwargs):
    p = BuildProfile(**kwargs)
    PROFILES[p.name] = p
    return p


MAXIMUM = _profile(
    name=MODE_MAXIMUM,
    label='Maximum',
    description=(
        'The ARB2 renderer\'s own draw sequence: one Principled BSDF per '
        'flushed drawInteraction_t, engine-exact specular, cube maps, and '
        'every ambient stage composited through its real OpenGL blend '
        'equation. The reference rung, and the most expensive.'),
    surface='EnginePassSurface',
    interactions='engine_passes',
    lit_shader='principled',
    specular='engine',
    normals=True,
    heightmaps=True,
    cubemaps=True,
    ambient='engine_composite',
    ambient_cap=None,
    vertex_color=True,
    prefer_editor_image=False,
    est_speedup=1.0,
)

GOOD = _profile(
    name=MODE_GOOD,
    label='Good',
    description=(
        'Measured 1.14x faster than Maximum on real Doom 3 materials. '
        'Collapses the engine\'s interaction passes into one and uses a '
        'Diffuse BSDF; the pass collapse is worth 1.77x on a material that '
        'flushes three of them, but almost every real material flushes one. '
        'Keeps normals, heightmaps, cube maps, the roughness estimate and '
        'full ambient compositing. What it gives up is the specular '
        'highlight.'),
    surface='MergedPassSurface',
    interactions='merged',
    lit_shader='diffuse',
    specular='roughness_only',
    normals=True,
    heightmaps=True,
    cubemaps=True,
    ambient='engine_composite',
    ambient_cap=8,
    vertex_color=True,
    prefer_editor_image=False,
    est_speedup=1.14,
)

BASIC = _profile(
    name=MODE_BASIC,
    label='Basic',
    description=(
        'Measured 1.30x faster than Maximum, and 91MB lighter than Good '
        'across 300 materials: it loads no specular texture at all, and '
        '16,593 corpus materials ship one. Builds only the ambient stages '
        'that carry alpha - decals, overlays and cutouts - and caps the '
        'stack; four additive stages measured +31%, and Prey ships materials '
        '200 stages deep.'),
    surface='FlatLitSurface',
    interactions='merged',
    lit_shader='diffuse',
    specular='none',
    normals=True,
    heightmaps=True,
    cubemaps=True,
    ambient='alpha_only',
    ambient_cap=4,
    vertex_color=False,
    prefer_editor_image=False,
    est_speedup=1.30,
)

SIMPLE = _profile(
    name=MODE_SIMPLE,
    label='Simple',
    description=(
        'Measured 1.50x faster than Maximum and a third of its resident '
        'texture memory (115MB against 338MB over 300 materials): the diffuse '
        'texture on a lit Diffuse BSDF, nothing else. Lit rather than '
        'emissive on purpose - it measured the same and geometry placement '
        'reads correctly against scene lighting, which is the point of this '
        'mode.'),
    surface='UnlitDiffuseSurface',
    interactions='single',
    lit_shader='diffuse',
    specular='none',
    normals=False,
    heightmaps=False,
    cubemaps=True,
    ambient='alpha_only',
    ambient_cap=2,
    vertex_color=False,
    prefer_editor_image=False,
    est_speedup=1.50,
)

MODE_ORDER = (MODE_MAXIMUM, MODE_GOOD, MODE_BASIC, MODE_SIMPLE)

def get_profile(mode, ambient_cap=None, prefer_editor_image=None):
    """The profile for a mode, with the panel's overrides applied.

    ambient_cap of 0 from the UI means uncapped; None means "leave the mode's
    own value alone".
    """
    base = PROFILES.get(mode) or PROFILES[MODE_GOOD]
    overrides = {}
    if ambient_cap is not None:
        overrides['ambient_cap'] = None if ambient_cap <= 0 else ambient_cap
    if prefer_editor_image is not None:
        overrides['prefer_editor_image'] = bool(prefer_editor_image)
    return base.copy(**overrides) if overrides else base


# ===========================================================================
# END BUILD PROFILES
# ===========================================================================


# ===========================================================================
# BEGIN BUILDER
#
# One MaterialBuilder, holding the primitives that do not vary by mode. It
# contains NO mode policy: everything it does is either mode-independent or is
# read off the BuildProfile it was handed. The only polymorphic piece in the
# whole file is SurfaceStrategy, in the next section.
#
# The primitives:
#   TexCoords          texgen + the texture-matrix chain -> one UV socket
#   ImageWiring        one MtrImage AST  -> (color socket, alpha socket)
#   StageSampler       one MtrStage      -> (color, alpha), policy applied
#   AmbientCompositor  the OpenGL blend equations, over a background
#   InteractionModel   engine pass splitting into drawInteraction_t sets
# ===========================================================================


def _bind_frame_dependency(fcurve):
    """Give a driver an explicit dependency on the scene frame.

    Blender exposes `frame` inside driver expressions, but a driver that only
    reads that builtin is not registered as depending on anything, so it is
    never re-evaluated on frame change. Binding a variable to
    scene.frame_current is what actually makes time-varying material drivers
    animate in the viewport.
    """
    driver = fcurve.driver
    driver.use_self = False
    if any(v.name == 'frame' for v in driver.variables):
        return
    var = driver.variables.new()
    var.name = 'frame'
    var.type = 'SINGLE_PROP'
    target = var.targets[0]
    target.id_type = 'SCENE'
    try:
        target.id = bpy.context.scene
    except (AttributeError, TypeError):
        pass
    target.data_path = 'frame_current'


# ---------------------------------------------------------------------------
# The surfaces that are not surfaces
# ---------------------------------------------------------------------------


class SpecialSurfaces(object):
    """Materials the fidelity ladder has no opinion about.

    Light shaders, broken declarations and editor-image-only decls are not
    surface shaders at all, so a strategy would be answering the wrong
    question about them. Each of these is drawn the same way in every mode.
    """

    def special_surface(self):
        """The finished surface socket, or None to let the strategy run."""
        ir = self.ir
        if ir.failed:
            self.stood_in = 'broken declaration'
            return self.broken_surface()
        if ir.is_light:
            self.stood_in = 'light shader'
            return self.light_surface()
        if ir.editor_image and not ir.stages:
            # Clip brushes, triggers, origins and the rest of textures/common:
            # the engine draws nothing at all, and the editor image is the
            # only thing there is to show.
            self.stood_in = 'editor image only'
            return self.editor_surface(listed=True)
        if ir.deform and ir.deform[0] == 'flare':
            # The surface the stages describe is not the surface the engine
            # draws - the deform generates its own geometry and writes the
            # vertex colour every stage is then multiplied by.
            self.note(DIAG_UNSUPPORTED, 'deform flare',
                      'deform flare builds its own geometry in the engine; '
                      'the editor image stands in for it')
            self.stood_in = 'deform flare'
            return self.editor_surface(listed=True)
        return None

    def light_surface(self):
        """A flat unlit preview of a light material's projection texture.

        Light shaders are not surface shaders - building one as a lit surface
        produces a meaningless object. But 1,267 of the corpus's 1,271 light
        materials do have a stage map, and that map IS what the light
        projects, so showing it unlit tells you at a glance which light shader
        you are looking at. Only 15 of them have a qer_editorimage, which is
        why that is a toggle rather than the default.
        """
        ir = self.ir
        prefer_editor = bool(getattr(self.settings, 'light_prefer_editor',
                                     False)) if self.settings else False
        source = None
        source_stage = None
        if prefer_editor and ir.editor_image:
            source = MtrImage(path=ir.editor_image, canonical=ir.editor_image)
        if source is None:
            for stage in ir.stages:
                if stage.image is not None and stage.image.base_path():
                    source, source_stage = stage.image, stage
                    break
        if source is None and ir.editor_image:
            source = MtrImage(path=ir.editor_image, canonical=ir.editor_image)

        uv_key = ()
        sampling = DEFAULT_SAMPLING
        if source_stage is not None:
            sampling = Sampling.for_stage(source_stage)
            uv_key = self.plan.want_uv(
                self.params.transform_chain(source_stage),
                source_stage.texgen, source_stage.texgen_args).key
        color, alpha = self.images.color(source, None, 'sRGB', sampling,
                                         uv_key)
        emit = self.work('ShaderNodeEmission', 'light projection')
        if color is not None:
            self.link(color, emit.inputs['Color'])
        else:
            emit.inputs['Color'].default_value = (0.5, 0.5, 0.5, 1.0)
        transparent = self.work('ShaderNodeBsdfTransparent', '')
        mix = self.work('ShaderNodeMixShader', 'light preview')
        if alpha is not None:
            self.link(alpha, mix.inputs['Fac'])
        else:
            mix.inputs['Fac'].default_value = 1.0
        self.link(transparent.outputs['BSDF'], mix.inputs[1])
        self.link(emit.outputs['Emission'], mix.inputs[2])
        self.note(DIAG_UNSUPPORTED, 'light material',
                  'light shader (%s) - shown as an unlit preview of its '
                  'projection texture' % ir.kind)
        return mix.outputs['Shader']

    def editor_surface(self, listed=False):
        """The editor-image-only path.

        Keeps the IDTECH4_EditorMix node the Editor Textures panel's opacity
        slider looks for. `listed` flags the material for that panel: it is
        the whole surface here, not a fallback that happens to use the editor
        image, and only the former should turn up in a list whose purpose is
        hiding clip volumes wholesale.
        """
        ir = self.ir
        path = ir.editor_image or ir.first_image_path()
        diffuse = self.work('ShaderNodeBsdfDiffuse', 'editor image')
        resolved = self.assets.resolve(path) if path else ''
        if resolved and os.path.isfile(resolved):
            tex = self.work('ShaderNodeTexImage', 'qer_editorimage')
            tex.image = self.assets.image_at(resolved, 'sRGB')
            self.link(tex.outputs['Color'], diffuse.inputs['Color'])
        elif 'glass' in ir.flags or 'glass_macro' in ir.flags:
            # A glass material with no image on disk would otherwise get
            # Blender's magenta "missing texture" checker, which reads as
            # broken when the material would look like perfectly plausible
            # pale glass.
            tex = self.work('ShaderNodeTexImage', 'glass placeholder')
            tex.image = self.assets.glass_placeholder(
                ir.name.replace('/', '_'))
            self.link(tex.outputs['Color'], diffuse.inputs['Color'])
            _detail = ('no image found on disk - using a procedural '
                       'glass stand-in')
            if path:
                # A missing ASSET, so the report names where it was looked
                # for - not just that it wasn't there.
                _detail += ('; searched:\n        '
                            + '\n        '.join(self.assets.search_paths(path)))
            self.note(DIAG_UNSUPPORTED, 'missing image', _detail)
        elif path:
            tex = self.work('ShaderNodeTexImage', 'qer_editorimage')
            tex.image = self.assets.image_at(resolved, 'sRGB')
            self.link(tex.outputs['Color'], diffuse.inputs['Color'])
        else:
            diffuse.inputs['Color'].default_value = (0.15, 0.15, 0.15, 1.0)
            self.note(DIAG_UNSUPPORTED, 'no image',
                      'no qer_editorimage and no stage texture - using a '
                      'flat grey')
        transparent = self.work('ShaderNodeBsdfTransparent', '')
        mix = self.work('ShaderNodeMixShader', 'Editor Texture Opacity')
        mix.name = 'IDTECH4_EditorMix'
        mix.inputs['Fac'].default_value = 1.0
        self.link(transparent.outputs['BSDF'], mix.inputs[1])
        self.link(diffuse.outputs['BSDF'], mix.inputs[2])
        if listed:
            self.mat['idtech4_editor_texture'] = True
            if self.mat.get('idtech4_visible') is None:
                self.mat['idtech4_visible'] = True
        return mix.outputs['Shader']

    def broken_surface(self):
        """The .mtr definition is broken and the engine would MakeDefault().

        Shown as something unmistakably wrong rather than a plausible grey,
        because a plausible grey is how a broken material stays broken for
        months.
        """
        checker = self.work('ShaderNodeTexChecker', 'PARSE FAILED')
        checker.inputs['Color1'].default_value = (1.0, 0.0, 1.0, 1.0)
        checker.inputs['Color2'].default_value = (0.0, 0.0, 0.0, 1.0)
        checker.inputs['Scale'].default_value = 16.0
        emit = self.work('ShaderNodeEmission', 'PARSE FAILED')
        self.link(checker.outputs['Color'], emit.inputs['Color'])
        frame = self.nt.nodes.new('NodeFrame')
        frame.label = 'idTech4: this material failed to parse - see the report'
        frame.location = (0, 260)
        self.note(DIAG_PARSE_ERROR, 'broken material',
                  'the declaration is broken; the engine would MakeDefault() '
                  'it, and this draws a magenta checker so it cannot be '
                  'mistaken for a working material')
        return emit.outputs['Emission']


# ---------------------------------------------------------------------------
# Whole-material assembly
# ---------------------------------------------------------------------------
# Everything from here to the end of the section is mode-INDEPENDENT: the
# strategy hands back one shader socket and this applies the coverage rules,
# the alphaTest cutout, the spectrum gate and the render settings to it, the
# same way for all four modes.


class SurfaceAssembly(SpecialSurfaces):
    """Mixed into MaterialBuilder; kept separate only for readability."""

    def build(self):
        """Build the whole material. Returns self."""
        output = self.new('ShaderNodeOutputMaterial', '', COL_OUTPUT, 0)
        surface = None
        try:
            surface = self.special_surface()
            if surface is None:
                surface = self.finish_surface(self.strategy.build(self))
        except Exception as exc:                            # noqa: BLE001
            self.note(DIAG_PARSE_ERROR, 'build-failed',
                      'the surface could not be built (%s); a placeholder is '
                      'drawn instead' % exc)
            surface = None
        if surface is not None:
            self.link(surface, output.inputs['Surface'])
        self.apply_material_settings()
        self.fallback_reason = ensure_visible(self)
        self.graph.prune_orphans()
        try:
            arrange_node_tree(self.nt)
        except Exception as exc:                            # noqa: BLE001
            # Layout is cosmetic. A material that shades correctly must not
            # be lost to a bad node position, so this is a note rather than a
            # build failure - the graph is still wired, just untidy.
            self.note(DIAG_APPROXIMATED, 'layout',
                      'the nodes could not be arranged (%s); the material is '
                      'correct but its graph is laid out as it was built'
                      % exc)
        _store_static_expressions(self.nt, self.static_expressions)
        return self

    # -- coverage and cutout ------------------------------------------------

    def finish_surface(self, surface):
        # Translucency is NOT here: it belongs to the lit result and is
        # applied by SurfaceStrategy.build before the ambient stack, the way
        # the engine's translucentInteractions are drawn under them.
        surface = self.apply_alpha_test(surface)
        surface = self.gate_spectrum(surface)
        self.label_gui_surface()
        return surface

    def apply_translucency(self, surface):
        """Translucency is NOT alpha blending.

        A translucent material is skipped by RB_T_FillDepthBuffer, so it never
        lays down the black the depth-fill pass leaves behind; its
        interactions are linked into vLight->translucentInteractions and drawn
        with GLS_SRCBLEND_ONE | GLS_DSTBLEND_ONE - pure addition onto whatever
        is already in the colour buffer. interaction.vfp ends on
        `MUL result.color, color, fragment.color`, and that alpha is never
        read by a ONE/ONE blend.

        So the apparent opacity of a translucent surface is the BRIGHTNESS of
        its own lit result, never an alpha channel: where diffuse and specular
        are both black the surface is not drawn at all. Add Shader(Transparent,
        lit) is exactly that. Mixing by the diffuse alpha instead - which is
        the obvious-looking thing to do - makes every translucent material
        whose diffuse is a plain 24-bit .tga render fully opaque, and
        models/monsters/revenant/revenant2 is one of those.

        A subview reaches the same place by the other route in the same
        function: it IS drawn by RB_T_FillDepthBuffer, but down-modulated
        rather than filled black, so its interactions also add onto a
        surviving background. textures/lab/labfloor1_d is `mirror` plus a
        diffusemap - a lit mirror floor - and without this the reflection
        under it was painted out.
        """
        if self.ir.coverage != COVERAGE_TRANSLUCENT \
                and self.ir.sort_name != 'subview':
            return surface
        transparent = self.work('ShaderNodeBsdfTransparent', 'translucent')
        add = self.work('ShaderNodeAddShader', 'translucent add')
        self.link(transparent.outputs['BSDF'], add.inputs[0])
        self.link(surface, add.inputs[1])
        return add.outputs['Shader']

    def apply_alpha_test(self, surface):
        """alphaTest, as the engine applies it: to the whole surface.

        Material.cpp sets GLS_DEPTHFUNC_EQUAL on every stage of an opaque or
        perforated material, with the comment "which gets alpha test correct",
        and RB_T_FillDepthBuffer is the only place GL_ALPHA_TEST is ever
        enabled. So the depth pass draws each alpha-tested stage in turn,
        writing depth only where that stage's own test passes, and every later
        pass - interaction AND ambient - is clipped to the union of those
        masks. Two consequences worth stating:

          * the cutout reaches the ambient stages, not only the lit ones;
          * a material with several alpha-tested stages (548 across the five
            game bases) is cut by the UNION of their tests, not by the first.

        Translucent and post-process materials get GLS_DEPTHFUNC_LESS instead
        and are never cut out, which is why this is gated on coverage rather
        than on the presence of an alphaTest.

        The alpha tested is the texture's own times the stage's alpha register
        - what the depth pass's glColor4fv modulates it by. Vertex colour is
        deliberately not in it: the depth pass binds no colour array.
        """
        if self.ir.coverage != COVERAGE_PERFORATED:
            return surface
        tested = self._alpha_tested_stages()
        if not tested:
            return surface

        mask = None
        for stage, alpha in tested:
            test = self.work('ShaderNodeMath', 'alphaTest')
            test.operation = 'GREATER_THAN'
            self.put_expr(test.inputs[1], stage.alpha_test, 0.5)
            self.link(alpha, test.inputs[0])
            gated = self._gate_test_on_condition(stage, test.outputs['Value'])
            if mask is None:
                mask = gated
                continue
            union = self.work('ShaderNodeMath', 'alphaTest union')
            union.operation = 'MAXIMUM'
            self.link(mask, union.inputs[0])
            self.link(gated, union.inputs[1])
            mask = union.outputs['Value']

        mask = self._solid_when_all_conditioned_off([s for s, _a in tested],
                                                    mask)
        transparent = self.work('ShaderNodeBsdfTransparent', 'cutout')
        mix = self.work('ShaderNodeMixShader', 'perforated')
        self.link(mask, mix.inputs['Fac'])
        self.link(transparent.outputs['BSDF'], mix.inputs[1])
        self.link(surface, mix.inputs[2])
        return mix.outputs['Shader']

    def _alpha_tested_stages(self):
        """(stage, alpha socket) for every alpha-tested stage in the material.

        Sampled here rather than collected during the surface build, because
        the union has to cover stages a strategy never drew - a bumpmap stage
        can carry an alphaTest and still be dropped by Simple - and because
        ImageWiring memoises, so asking again costs no extra nodes.
        """
        out = []
        for stage in self.ir.stages:
            if stage.alpha_test is None:
                continue
            _color, alpha = self.sampler.sample(stage, want_alpha=True,
                                                want_vertex_color=False)
            if alpha is not None:
                out.append((stage, alpha))
        return out

    def _condition_socket(self, stage):
        """0 or 1 for a stage's `if`, live where the policy keeps it live."""
        node = self.work('ShaderNodeValue', 'if %s'
                         % (stage.condition.source or ''))
        res = self.params.resolve(stage.condition, 1.0)
        res.value = 1.0 if res.value != 0.0 else 0.0
        if res.driver is not None:
            res.driver = 'float(bool(%s))' % res.driver
        elif res.record is not None:
            res.record = 'float(bool(%s))' % res.record
        self.put(node.outputs['Value'], res)
        return node.outputs['Value']

    def gate_on_condition(self, without, with_stage, stage):
        """Mix between the surface with and without one conditional stage."""
        mix = self.work('ShaderNodeMixShader',
                        'if %s' % (stage.condition.source or ''))
        res = self.params.resolve(stage.condition, 0.0)
        res.value = 1.0 if res.value != 0.0 else 0.0
        if res.driver is not None:
            res.driver = 'float(bool(%s))' % res.driver
        elif res.record is not None:
            res.record = 'float(bool(%s))' % res.record
        self.put(mix.inputs['Fac'], res)
        self.link(without, mix.inputs[1])
        self.link(with_stage, mix.inputs[2])
        try:
            mix['idtech4_expr'] = stage.condition.source or ''
        except (TypeError, AttributeError):
            pass
        return mix.outputs['Shader']

    def _gate_test_on_condition(self, stage, test_socket):
        """A conditioned alpha-tested stage cuts nothing while it is off."""
        if stage.condition is None:
            return test_socket
        live = self._condition_socket(stage)
        gate = self.work('ShaderNodeMath', 'test if live')
        gate.operation = 'MULTIPLY'
        self.link(test_socket, gate.inputs[0])
        self.link(live, gate.inputs[1])
        return gate.outputs['Value']

    def _solid_when_all_conditioned_off(self, stages, mask):
        """RB_T_FillDepthBuffer's `if ( !didDraw ) drawSolid = true;`.

        A perforated material whose alpha-tested stages are ALL switched off
        by their conditions is drawn as an ordinary solid surface - the depth
        pass never ran an alpha test, so nothing was cut out. Every monster in
        Doom 3 depends on this: the dissolve-on-death materials carry two
        alpha-tested stages gated on `if parm7`, and parm7 is zero until the
        thing dies. Without this the mask reads zero and the monster is
        invisible the moment it is imported.

        Only conditioned stages need the rescue, so a material whose tests are
        unconditional keeps the plain union.
        """
        conditioned = [s for s in stages if s.condition is not None]
        if not conditioned or len(conditioned) != len(stages):
            return mask
        drew = None
        for stage in conditioned:
            live = self._condition_socket(stage)
            if drew is None:
                drew = live
                continue
            either = self.work('ShaderNodeMath', 'or')
            either.operation = 'MAXIMUM'
            self.link(drew, either.inputs[0])
            self.link(live, either.inputs[1])
            drew = either.outputs['Value']
        solid = self.work('ShaderNodeMath', 'no test ran')
        solid.operation = 'SUBTRACT'
        solid.use_clamp = True
        solid.inputs[0].default_value = 1.0
        self.link(drew, solid.inputs[1])
        rescued = self.work('ShaderNodeMath', 'cut or solid')
        rescued.operation = 'MAXIMUM'
        self.link(mask, rescued.inputs[0])
        self.link(solid.outputs['Value'], rescued.inputs[1])
        return rescued.outputs['Value']

    # -- spectrum -----------------------------------------------------------

    def gate_spectrum(self, surface):
        """`spectrum <n>` - invisible writing, only lit by a matching light.

        Interaction.cpp refuses to generate an interaction when the surface
        and light spectra differ, so a spectrum-N surface is invisible unless
        you are carrying the matching light. Here that becomes one scene-level
        Spectrum value: 0 shows the ordinary world, N additionally reveals the
        spectrum-N materials. Only 45 materials in the whole corpus, but they
        vanish entirely without it.
        """
        if not self.ir.spectrum:
            return surface
        transparent = self.work('ShaderNodeBsdfTransparent', 'hidden')
        mix = self.work('ShaderNodeMixShader', 'spectrum %d' % self.ir.spectrum)
        mix.name = 'IDTECH4_SpectrumGate'
        scene_spectrum = int(getattr(self.settings, 'spectrum', 0)
                             if self.settings is not None else 0)
        mix.inputs['Fac'].default_value = \
            1.0 if scene_spectrum == self.ir.spectrum else 0.0
        self.link(transparent.outputs['BSDF'], mix.inputs[1])
        self.link(surface, mix.inputs[2])
        if self.params.name == PARAMS_DYNAMIC:
            # The scene Spectrum is a panel value, never a function of time.
            self._record(mix.inputs['Fac'], None,
                         'float(%s() == %d)' % (DRIVER_SPECTRUM_FN,
                                                self.ir.spectrum))
        self.note(DIAG_APPROXIMATED, 'spectrum',
                  'spectrum %d - visible only when the scene Spectrum '
                  'matches; the engine hides only the lit contribution, this '
                  'hides the whole material' % self.ir.spectrum)
        return mix.outputs['Shader']

    def label_gui_surface(self):
        """guiSurf materials draw an interactive GUI no importer reproduces.

        Whatever surface was built stands in for it; drop a frame into the
        tree so that is obvious when the material is opened, not only when
        the report is read.
        """
        if not self.ir.gui_surf:
            return
        frame = self.nt.nodes.new('NodeFrame')
        frame.label = 'guiSurf: %s (GUI not reproducible)' % self.ir.gui_surf
        frame.location = (900, 400)

    # -- datablock settings -------------------------------------------------

    def apply_material_settings(self):
        """Blend method, backface culling and sort offset on the datablock.

        Blender 4.2+ collapsed blend_method: OPAQUE, CLIP and HASHED all land
        on HASHED and only BLEND is distinct, so idTech4's three coverage
        modes map to two material settings and the alphaTest cutout is done in
        the graph rather than by the material.
        """
        mat = self.mat
        translucent = self.ir.coverage == COVERAGE_TRANSLUCENT
        _set_render_method(mat, 'BLENDED' if translucent else 'DITHERED')
        try:
            mat.blend_method = 'BLEND' if translucent else 'HASHED'
        except (TypeError, AttributeError):
            pass
        try:
            mat.use_backface_culling = (self.ir.cull == 'front')
        except AttributeError:
            pass
        # polygonOffset is deliberately NOT reported. decal / polygonOffset
        # materials sit fractionally in front of the surface they are drawn
        # on and Blender has no per-material depth bias, so it cannot be
        # reproduced - but there is nothing anyone can do about that, and the
        # corpus carries enough of them that a diagnostic each buried the
        # findings that ARE actionable. The material still builds exactly the
        # same way; only the note is gone.
        mat['idtech4_material'] = self.ir.name
        mat['idtech4_mode'] = self.profile.name
        mat['idtech4_params'] = self.params.name


def _set_render_method(mat, method):
    """EEVEE Next picks transparency behaviour from surface_render_method;
    blend_method still drives the legacy path and Workbench, so set both."""
    if hasattr(mat, 'surface_render_method'):
        try:
            mat.surface_render_method = method
        except TypeError:
            pass


class MaterialBuilder(SurfaceAssembly):
    """Builds one Blender material. Contains NO mode policy."""

    def __init__(self, mat_ir, blender_mat, profile, params, assets,
                 settings=None):
        self.ir = mat_ir
        self.mat = blender_mat
        self.profile = profile
        self.params = params
        self.assets = assets
        self.settings = settings
        self.notes = []
        self.static_expressions = []
        self.driver_count = 0
        self.fallback_reason = None
        # Set when SpecialSurfaces answered instead of a SurfaceStrategy: a
        # light shader, a broken decl, an editor-image-only decl or a
        # `deform flare`. Callers that compare a built graph against what the
        # engine would draw need to know the graph is a stand-in and not an
        # attempt at the shading.
        self.stood_in = None

        blender_mat.use_nodes = True
        self.nt = blender_mat.node_tree
        self.nt.nodes.clear()

        self.plan = ResourcePlan.build(mat_ir, profile, params, assets)
        self.graph = NodeGraph(self.nt, self.plan, assets)
        self.texcoords = TexCoords(self)
        self.graph.uv_builder = self.texcoords
        self.images = ImageWiring(self)
        self.sampler = StageSampler(self)
        self.compositor = AmbientCompositor(self)
        self.interactions = InteractionModel(self)
        self.strategy = make_strategy(profile)

        for level, kind, message in self.plan.notes:
            self.note(level, kind, message)
        for stage, reason, level, kind in self.plan.dropped_stages:
            self.note(level, kind,
                      '%s not built: %s' % (self._describe_stage(stage),
                                            reason))

    # -- diagnostics --------------------------------------------------------

    def _describe_stage(self, stage):
        """Which stage, in the terms the .mtr author would recognise.

        The index within the material, what kind of stage it is, the map it
        reads and the line it is declared on. The old wording gave only
        `stage <source_line>`, which reads as a stage number and is nothing of
        the sort - "stage 5662" on a material with one stage is not a useful
        thing to tell anyone.
        """
        try:
            index = self.ir.stages.index(stage)
            position = 'stage %d of %d' % (index + 1, len(self.ir.stages))
        except (ValueError, AttributeError):
            position = 'stage'
        kind = stage.blend_name or stage.lighting
        image = ''
        if stage.image is not None:
            image = stage.image.canonical or stage.image.path or ''
        detail = ' '.join(p for p in (kind, image) if p)
        return '%s (%s, line %d)' % (position, detail or 'no map',
                                     stage.source_line)

    def note(self, level, kind, message):
        self.notes.append(MtrDiagnostic(
            level, kind, message,
            filename=self.ir.filename if self.ir else '',
            line=self.ir.line if self.ir else 0,
            material=self.ir.name if self.ir else ''))

    # -- writing a resolved expression into a socket ------------------------

    def put(self, socket, resolution, index=None, source=''):
        """Write one Resolution into one socket.

        The single funnel every parameter goes through. Baked and Skip only
        ever land in the first branch; Dynamic is the only policy that reaches
        the other two.
        """
        if socket is None:
            return False
        value = resolution.value
        try:
            if index is None:
                socket.default_value = value
            else:
                socket.default_value[index] = value
        except (TypeError, AttributeError, IndexError):
            return False

        if resolution.driver is not None:
            return self._drive(socket, index, resolution.driver)
        if resolution.record is not None:
            return self._record(socket, index, resolution.record)
        return True

    def _drive(self, socket, index, expression):
        try:
            fcurve = socket.driver_add('default_value') if index is None \
                else socket.driver_add('default_value', index)
        except (TypeError, RuntimeError):
            return False
        fcurve.driver.type = 'SCRIPTED'
        fcurve.driver.expression = expression
        _bind_frame_dependency(fcurve)
        self.driver_count += 1
        return True

    def _record(self, socket, index, expression):
        """Remember a socket the panel re-folds when a slider moves.

        The path is recorded rather than the socket because prune_orphans()
        runs after the build and can delete the node underneath it.
        """
        try:
            path = socket.path_from_id()
        except (AttributeError, ValueError, TypeError):
            return False
        self.static_expressions.append(
            [path, -1 if index is None else int(index), expression])
        return True

    def put_expr(self, socket, expr, default=1.0, index=None):
        """Resolve an MtrExpr through the policy and write it. Returns the
        Resolution so callers can see whether it was dropped."""
        res = self.params.resolve(expr, default)
        if not res.dropped:
            self.put(socket, res, index)
        if (expr is not None and res.driver is not None and
                expr_grouping_is_surprising(expr)):
            self.note(DIAG_APPROXIMATED, 'expression-grouping',
                      '"%s" groups right-to-left in idTech 4 - the driver is '
                      'parenthesised to match the engine, not C'
                      % (expr.source or repr(expr)))
        return res

    # -- convenience --------------------------------------------------------

    def new(self, *args, **kwargs):
        return self.graph.new(*args, **kwargs)

    def work(self, *args, **kwargs):
        return self.graph.work(*args, **kwargs)

    def link(self, a, b):
        return self.graph.link(a, b)


class TexCoords(object):
    """texgen plus the texture-matrix chain, as one UV socket.

    Reached only through NodeGraph.uv(), which caches by UVKey, so two stages
    that scroll identically share one chain of nodes rather than building two.
    """

    def __init__(self, builder):
        self.b = builder
        self.graph = builder.graph

    def build(self, request):
        if request.is_default:
            return None
        y = 300 - ROW_STEP * request.order
        socket = self._texgen(request, COL_UV, y)
        plan = request.transform
        if plan.kind == 'matrix':
            return self._folded(plan.matrix, socket, COL_UV + COL_UV_STEP, y)
        if plan.kind == 'chain':
            if socket is None:
                socket = self._default_uv(COL_UV, y)
            # MultiplyTextureMatrix composes each new keyword on the RIGHT of
            # the accumulated matrix, so the LAST keyword written in the .mtr
            # is applied to the texture coordinates FIRST. Walking the list in
            # reverse and chaining the nodes reproduces that exactly.
            for step, tform in enumerate(reversed(plan.transforms)):
                socket = self._transform(
                    tform, socket, COL_UV + COL_UV_STEP * (step + 1), y)
            return socket
        return socket

    def _default_uv(self, x, y):
        return self.graph.new('ShaderNodeTexCoord', '', x, y).outputs['UV']

    def _texgen(self, request, x, y):
        """The coordinate source a texgen keyword selects, or None for UV."""
        texgen = request.texgen
        if texgen in (None, '', 'base'):
            return None
        if texgen in ('skybox', 'wobblesky'):
            # R_SkyboxTexGen indexes by (surface point - view origin); see
            # StageSampler._cube_vector. Geometry.Incoming is its negation.
            geom = self.graph.new('ShaderNodeNewGeometry', texgen, x, y)
            flip = self.graph.new('ShaderNodeVectorMath', 'eye to surface',
                                  x + COL_UV_STEP, y)
            flip.operation = 'SCALE'
            flip.inputs['Scale'].default_value = -1.0
            self.graph.link(geom.outputs['Incoming'], flip.inputs[0])
            return flip.outputs['Vector']
        coords = self.graph.new('ShaderNodeTexCoord', texgen, x, y)
        if texgen == 'reflect':
            return coords.outputs['Reflection']
        if texgen == 'screen':
            return coords.outputs['Window']
        return None

    def _folded(self, m, in_socket, x, y):
        """A wholly-constant chain as one Mapping node.

        This is the node-count budget that keeps whole-map imports affordable:
        a typical scrolling-and-scaling stage drops from four nodes to two.
        Only reached when the composed matrix has no rotation or shear term,
        which is what lets Location and Scale carry all of it.
        """
        if in_socket is None:
            in_socket = self._default_uv(x - COL_UV_STEP, y)
        mapping = self.graph.new('ShaderNodeMapping', 'texture matrix', x, y)
        mapping.vector_type = 'POINT'
        mapping.inputs['Location'].default_value = (m[2], m[5], 0.0)
        mapping.inputs['Scale'].default_value = (m[0], m[4], 1.0)
        self.graph.link(in_socket, mapping.inputs['Vector'])
        return mapping.outputs['Vector']

    def _transform(self, tform, in_socket, x, y):
        op = tform.op
        if op == 'shear':
            return self._shear(tform, in_socket, x, y)

        mapping = self.graph.new('ShaderNodeMapping', op, x, y)
        mapping.vector_type = 'POINT'
        self.graph.link(in_socket, mapping.inputs['Vector'])
        b = self.b

        if op in ('translate', 'scroll'):
            b.put_expr(mapping.inputs['Location'], tform.x, 0.0, 0)
            b.put_expr(mapping.inputs['Location'], tform.y, 0.0, 1)

        elif op == 'scale':
            b.put_expr(mapping.inputs['Scale'], tform.x, 1.0, 0)
            b.put_expr(mapping.inputs['Scale'], tform.y, 1.0, 1)

        elif op == 'centerscale':
            # s' = a*s + (0.5 - 0.5a): scale about the texture centre, so the
            # offset has to track the scale rather than being written once.
            rx = b.put_expr(mapping.inputs['Scale'], tform.x, 1.0, 0)
            ry = b.put_expr(mapping.inputs['Scale'], tform.y, 1.0, 1)
            self._centered_offset(mapping, 0, tform.x, rx)
            self._centered_offset(mapping, 1, tform.y, ry)

        elif op == 'rotate':
            self._rotate(mapping, tform)

        return mapping.outputs['Vector']

    def _centered_offset(self, mapping, index, expr, resolution):
        socket = mapping.inputs['Location']
        socket.default_value[index] = 0.5 - 0.5 * resolution.value
        if resolution.driver is not None:
            self.b._drive(socket, index, '0.5 - 0.5 * (%s)' % resolution.driver)
        elif resolution.record is not None:
            self.b._record(socket, index, '0.5 - 0.5 * (%s)' % resolution.record)

    def _rotate(self, mapping, tform):
        """`rotate` is IN CYCLES, not degrees.

        The engine feeds the value straight into sinTable[]/cosTable[], which
        span one full turn over 0..1. It also rotates about (0.5, 0.5), where
        Blender's Mapping node rotates about the origin, so the pivot has to be
        rebuilt in Location.
        """
        res = self.b.params.resolve(tform.x, 0.0)
        angle = res.value * 2.0 * math.pi
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        mapping.inputs['Rotation'].default_value = (0.0, 0.0, angle)
        mapping.inputs['Location'].default_value = (
            -0.5 * cos_a + 0.5 * sin_a + 0.5,
            -0.5 * sin_a - 0.5 * cos_a + 0.5, 0.0)
        live = res.driver or res.record
        if live is None:
            return
        cyc = '(%s) * 6.283185307179586' % live
        loc_x = '(-0.5 * cos(%s)) + (0.5 * sin(%s)) + 0.5' % (cyc, cyc)
        loc_y = '(-0.5 * sin(%s)) - (0.5 * cos(%s)) + 0.5' % (cyc, cyc)
        write = self.b._drive if res.driver is not None else self.b._record
        write(mapping.inputs['Rotation'], 2, cyc)
        write(mapping.inputs['Location'], 0, loc_x)
        write(mapping.inputs['Location'], 1, loc_y)

    def _shear(self, tform, in_socket, x, y):
        """shear a, b -> s' = s + a*t - 0.5a ; t' = b*s + t - 0.5b.

        Not expressible as a Mapping node, and rare enough - 24 stages in the
        whole corpus - that spending six nodes on an exact result is fine.
        """
        g = self.graph
        b = self.b
        ra = b.params.resolve(tform.x, 0.0)
        rb = b.params.resolve(tform.y, 0.0)
        sep = g.new('ShaderNodeSeparateXYZ', 'shear split', x, y)
        g.link(in_socket, sep.inputs['Vector'])

        sx = g.new('ShaderNodeMath', 'shear s', x + 180, y + 100)
        sx.operation = 'MULTIPLY_ADD'
        g.link(sep.outputs['Y'], sx.inputs[0])
        b.put(sx.inputs[1], ra)
        b.put(sx.inputs[2], Resolution(-0.5 * ra.value))
        ax = g.new('ShaderNodeMath', 'shear s+', x + 360, y + 100)
        ax.operation = 'ADD'
        g.link(sep.outputs['X'], ax.inputs[0])
        g.link(sx.outputs['Value'], ax.inputs[1])

        sy = g.new('ShaderNodeMath', 'shear t', x + 180, y - 100)
        sy.operation = 'MULTIPLY_ADD'
        g.link(sep.outputs['X'], sy.inputs[0])
        b.put(sy.inputs[1], rb)
        b.put(sy.inputs[2], Resolution(-0.5 * rb.value))
        ay = g.new('ShaderNodeMath', 'shear t+', x + 360, y - 100)
        ay.operation = 'ADD'
        g.link(sep.outputs['Y'], ay.inputs[0])
        g.link(sy.outputs['Value'], ay.inputs[1])

        comb = g.new('ShaderNodeCombineXYZ', 'shear join', x + 540, y)
        g.link(ax.outputs['Value'], comb.inputs['X'])
        g.link(ay.outputs['Value'], comb.inputs['Y'])
        return comb.outputs['Vector']


class ImageWiring(object):
    """One MtrImage AST -> (color socket, alpha socket).

    Handles every image-program operator the engine bakes offline, plus the
    .bimage fallback for a binary-only distribution. Results are memoised on
    the graph by (program, colorspace, sampling, uv), so a program wired twice
    in one material is wired once.
    """

    def __init__(self, builder):
        self.b = builder
        self.graph = builder.graph
        self.plan = builder.plan
        self.assets = builder.assets

    # -- entry points -------------------------------------------------------

    def color(self, img_expr, usage=None, colorspace='sRGB',
              sampling=DEFAULT_SAMPLING, uv_key=()):
        """(color, alpha) for an image program, or (None, None).

        (None, None) means there is nothing to load - a builtin like _white,
        or a program with no file underneath it - not that something failed.
        """
        if img_expr is None:
            return None, None
        if img_expr.op is None and (img_expr.is_builtin() or not img_expr.path):
            return None, None
        key = (img_expr.canonical or img_expr.path, colorspace,
               sampling.interpolation, sampling.extension, uv_key)
        hit = self.graph.program(key)
        if hit is not None:
            return hit
        return self.graph.remember_program(
            key, self._wire(img_expr, usage, colorspace, sampling, uv_key))

    def normal(self, img_expr, sampling=DEFAULT_SAMPLING, uv_key=()):
        """A Normal socket from a bumpmap image program, or None.

        `map _flat` - which AddImplicitStages gives every interaction material
        that declares no bumpmap of its own - is the engine's flatNormalMap,
        i.e. "use the geometry's own normal". There is no file at base/_flat,
        and resolving it as one put Blender's magenta missing-image
        placeholder on 225 Doom 3 materials. Returning None leaves the BSDF's
        Normal input unlinked, which IS a flat normal.
        """
        if img_expr is None:
            return None
        if img_expr.op is None and img_expr.is_builtin():
            return None
        return self._normal(img_expr, sampling, uv_key)

    # -- the operators ------------------------------------------------------

    def _leaf(self, path, colorspace, sampling, uv_key, label=''):
        """The Image Texture node for one file, via the plan's dedup key."""
        req = self.plan.want_image(path, colorspace, sampling, uv_key,
                                   label or path)
        return self.graph.image(req)

    def _wire(self, img_expr, usage, colorspace, sampling, uv_key):
        op = img_expr.op

        if op is None:
            node = self._leaf(self.assets.resolve(img_expr.path, usage),
                              colorspace, sampling, uv_key, img_expr.path)
            return node.outputs['Color'], node.outputs['Alpha']

        # The engine bakes a whole image program once and caches the result;
        # when the program's own source files are not on disk (a binary-only
        # distribution) that cache is the only way back to a usable texture -
        # and it is already finished, so wire it like an ordinary file rather
        # than rebuilding it from parts.
        baked = self.assets.program_bimage(img_expr, usage)
        if baked is not None:
            node = self._leaf(baked, colorspace, sampling, uv_key,
                              img_expr.canonical)
            return node.outputs['Color'], node.outputs['Alpha']

        if op in ('downsize', 'smoothnormals', 'nativelayout', 'cameralayout',
                  'bakeambientdiffuse', 'bakeambientspecular'):
            # Resolution, layout and prefilter operations with no shading
            # consequence we can reproduce - use the source image directly.
            return self.color(img_expr.args[0], usage, colorspace, sampling,
                              uv_key)

        inner_c, inner_a = self.color(img_expr.args[0] if img_expr.args
                                      else None, usage, colorspace, sampling,
                                      uv_key)
        if inner_c is None:
            return None, None
        g = self.graph

        if op == 'makealpha':
            # "Sets the alpha channel to an average of the RGB channels, sets
            # the RGB channels to white." The source file usually has no real
            # alpha at all, so reading the node's Alpha output would give a
            # constant 1.0 and turn every makeAlpha decal opaque.
            avg = g.work('ShaderNodeVectorMath', 'makeAlpha')
            avg.operation = 'DOT_PRODUCT'
            avg.inputs[1].default_value = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
            g.link(inner_c, avg.inputs[0])
            white = g.work('ShaderNodeRGB', 'makeAlpha white')
            white.outputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
            return white.outputs['Color'], avg.outputs['Value']

        if op == 'makeintensity':
            lum = g.work('ShaderNodeVectorMath', 'makeIntensity')
            lum.operation = 'DOT_PRODUCT'
            lum.inputs[1].default_value = (1.0, 0.0, 0.0)   # engine uses red
            g.link(inner_c, lum.inputs[0])
            return lum.outputs['Value'], lum.outputs['Value']

        if op == 'invertcolor':
            inv = g.work('ShaderNodeInvert', 'invertColor')
            g.link(inner_c, inv.inputs['Color'])
            return inv.outputs['Color'], inner_a

        if op == 'invertalpha':
            if inner_a is None:
                return inner_c, None
            sub = g.work('ShaderNodeMath', 'invertAlpha')
            sub.operation = 'SUBTRACT'
            sub.inputs[0].default_value = 1.0
            g.link(inner_a, sub.inputs[1])
            return inner_c, sub.outputs['Value']

        if op == 'scale':
            s = (list(img_expr.scalars) + [1.0, 1.0, 1.0, 1.0])[:4]
            mul = g.work('ShaderNodeMix', 'scale')
            mul.data_type = 'RGBA'
            mul.blend_type = 'MULTIPLY'
            # R_ImageScale writes back into the 8-bit image, so a scale above
            # 1 saturates rather than running away.
            mul.clamp_result = True
            mul.inputs['Factor'].default_value = 1.0
            g.link(inner_c, mul.inputs[6])
            mul.inputs[7].default_value = (s[0], s[1], s[2], 1.0)
            alpha_out = inner_a
            if inner_a is not None and s[3] != 1.0:
                am = g.work('ShaderNodeMath', 'scale alpha')
                am.operation = 'MULTIPLY'
                am.inputs[1].default_value = s[3]
                g.link(inner_a, am.inputs[0])
                alpha_out = am.outputs['Value']
            return mul.outputs[2], alpha_out

        if op == 'add':
            b_c, _b_a = self.color(img_expr.args[1] if len(img_expr.args) > 1
                                   else None, usage, colorspace, sampling,
                                   uv_key)
            if b_c is None:
                return inner_c, inner_a
            addn = g.work('ShaderNodeMix', 'add')
            addn.data_type = 'RGBA'
            addn.blend_type = 'ADD'
            # R_ImageAdd is ClampInt(0, 255, a + b) - two textures added into
            # an 8-bit buffer, not an HDR sum.
            addn.clamp_result = True
            addn.inputs['Factor'].default_value = 1.0
            g.link(inner_c, addn.inputs[6])
            g.link(b_c, addn.inputs[7])
            return addn.outputs[2], inner_a

        return inner_c, inner_a

    # -- normals ------------------------------------------------------------

    def _normal(self, img_expr, sampling, uv_key):
        op = img_expr.op

        if op is not None:
            baked = self.assets.program_bimage(img_expr, USE_NORMAL)
            if baked is not None:
                # The cached composite is already a finished normal map, not a
                # height field, so it needs no Bump-node conversion.
                node = self._leaf(baked, 'Non-Color', sampling, uv_key,
                                  img_expr.canonical)
                return self._normal_map(node).outputs['Normal']

        if op == 'addnormals' and len(img_expr.args) == 2:
            return self._addnormals(img_expr, sampling, uv_key)

        if op == 'heightmap':
            bump = self._heightmap_bump(img_expr, sampling, uv_key)
            return bump.outputs['Normal'] if bump is not None else None

        if op in ('downsize', 'smoothnormals'):
            return self._normal(img_expr.args[0], sampling, uv_key)

        node = self._normal_source(img_expr, sampling, uv_key)
        if node is None:
            return None
        return self._normal_map(node).outputs['Normal']

    def _addnormals(self, img_expr, sampling, uv_key):
        """addnormals(a, b) as R_AddNormalMaps sums it.

        When b is a heightmap, Blender's Bump node stacked on a's Normal is
        the same construction and cheaper. When b is a second NORMAL MAP the
        engine decodes both, adds, renormalises and re-encodes - 108 Dark Mod
        materials were losing half their surface detail to a second pass that
        was built and then left orphaned.
        """
        base, detail = img_expr.args[0], img_expr.args[1]
        if detail.op == 'heightmap':
            bump = self._heightmap_bump(detail, sampling, uv_key)
            base_node = self._normal_source(base, sampling, uv_key)
            if bump is None:
                return (self._normal_map(base_node).outputs['Normal']
                        if base_node is not None else None)
            if base_node is not None:
                self.graph.link(self._normal_map(base_node).outputs['Normal'],
                                bump.inputs['Normal'])
            return bump.outputs['Normal']

        a_node = self._normal_source(base, sampling, uv_key)
        b_node = self._normal_source(detail, sampling, uv_key)
        if a_node is None:
            return (self._normal_map(b_node).outputs['Normal']
                    if b_node is not None else None)
        if b_node is None:
            return self._normal_map(a_node).outputs['Normal']
        return self._sum_normals(a_node, b_node)

    def _sum_normals(self, a_node, b_node):
        """decode, add, renormalise, re-encode - R_AddNormalMaps.

        The engine works in 0..255 bytes: it maps each channel to -1..1, adds
        the two vectors, normalises and writes the encoded result back. Doing
        it in the shader means the same three steps on the colour, then one
        Normal Map node reading the re-encoded result.
        """
        g = self.graph
        dec_a = g.work('ShaderNodeVectorMath', 'decode a')
        dec_a.operation = 'MULTIPLY_ADD'
        dec_a.inputs[1].default_value = (2.0, 2.0, 2.0)
        dec_a.inputs[2].default_value = (-1.0, -1.0, -1.0)
        g.link(a_node.outputs['Color'], dec_a.inputs[0])

        dec_b = g.work('ShaderNodeVectorMath', 'decode b')
        dec_b.operation = 'MULTIPLY_ADD'
        dec_b.inputs[1].default_value = (2.0, 2.0, 2.0)
        dec_b.inputs[2].default_value = (-1.0, -1.0, -1.0)
        g.link(b_node.outputs['Color'], dec_b.inputs[0])

        add = g.work('ShaderNodeVectorMath', 'addnormals')
        add.operation = 'ADD'
        g.link(dec_a.outputs['Vector'], add.inputs[0])
        g.link(dec_b.outputs['Vector'], add.inputs[1])

        norm = g.work('ShaderNodeVectorMath', 'renormalise')
        norm.operation = 'NORMALIZE'
        g.link(add.outputs['Vector'], norm.inputs[0])

        enc = g.work('ShaderNodeVectorMath', 'encode')
        enc.operation = 'MULTIPLY_ADD'
        enc.inputs[1].default_value = (0.5, 0.5, 0.5)
        enc.inputs[2].default_value = (0.5, 0.5, 0.5)
        g.link(norm.outputs['Vector'], enc.inputs[0])

        nmap = g.work('ShaderNodeNormalMap', 'Normal Map')
        if _NORMAL_MAP_HAS_CONVENTION:
            nmap.convention = 'DIRECTX'
        g.link(enc.outputs['Vector'], nmap.inputs['Color'])
        return nmap.outputs['Normal']

    def _normal_source(self, img_expr, sampling, uv_key):
        """The Image Texture node under a normal-map expression, or None."""
        if img_expr is None:
            return None
        if img_expr.op is not None:
            baked = self.assets.program_bimage(img_expr, USE_NORMAL)
            if baked is not None:
                return self._leaf(baked, 'Non-Color', sampling, uv_key,
                                  img_expr.canonical)
            if img_expr.op in ('downsize', 'smoothnormals'):
                return self._normal_source(img_expr.args[0], sampling, uv_key)
            return None
        if img_expr.is_builtin() or not img_expr.path:
            return None
        return self._leaf(self.assets.resolve(img_expr.path, USE_NORMAL),
                          'Non-Color', sampling, uv_key, img_expr.path)

    def _normal_map(self, node):
        """A Normal Map node reading an encoded normal texture.

        idTech4 normal maps are DirectX convention (green down). Blender 5.1+
        has an explicit switch; older versions need the green channel inverted
        by hand.
        """
        g = self.graph
        nmap = g.work('ShaderNodeNormalMap', 'Normal Map')
        if _NORMAL_MAP_HAS_CONVENTION:
            nmap.convention = 'DIRECTX'
            g.link(node.outputs['Color'], nmap.inputs['Color'])
            return nmap
        sep = g.work('ShaderNodeSeparateColor', 'split')
        sep.mode = 'RGB'
        inv = g.work('ShaderNodeMath', 'invert green')
        inv.operation = 'SUBTRACT'
        inv.use_clamp = True
        inv.inputs[0].default_value = 1.0
        comb = g.work('ShaderNodeCombineColor', 'recombine')
        comb.mode = 'RGB'
        g.link(node.outputs['Color'], sep.inputs['Color'])
        g.link(sep.outputs['Green'], inv.inputs[1])
        g.link(sep.outputs['Red'], comb.inputs['Red'])
        g.link(inv.outputs['Value'], comb.inputs['Green'])
        g.link(sep.outputs['Blue'], comb.inputs['Blue'])
        g.link(comb.outputs['Color'], nmap.inputs['Color'])
        return nmap

    def _heightmap_bump(self, img_expr, sampling, uv_key):
        node = self._normal_source(img_expr.args[0] if img_expr.args else None,
                                   sampling, uv_key)
        if node is None:
            return None
        bump = self.graph.work('ShaderNodeBump', 'heightmap')
        scale = img_expr.scalars[0] if img_expr.scalars else 1.0
        # R_HeightmapToNormalMap's scale is in 0-255 height units; Blender's
        # Bump Strength is 0..1 over the same visual range, so a direct copy
        # would be wildly over-strong. This ratio matches the engine's own
        # default heightmap scale of 1.
        bump.inputs['Strength'].default_value = min(1.0, abs(scale) / 6.0)
        bump.inputs['Distance'].default_value = 1.0
        self.graph.link(node.outputs['Color'], bump.inputs['Height'])
        return bump


# ---------------------------------------------------------------------------
# Stage sampling
# ---------------------------------------------------------------------------

class StageSampler(object):
    """One MtrStage -> (color socket, alpha socket).

    Everything between the image and the blend equation: the rgb and alpha
    colour registers, `colored`, vertexColor, and the parameter policy applied
    to all of them.
    """

    def __init__(self, builder):
        self.b = builder
        self.graph = builder.graph
        self.plan = builder.plan

    def sample(self, stage, want_alpha=True, want_vertex_color=True):
        """want_alpha is False for blend modes that ignore alpha entirely -
        the additive and multiplicative ones - so their alpha maths is never
        built rather than built and then pruned.

        want_vertex_color is False for interaction stages under the engine
        pass model, which apply vertexColor once per interaction SET: the
        engine holds one inter.vertexColor, not one per stage.
        """
        color, alpha = self._texture(stage)
        if color is None:
            color = self._builtin_color(stage)
        if self._is_bumpy_environment(stage):
            # RB_PrepareStageTexturing binds bumpyEnvironment.vfp instead of
            # environment.vfp when a `texgen reflect` stage sits on a
            # bumpmapped material, and that program ends on
            # `MOV result.color.xyz, R0` - there is no
            # `MUL result.color, color, fragment.color` at the end of it. So
            # neither the stage's own rgb registers nor its vertexColor reach
            # the screen, unlike its unbumped sibling which keeps both.
            return color, None
        color = self._apply_rgb(stage, color)
        alpha = self._apply_alpha(stage, alpha) if want_alpha else None
        if want_vertex_color and self.b.profile.vertex_color:
            color = self._apply_vertex_color(stage, color)
            alpha = self._apply_vertex_alpha(stage, alpha)
        return color, alpha

    def _is_bumpy_environment(self, stage):
        """True when this stage is drawn by bumpyEnvironment.vfp."""
        return (stage.texgen == 'reflect'
                and stage.tex_kind == TEX_CUBE
                and self.b.ir.stage_by_lighting('bump') is not None)

    # -- the texture --------------------------------------------------------

    def _texture(self, stage):
        if stage.tex_kind == TEX_CUBE:
            return self._cube(stage)
        if stage.tex_kind != TEX_FILE:
            return None, None
        sampling = Sampling.for_stage(stage)
        uv = self.plan.want_uv(self.b.params.transform_chain(stage),
                               stage.texgen, stage.texgen_args)
        return self.b.images.color(stage.image, _stage_usage(stage),
                                   _stage_colorspace(stage), sampling, uv.key)

    def _cube(self, stage):
        """A cubeMap/cameraCubeMap stage, as an Environment Texture.

        texGen reflect is the engine's environment mapping: the reflection
        vector indexes the cube. Blender's Reflection texture coordinate is
        the same vector, and an Environment Texture over the baked-out
        equirectangular version of the cube samples it by direction, so the
        pair reproduces the stage without a cube-map node existing at all.
        """
        img = self.b.assets.cubemap(
            stage.image, stage.tex_detail == 'cameracubemap',
            note=lambda rel: self.b.note(
                DIAG_UNSUPPORTED, 'cubeMap',
                'cubeMap %s: none of the six faces (%s_px.tga and siblings) '
                'are on disk, so the stage contributes black' % (rel, rel)))
        if img is None:
            return None, None
        node = self.graph.work('ShaderNodeTexEnvironment',
                               stage.tex_detail or 'cubeMap')
        node.projection = 'EQUIRECTANGULAR'
        node.image = img
        _set_image_colorspace(img, _stage_colorspace(stage))
        self.graph.link(self._cube_vector(stage), node.inputs['Vector'])
        # Environment Texture has no Alpha output; a cube stage that needs one
        # reads it from the framebuffer (gl_dst_alpha) or from a maskColor
        # stage, never from the cube itself.
        return node.outputs['Color'], None

    def _cube_vector(self, stage):
        """The direction that indexes the cube, per the engine's texgen.

        `texgen reflect` is environment mapping: R_ReflectionTexGen builds the
        reflection vector, which is what Blender's Reflection coordinate is.

        `texgen skybox` and `wobblesky` are NOT that. R_SkyboxTexGen sets
        texCoords[i] = verts[i].xyz - localViewOrigin - the vector FROM the
        eye TO the surface point - and indexes the cube with it. Blender's
        Geometry node gives Incoming, which points from the shading point back
        toward the viewer, so the engine's vector is its negation. This used
        to use the Window coordinate, which is a 2D screen position: an
        Environment Texture fed screen space draws the sky flattened onto the
        viewport instead of surrounding the scene, which is why skyboxes came
        out wrong while their cube map baked perfectly.
        """
        g = self.graph
        if stage.texgen in ('skybox', 'wobblesky'):
            geom = g.work('ShaderNodeNewGeometry', stage.texgen)
            flip = g.work('ShaderNodeVectorMath', 'eye to surface')
            flip.operation = 'SCALE'
            flip.inputs['Scale'].default_value = -1.0
            g.link(geom.outputs['Incoming'], flip.inputs[0])
            return flip.outputs['Vector']
        coords = g.work('ShaderNodeTexCoord', stage.texgen or 'reflect')
        return coords.outputs['Reflection']

    def _builtin_color(self, stage):
        """A flat colour standing in for a builtin image or no map at all.

        _white and _black are real engine images and the stage genuinely draws
        them; everything else that lands here (_currentRender, a render target)
        is something we cannot sample, and white is the neutral choice that
        leaves the stage's own colour registers in charge of what it looks
        like.
        """
        path = (stage.image.path or '').lower() if stage.image else ''
        level = 0.0 if path == '_black' else 1.0
        rgb = self.graph.work('ShaderNodeRGB', 'builtin ' + (path or 'white'))
        rgb.outputs['Color'].default_value = (level, level, level, 1.0)
        return rgb.outputs['Color']

    # -- colour registers ---------------------------------------------------

    def _apply_rgb(self, stage, color_socket):
        r, g, b = stage.color[0], stage.color[1], stage.color[2]
        if r is None and g is None and b is None:
            return color_socket
        tint = self.graph.work('ShaderNodeCombineColor', 'rgb')
        tint.mode = 'RGB'
        for socket_name, expr in (('Red', r), ('Green', g), ('Blue', b)):
            self.b.put(tint.inputs[socket_name], self._register(expr, 1.0))
        mul = self.graph.work('ShaderNodeMix', 'rgb x tex')
        mul.data_type = 'RGBA'
        mul.blend_type = 'MULTIPLY'
        mul.inputs['Factor'].default_value = 1.0
        self.graph.link(color_socket, mul.inputs[6])
        self.graph.link(tint.outputs['Color'], mul.inputs[7])
        return mul.outputs[2]

    def _apply_alpha(self, stage, alpha_socket):
        expr = stage.color[3]
        if expr is None:
            return alpha_socket
        res = self._register(expr, 1.0)
        if res.dropped:
            return alpha_socket
        if alpha_socket is None:
            val = self.graph.work('ShaderNodeValue', 'alpha')
            self.b.put(val.outputs['Value'], res)
            return val.outputs['Value']
        mul = self.graph.work('ShaderNodeMath', 'alpha x tex')
        mul.operation = 'MULTIPLY'
        self.b.put(mul.inputs[1], res)
        self.graph.link(alpha_socket, mul.inputs[0])
        return mul.outputs['Value']

    def _register(self, expr, default):
        """One colour register, resolved and clamped.

        Colour registers clamp to 0..1 in the engine - R_SetDrawInteraction
        and glColor4fv both do it - so an expression that runs past 1 does not
        blow the surface out, it saturates.
        """
        res = self.b.params.resolve(expr, default)
        res.value = min(1.0, max(0.0, res.value))
        if res.driver is not None:
            res.driver = 'min(1.0, max(0.0, %s))' % res.driver
        elif res.record is not None:
            res.record = 'min(1.0, max(0.0, %s))' % res.record
        return res

    # -- vertex colour ------------------------------------------------------

    def _apply_vertex_color(self, stage, color):
        if stage.vertex_color is None:
            return color
        return self.vertex_color_socket(stage.vertex_color, color)

    def _apply_vertex_alpha(self, stage, alpha):
        """vertexColor modulates ALPHA as well as rgb.

        The engine's per-vertex colour reaches the stage through glColor4fv,
        which carries four components, and RB_ARB2_DrawInteraction's
        env[16]/env[17] scale-and-bias pair is applied to the whole colour.
        Weighting only the rgb leaves a maskColor stage depositing a mask at
        full strength on geometry the vertex alpha is fading out, and every
        gl_dst_alpha stage reading it then draws too strong.

        `inverseVertexColor` does NOT invert this. It sets a GL_COMBINE RGB
        combiner, and only the RGB one - alpha stays a plain modulate either
        way. Inverting it here reads 1 - a where the engine reads a, which on
        Quake 4's smolder materials is a factor of 0.54 out.
        """
        if stage.vertex_color is None or alpha is None:
            return alpha
        g = self.graph
        attr = g.work('ShaderNodeVertexColor', 'vertexColor alpha')
        src = attr.outputs['Alpha']
        mul = g.work('ShaderNodeMath', 'alpha x vertex colour')
        mul.operation = 'MULTIPLY'
        g.link(alpha, mul.inputs[0])
        g.link(src, mul.inputs[1])
        return mul.outputs['Value']

    def vertex_color_socket(self, mode, color):
        """Multiply a colour by the mesh's vertex colour, or its inverse."""
        if mode is None or color is None:
            return color
        g = self.graph
        attr = g.work('ShaderNodeVertexColor', 'vertexColor')
        src = attr.outputs['Color']
        if mode == 'inverse':
            inv = g.work('ShaderNodeInvert', 'inverseVertexColor')
            g.link(src, inv.inputs['Color'])
            src = inv.outputs['Color']
        mul = g.work('ShaderNodeMix', 'vertex colour')
        mul.data_type = 'RGBA'
        mul.blend_type = 'MULTIPLY'
        mul.inputs['Factor'].default_value = 1.0
        g.link(color, mul.inputs[6])
        g.link(src, mul.inputs[7])
        return mul.outputs[2]


# ---------------------------------------------------------------------------
# Ambient compositing
# ---------------------------------------------------------------------------
# How one ambient stage lands on top of what is already drawn. These are the
# OpenGL blend equations the engine sets with GL_State, not an interpretation
# of them.
#
#   'replace'   overwrite; gl_one,gl_zero never reads source alpha at all
#   'add'       additive
#   'mix'       lerp by the source alpha
#   'filter'    multiplicative darkening
#   'invfilter' multiply by (1 - source)
#   'none'      draws nothing

_BLEND_STRATEGY = {
    ('gl_one', 'gl_zero'): 'replace',
    ('gl_zero', 'gl_one'): 'none',
    ('gl_one', 'gl_one'): 'add',
    ('gl_src_alpha', 'gl_one'): 'add',
    ('gl_dst_alpha', 'gl_one'): 'add',
    ('gl_one', 'gl_src_alpha'): 'add',
    ('gl_src_alpha_saturate', 'gl_one'): 'add',
    ('gl_src_alpha', 'gl_one_minus_src_alpha'): 'mix',
    ('gl_one', 'gl_one_minus_src_alpha'): 'mix',
    ('gl_dst_color', 'gl_zero'): 'filter',
    ('gl_zero', 'gl_src_color'): 'filter',
    ('gl_dst_color', 'gl_one'): 'add',
    ('gl_dst_color', 'gl_src_color'): 'filter',
    ('gl_zero', 'gl_one_minus_src_color'): 'invfilter',
    ('gl_zero', 'gl_one_minus_src_alpha'): 'invfilter',
    ('gl_one_minus_dst_color', 'gl_zero'): 'filter',
    ('gl_one_minus_dst_color', 'gl_one'): 'add',
    ('gl_dst_alpha', 'gl_one_minus_dst_alpha'): 'mix',
    ('gl_one_minus_src_alpha', 'gl_one_minus_src_color'): 'mix',
}

# A doubling the ENGINE performs on gamma-encoded bytes, expressed as the
# factor that does the same job to the linear values Blender hands us. Used
# by the gl_dst_color,gl_src_color 2x modulate - see AmbientCompositor._filter
# for why the exponent belongs here and nowhere else.
#
# 2.2 rather than sRGB's own 2.4: the piecewise curve's EFFECTIVE exponent is
# ~2.2, and that is what matters at the mid-grey these maps are built around.
# Exact only under a pure power law, so the residual error lives in sRGB's
# linear toe - the darkest texels of a decal, where it reads as a few percent
# of lift. Removing that would take a piecewise sRGB node group either side
# of the multiply, about a dozen Math nodes per stage, which is not worth it.
GAMMA_DOUBLE = 2.0 ** 2.2           # 4.5948

# Blend pairs that ignore source alpha entirely, so its maths is never built.
_ALPHA_BLIND_BLENDS = frozenset((
    ('gl_one', 'gl_zero'), ('gl_one', 'gl_one'),
    ('gl_dst_color', 'gl_zero'), ('gl_zero', 'gl_src_color'),
    ('gl_dst_color', 'gl_one'), ('gl_dst_color', 'gl_src_color'),
    ('gl_zero', 'gl_one_minus_src_color'),
    ('gl_one_minus_dst_color', 'gl_zero'),
    ('gl_one_minus_dst_color', 'gl_one'),
))


class AmbientCompositor(object):
    """Layers ambient stages over a background, one blend equation each."""

    def __init__(self, builder):
        self.b = builder
        self.graph = builder.graph
        self.dest_alpha = None          # what a maskColor stage deposited
        self.dest_alpha_set = False
        self.surface_is_base = True     # nothing drawn yet

    def wants_alpha(self, stage):
        return stage.blend_pair() not in _ALPHA_BLIND_BLENDS

    # -- the mask idiom -----------------------------------------------------

    def take_mask(self, stage, alpha):
        """Record a maskColor stage's alpha and draw nothing.

        GLS_COLORMASK becomes glColorMask(0, 0, 0, 1): RGB writes off, alpha
        writes on. Such a stage exists only to deposit a mask in the
        framebuffer's alpha for a later gl_dst_alpha stage to read back - its
        own colour never reaches the screen, and drawing it paints a solid
        white blob, because makeAlpha sets RGB to 255 by definition.

        The mask is an ordinary texture, so no framebuffer read is needed:
        gl_dst_alpha,gl_one multiplies the consumer's colour by it, and
        gl_dst_alpha,gl_one_minus_dst_alpha uses it as the Mix factor. 827
        maskColor stages and 1,086 gl_dst_alpha stages across the five bases;
        621 materials use the paired idiom.
        """
        if alpha is not None:
            self.dest_alpha = alpha
        self.dest_alpha_set = True

    # -- one stage ----------------------------------------------------------

    def composite(self, below, stage, color, alpha):
        strategy = _BLEND_STRATEGY.get(stage.blend_pair())
        if strategy is None:
            strategy = 'add'
            self.b.note(DIAG_APPROXIMATED, 'blend',
                        'blend %s,%s has no exact Blender equivalent - '
                        'approximated as additive'
                        % (stage.blend_src, stage.blend_dst))
        if strategy == 'none':
            return below

        if strategy == 'replace':
            return self._replace(below, color, alpha)

        if stage.reads_dest_alpha and self.dest_alpha is None \
                and not self.dest_alpha_set:
            self.b.note(DIAG_APPROXIMATED, 'blend gl_dst_alpha',
                        'source factor is the framebuffer alpha but no '
                        'maskColor stage set one - drawn unmasked')

        if strategy == 'add':
            result = self._add(below, stage, color)
        elif strategy == 'mix':
            result = self._mix(below, stage, color, alpha)
        else:
            result = self._filter(below, stage, strategy, color, alpha)
        self.surface_is_base = False
        return result

    def _emission(self, color, label='stage'):
        node = self.graph.work('ShaderNodeEmission', label)
        node.inputs['Strength'].default_value = 1.0
        self.graph.link(color, node.inputs['Color'])
        return node.outputs['Emission']

    def _replace(self, below, color, alpha):
        """`blend gl_one, gl_zero` overwrites and never reads source alpha.

        368 Doom 3 stages, and `map <texture>` with no blend keyword is the
        commonest stage there is. The alpha mix here is only for the case
        where a caller has handed one in deliberately.
        """
        shader = self._emission(color, 'replace')
        self.surface_is_base = False
        if alpha is None:
            return shader
        mix = self.graph.work('ShaderNodeMixShader', 'alpha')
        self.graph.link(alpha, mix.inputs['Fac'])
        self.graph.link(below, mix.inputs[1])
        self.graph.link(shader, mix.inputs[2])
        return mix.outputs['Shader']

    def _add(self, below, stage, color):
        if stage.blend_src in ('gl_dst_color', 'gl_one_minus_dst_color'):
            # dst*src + dst is not an addition: the source factor reads the
            # framebuffer, so the stage brightens what is behind it in
            # proportion to what is already there. Blender cannot sample that,
            # and a plain Add Shader is the closest thing that keeps the
            # stage's own colour - but it is an approximation and says so.
            self.b.note(DIAG_APPROXIMATED, 'blend %s,%s'
                        % (stage.blend_src, stage.blend_dst),
                        'the source factor multiplies by the framebuffer, '
                        'which Blender cannot read; drawn as a plain additive '
                        'stage')
        node = self.graph.work('ShaderNodeEmission', 'add')
        self.graph.link(color, node.inputs['Color'])
        if stage.reads_dest_alpha and self.dest_alpha is not None:
            # gl_dst_alpha as the SOURCE factor means the stage's colour is
            # multiplied by the mask before being added. Emission Strength
            # does exactly that multiply, for one link.
            self.graph.link(self.dest_alpha, node.inputs['Strength'])
        else:
            node.inputs['Strength'].default_value = 1.0
        add = self.graph.work('ShaderNodeAddShader', 'blend add')
        self.graph.link(below, add.inputs[0])
        self.graph.link(node.outputs['Emission'], add.inputs[1])
        return add.outputs['Shader']

    def _mix(self, below, stage, color, alpha):
        shader = self._emission(color, 'blend')
        mix = self.graph.work('ShaderNodeMixShader', 'blend')
        # gl_dst_alpha,gl_one_minus_dst_alpha lerps between the stage and what
        # is behind it using the mask, not the stage's own alpha.
        fac = self.dest_alpha if (stage.reads_dest_alpha and
                                  self.dest_alpha is not None) else alpha
        if fac is not None:
            self.graph.link(fac, mix.inputs['Fac'])
        else:
            mix.inputs['Fac'].default_value = 1.0
        self.graph.link(below, mix.inputs[1])
        self.graph.link(shader, mix.inputs[2])
        return mix.outputs['Shader']

    def _filter(self, below, stage, strategy, color, alpha):
        """Multiplicative blends against the framebuffer.

        A Transparent BSDF tinted with a colour IS a multiply of what is
        behind it, per channel, with no approximation at all. So
        `blend gl_dst_color, gl_zero` (result = dst * src) is Transparent(src)
        exactly - as long as nothing has been drawn underneath yet, which for
        a decal on its own geometry is the normal case. The older code threw
        the stage's colour away and mixed toward BLACK by 1 - luminance, which
        turned every coloured filter decal into a grey smudge:
        textures/decals/scannersquare is a red scanner square and came out
        black because its red never reached the output.

        gl_dst_color,gl_src_color is src*dst + dst*src - a 2x modulate, not a
        framebuffer read Blender cannot do, so it belongs here too. The 2x is
        not cosmetic: these maps are mid-grey centred (grunge7.tga means 0.53)
        so 0.5 is the no-op value, and tinting with src alone halves the whole
        decal. 139 Quake 4 materials and 100 Prey ones use it, nearly all of
        them grunge and stain decals.
        """
        multiply_of_dst = {
            ('gl_dst_color', 'gl_zero'): 'src',
            ('gl_zero', 'gl_src_color'): 'src',
            ('gl_dst_color', 'gl_src_color'): 'twice_src',
            ('gl_zero', 'gl_one_minus_src_color'): 'inv_src',
            ('gl_zero', 'gl_one_minus_src_alpha'): 'inv_alpha',
        }.get(stage.blend_pair())
        g = self.graph

        if multiply_of_dst is not None and self.surface_is_base:
            tint = color
            if multiply_of_dst == 'twice_src':
                # 2 ** 2.2, not 2, and the exponent is the whole point.
                #
                # The engine multiplies 8-bit GAMMA-ENCODED values: its
                # textures and its framebuffer are both stored that way and
                # glBlendFunc runs straight on them. Blender multiplies
                # LINEAR ones, because the image node has already decoded the
                # texture. For a plain filter that costs nothing - a power
                # law distributes over multiplication, so
                # display(linear(s) * linear(d)) is exactly s * d - which is
                # why every other multiply here needs no correction. A
                # CONSTANT does not survive that identity: to be worth 2x in
                # gamma space it has to be 2**gamma in linear space.
                #
                # Doubling in the wrong space is not subtle. These maps are
                # authored around byte 128 as the no-op value (2 * 0.502
                # clamps to 1.0, so the background vanishes) - 135 of the 163
                # Quake 4 and Prey textures measured sit in 0.49..0.51. A
                # plain 2 turns that neutral into a 0.43 tint and every decal
                # grows a smoked-glass panel: blasterscorch_strip rendered a
                # white wall at 0.671 instead of leaving it alone.
                #
                # Applied to the sampled socket rather than to a re-read of
                # the image, so it composes with whatever else fed the stage
                # colour - `colored`, an rgb expression, a texture matrix.
                # levelShotDetail is `blend GL_DST_COLOR, GL_SRC_COLOR` plus
                # `colored`, and sampling the raw image again would drop its
                # vertex colour on the floor.
                #
                # Still deliberately unclamped: GL clamps the finished pixel,
                # not the factor, so a tint above 1 is what correctly
                # brightens where the destination is dark.
                twice = g.work('ShaderNodeVectorMath', 'twice src (gamma)')
                twice.operation = 'SCALE'
                twice.inputs['Scale'].default_value = GAMMA_DOUBLE
                g.link(color, twice.inputs[0])
                tint = twice.outputs['Vector']
            elif multiply_of_dst == 'inv_alpha':
                inv = g.work('ShaderNodeMath', 'one minus alpha')
                inv.operation = 'SUBTRACT'
                inv.use_clamp = True
                inv.inputs[0].default_value = 1.0
                g.link(alpha, inv.inputs[1])
                tint = inv.outputs['Value']
            elif multiply_of_dst == 'inv_src':
                invert = g.work('ShaderNodeInvert', 'one minus src')
                g.link(color, invert.inputs['Color'])
                tint = invert.outputs['Color']
            filt = g.work('ShaderNodeBsdfTransparent', 'filter')
            g.link(tint, filt.inputs['Color'])
            return filt.outputs['BSDF']

        # Either not a pure multiply of the destination
        # (gl_one_minus_dst_color,gl_zero is src * (1 - dst)) or layered over
        # something already drawn. Both genuinely need to read the
        # framebuffer, so keep the luminance approximation.
        self.b.note(DIAG_APPROXIMATED,
                    'blend ' + ('filter' if strategy == 'filter'
                                else 'inverse filter'),
                    'multiplicative blend approximated by luminance darkening')
        lum = g.work('ShaderNodeVectorMath', 'filter luminance')
        lum.operation = 'DOT_PRODUCT'
        lum.inputs[1].default_value = (0.299, 0.587, 0.114)
        g.link(color, lum.inputs[0])
        keep = lum.outputs['Value']            # of the destination, how much
        if strategy == 'invfilter':
            sub = g.work('ShaderNodeMath', 'darken')
            sub.operation = 'SUBTRACT'
            sub.use_clamp = True
            sub.inputs[0].default_value = 1.0
            g.link(keep, sub.inputs[1])
            keep = sub.outputs['Value']

        if self.surface_is_base:
            # Mixing a white Transparent toward a black Emission by 1 - keep
            # is algebraically background * keep - and a Transparent tinted
            # with grey `keep` is the same thing in one node. It is not the
            # same to surface_draws, though: an untinted Transparent and a
            # black Emission both read as "draws nothing", so the Mix form
            # was invisible whenever nothing had been drawn underneath, and
            # ensure_visible threw the whole material away for a placeholder.
            filt = g.work('ShaderNodeBsdfTransparent', 'filter')
            g.link(keep, filt.inputs['Color'])
            return filt.outputs['BSDF']

        fac = g.work('ShaderNodeMath', 'one minus keep')
        fac.operation = 'SUBTRACT'
        fac.use_clamp = True
        fac.inputs[0].default_value = 1.0
        g.link(keep, fac.inputs[1])
        fac = fac.outputs['Value']
        black = g.work('ShaderNodeEmission', 'filter')
        black.inputs['Color'].default_value = (0.0, 0.0, 0.0, 1.0)
        mix = g.work('ShaderNodeMixShader', 'filter')
        g.link(fac, mix.inputs['Fac'])
        g.link(below, mix.inputs[1])
        g.link(black.outputs['Emission'], mix.inputs[2])
        return mix.outputs['Shader']


# ---------------------------------------------------------------------------
# Interaction model
# ---------------------------------------------------------------------------

class InteractionSet(object):
    """One flushed drawInteraction_t.

    vertex_color is whatever the LAST diffuse or specular stage of the set
    left in the accumulator, which is not necessarily the diffuse stage's own.
    """

    __slots__ = ('bump', 'diffuse', 'specular', 'vertex_color')

    def __init__(self, bump, diffuse, specular, vertex_color):
        self.bump = bump
        self.diffuse = diffuse
        self.specular = specular
        self.vertex_color = vertex_color


class InteractionModel(object):
    """Splits interaction stages into passes the way the engine does."""

    def __init__(self, builder):
        self.b = builder

    def sets(self):
        """RB_CreateSingleDrawInteractions, reproduced.

        The engine walks the stages with ONE running drawInteraction_t and
        flushes it as it goes (tr_render.cpp):

            SL_BUMP     submit, clear diffuse+specular, take the new bump
            SL_DIFFUSE  submit *if a diffuse is already held*, take this one
            SL_SPECULAR submit *if a specular is already held*, take this one

        and submits once more at the end. Each submitted set is a separate
        additive interaction pass, which is how a vertex-blended terrain
        material works: two full bump/diffuse/specular sets, one weighted by
        vertexColor and one by inverseVertexColor, summed.

        Taking only the first stage of each kind built half of such a material
        and dropped the rest, so textures/rock/skysandnew_sharprock rendered
        as sand diffuse dimmed by the vertex colour with the rock's specular
        bolted on.

        vertexColor is a property of the SET, not of a stage: the engine keeps
        one inter.vertexColor and EVERY diffuse and EVERY specular stage
        overwrites it, after which RB_ARB2_DrawInteraction turns it into the
        single env[16]/env[17] pair that modulates the whole pass. A plain
        `specularmap` line following a vertex-blended pair of diffuse stages
        therefore RESETS the weighting to 1.0 for that set - see
        textures/base_wall/blend.

        Note the accumulator is NOT reset on a diffuse/specular flush, only on
        a bump, so consecutive sets share whatever bump and specular were last
        set. That is a real engine quirk with teeth: skysandnew_sharprock
        lists both of its bumpmaps before either diffusemap, so the second
        bump overwrites the first before any diffuse is seen and the sand
        normal map never gets used at all. Reproduce it rather than tidy it.
        """
        sets = []
        state = {'bump': None, 'diffuse': None, 'specular': None,
                 'vertex_color': None}

        def submit():
            # RB_SubmittInteraction bails on a null bump image and on a set
            # with neither a diffuse nor a specular. The first of those is not
            # dead code: AddImplicitStages appends its `map _flat` bump at the
            # END of the stage list, and only the engine's own
            # SortInteractionStages moves it in front of the diffuse it
            # belongs to.
            if state['bump'] is None:
                return
            if state['diffuse'] is not None or state['specular'] is not None:
                sets.append(InteractionSet(state['bump'], state['diffuse'],
                                           state['specular'],
                                           state['vertex_color']))

        for stage in self._live_stages():
            lighting = stage.lighting
            if lighting in ('bump', 'parallax'):
                submit()
                state['diffuse'] = state['specular'] = None
                state['bump'] = stage
            elif lighting == 'diffuse':
                if state['diffuse'] is not None:
                    submit()
                state['diffuse'] = stage
                state['vertex_color'] = stage.vertex_color
            elif lighting == 'specular':
                if state['specular'] is not None:
                    submit()
                state['specular'] = stage
                state['vertex_color'] = stage.vertex_color
        submit()
        return sets

    def _live_stages(self):
        """Interaction stages the accumulator actually sees.

        `if <expr>` on an interaction stage is checked BEFORE any of the flush
        logic - RB_CreateSingleDrawInteractions does
        `if ( !surfaceRegs[conditionRegister] ) break;` at the top of each
        case - so a stage whose condition is false is invisible to the
        accumulator and does not split a set.

        This matters more than it looks. Prey's crawlerweb materials are frame
        animations: seventeen diffusemap stages and seventeen specularmap
        stages, each gated on its own frame. Ignoring the conditions turns
        that into 33 interaction sets summed together - seventeen times too
        bright - instead of the one frame that is actually live.
        """
        for stage in self.b.ir.stages:
            if stage.lighting not in ('bump', 'parallax', 'diffuse',
                                      'specular'):
                continue
            if stage.condition is not None and \
                    self.b.params.fold(stage.condition, 1.0) == 0.0:
                continue
            yield stage

    def merged_set(self):
        """Every interaction stage collapsed into one pass.

        The single biggest render-time lever in the whole ladder: three passes
        against one measured 1.77x on its own, more than the Principled-versus
        -Diffuse swap that gets more attention. What is given up is the
        vertex-colour weighting BETWEEN passes, so a two-layer blended terrain
        material shows its first layer rather than the blend - diagnosed, not
        silent.
        """
        sets = self.sets()
        if not sets:
            return None
        if len(sets) > 1:
            self.b.note(DIAG_APPROXIMATED, 'interaction-passes',
                        '%d interaction passes merged into one; the vertex '
                        'colour weighting between them is not reproduced'
                        % len(sets))
        first = sets[0]
        diffuse = next((s.diffuse for s in sets if s.diffuse is not None),
                       None)
        specular = next((s.specular for s in sets if s.specular is not None),
                        None)
        bump = next((s.bump for s in sets if s.bump is not None), first.bump)
        return InteractionSet(bump, diffuse, specular, first.vertex_color)


# ===========================================================================
# END BUILDER
# ===========================================================================


# ===========================================================================
# BEGIN SURFACE STRATEGIES
#
# The ONLY polymorphic piece in this file. Four small classes that assemble
# the same primitives differently; everything they use lives in the builder,
# so a change to image wiring or texture matrices lands in all four at once.
#
# Measured cost of each rung (tests/tier_bench.py, 2026-09-01):
#   EnginePassSurface   1.00x   21.0 nodes/mat   the reference
#   MergedPassSurface   2.12x    7.0            one pass, Diffuse BSDF
#   FlatLitSurface      2.18x    7.0            + no specular texture, capped
#   UnlitDiffuseSurface 2.85x    3.0            diffuse only, still lit
# ===========================================================================


class SurfaceStrategy(object):
    """How one mode turns a material's stages into one shader socket."""

    def build(self, b):
        """The shared assembly: lit surface first, then the ambient stack."""
        lit = self.lit_surface(b)
        # Whether `below` is still nothing but the stand-in for "the
        # background shows through". A multiplicative stage can be built
        # exactly while that holds and only while it holds - see
        # AmbientCompositor._filter.
        b.compositor.surface_is_base = lit is None
        if lit is not None:
            # Translucency wraps the LIT result, before anything is composited
            # over it - not the finished surface. Applying it at the end adds
            # a second Transparent BSDF on top of the one base_surface already
            # supplies when there is no lit pass, and the background then
            # comes through twice.
            below = b.apply_translucency(lit)
        else:
            below = self.base_surface(b)
        for stage, verdict in b.plan.stages:
            if stage.lighting != 'ambient':
                continue
            layered = self.ambient_pass(b, below, stage)
            if verdict == VERDICT_GATE and layered is not below:
                # `if <expr>` under the DYNAMIC policy: both branches are
                # built and the mix follows the condition, so moving a slider
                # switches the stage on or off without a rebuild. BAKED and
                # SKIP never reach here - they resolved the condition into a
                # DROP or a BUILD before anything was planned.
                layered = b.gate_on_condition(below, layered, stage)
            below = layered
        return below

    # -- hooks --------------------------------------------------------------

    def lit_surface(self, b):
        raise NotImplementedError

    def ambient_pass(self, b, below, stage):
        """Default: sample the stage and hand it to the blend equation."""
        comp = b.compositor
        # A maskColor stage exists ONLY for its alpha, so its own blend mode
        # says nothing about whether that alpha is wanted. Reading it off
        # wants_alpha() dropped the mask on every such stage whose blend pair
        # happened to be alpha-blind - which is most of them, since a mask
        # stage usually carries no blend keyword at all.
        wants = True if stage.writes_no_color else comp.wants_alpha(stage)
        color, alpha = b.sampler.sample(stage, want_alpha=wants)
        if stage.writes_no_color:
            # It draws nothing; it only deposits a mask in the framebuffer
            # alpha for a later gl_dst_alpha stage to read back.
            comp.take_mask(stage, alpha)
            return below
        return comp.composite(below, stage, color, alpha)

    # -- shared -------------------------------------------------------------

    def base_surface(self, b):
        """What the ambient stages are drawn onto when nothing lights this.

        An opaque surface does not let the background through.
        RB_STD_FillDepthBuffer paints every drawn opaque or perforated surface
        black before the ambient pass runs, so a stage that adds to the
        framebuffer adds to black, not to the room behind the wall.
        textures/skies/desert is one additive stage on an opaque material, and
        starting from a Transparent BSDF left the world visible straight
        through the sky.

        The one opaque surface that is NOT filled black is a subview. That
        branch of RB_T_FillDepthBuffer takes gl_dst_color,gl_zero with a
        colour of 1/overBright instead, because painting a mirror black would
        erase the reflection the subview pass just rendered into it. Five
        mirrors across Doom 3 and Prey are `mirror` + a filter stage, and
        blacking their base turned all five into placeholder swatches.
        """
        if b.ir.sort_name == 'subview':
            return b.work('ShaderNodeBsdfTransparent', 'subview').outputs['BSDF']
        if b.ir.stages and b.ir.coverage != COVERAGE_TRANSLUCENT:
            black = b.work('ShaderNodeEmission', 'depth fill')
            black.inputs['Color'].default_value = (0.0, 0.0, 0.0, 1.0)
            # The destination is no longer "whatever is behind the surface",
            # so the exact framebuffer-reading blends must not assume it is.
            b.compositor.surface_is_base = False
            return black.outputs['Emission']
        return b.work('ShaderNodeBsdfTransparent', 'base').outputs['BSDF']

    def _diffuse_color(self, b, iset):
        """diffuseMap x diffuseColor, with the set's vertexColor if wanted."""
        if iset.diffuse is None:
            return None
        color, _alpha = b.sampler.sample(iset.diffuse, want_alpha=False,
                                         want_vertex_color=False)
        if b.profile.vertex_color:
            color = b.sampler.vertex_color_socket(iset.vertex_color, color)
        return color

    def _normal(self, b, iset):
        if not b.profile.normals or iset.bump is None:
            return None
        stage = iset.bump
        sampling = Sampling.for_stage(stage)
        uv = b.plan.want_uv(b.params.transform_chain(stage), stage.texgen,
                            stage.texgen_args)
        return b.images.normal(stage.image, sampling, uv.key)

    def _link_diffuse(self, b, iset, socket):
        """Wire the diffuse colour, or pin the socket black.

        A set with no diffuse binds the engine's blackImage; leaving the
        socket at its node default would draw a lit grey surface the engine
        never draws.
        """
        color = self._diffuse_color(b, iset)
        if color is not None:
            b.link(color, socket)
        else:
            socket.default_value = (0.0, 0.0, 0.0, 1.0)

    def _specular_color(self, b, iset):
        """specularMap x specularColor, with the SET's vertexColor.

        The engine keeps one inter.vertexColor per drawInteraction_t and
        RB_ARB2_DrawInteraction turns it into the env[16]/env[17] pair that
        modulates the whole pass - the specular term as much as the diffuse
        one. Weighting only the diffuse leaves the highlight at full strength
        on a surface the vertex colour is fading out.
        """
        if not b.profile.wants_specular_texture or iset.specular is None:
            return None
        color, _alpha = b.sampler.sample(iset.specular, want_alpha=False,
                                         want_vertex_color=False)
        if b.profile.vertex_color:
            color = b.sampler.vertex_color_socket(iset.vertex_color, color)
        return color


class EnginePassSurface(SurfaceStrategy):
    """MAXIMUM - one Principled BSDF per flushed drawInteraction_t, summed.

    The reference rung and the most expensive: three passes measured 1.77x
    slower than one, and the Principled closure a further ~20% over Diffuse.
    Both of those are what the mode is for.
    """

    def lit_surface(self, b):
        sets = b.interactions.sets()
        if not sets:
            return None
        shader = None
        for index, iset in enumerate(sets):
            one = self._one_pass(b, iset)
            if one is None:
                continue
            if shader is None:
                shader = one
                continue
            # Separate interaction passes are additive in the engine
            # (GLS_SRCBLEND_ONE | GLS_DSTBLEND_ONE), and the vertexColour
            # weights on each set are what keep the sum in range.
            add = b.work('ShaderNodeAddShader', 'interaction set %d'
                         % (index + 1))
            b.link(shader, add.inputs[0])
            b.link(one, add.inputs[1])
            shader = add.outputs['Shader']
        return shader

    def _one_pass(self, b, iset):
        bsdf = b.work('ShaderNodeBsdfPrincipled', 'interaction')
        bsdf.inputs['Metallic'].default_value = 0.0
        diffuse = self._diffuse_color(b, iset)
        if diffuse is not None:
            b.link(diffuse, bsdf.inputs['Base Color'])
        else:
            # A set with a specular but no diffuse binds the engine's
            # blackImage, not a default grey - RB_SubmittInteraction leaves
            # inter.diffuseImage null and the interaction program multiplies
            # by it. Left at the Principled default this pass would add a
            # whole lit grey surface that the engine never draws.
            bsdf.inputs['Base Color'].default_value = (0.0, 0.0, 0.0, 1.0)
        b.link(self._normal(b, iset), bsdf.inputs['Normal'])

        spec = self._specular_color(b, iset)
        if spec is not None:
            # A highlight exists iff the engine binds a specular map - not its
            # black image. Specular Tint carries the map's own colour, which
            # is what scales the highlight in interaction.vfp.
            group = b.work('ShaderNodeGroup', 'roughness')
            group.node_tree = get_or_create_estimate_roughness_group()
            b.link(spec, group.inputs['Specular Color'])
            b.link(group.outputs['Roughness'], bsdf.inputs['Roughness'])
            if 'Specular Tint' in bsdf.inputs:
                b.link(spec, bsdf.inputs['Specular Tint'])
        else:
            bsdf.inputs['Roughness'].default_value = 1.0
            if 'Specular IOR Level' in bsdf.inputs:
                bsdf.inputs['Specular IOR Level'].default_value = 0.0
        return bsdf.outputs['BSDF']


class MergedPassSurface(SurfaceStrategy):
    """GOOD - every interaction stage collapsed into one Diffuse BSDF.

    Measured 2.12x over Maximum, from two independent changes: one pass
    instead of N (1.77x on its own) and Diffuse instead of Principled (~20%
    more). The specular map is still loaded, but only to drive Roughness -
    there is no highlight term and no specular tint. Cube maps are dropped
    entirely, which also sheds a ~7.7MB equirect bake per material.
    """

    def lit_surface(self, b):
        iset = b.interactions.merged_set()
        if iset is None:
            return None
        bsdf = b.work('ShaderNodeBsdfDiffuse', 'interaction')
        self._link_diffuse(b, iset, bsdf.inputs['Color'])
        b.link(self._normal(b, iset), bsdf.inputs['Normal'])
        spec = self._specular_color(b, iset)
        if spec is not None:
            group = b.work('ShaderNodeGroup', 'roughness')
            group.node_tree = get_or_create_estimate_roughness_group()
            b.link(spec, group.inputs['Specular Color'])
            b.link(group.outputs['Roughness'], bsdf.inputs['Roughness'])
        return bsdf.outputs['BSDF']


class FlatLitSurface(SurfaceStrategy):
    """BASIC - Good's lit surface with no specular texture loaded at all.

    The lit surface alone measured 2.9% faster than Good, which is the noise
    floor, so this rung is not earned here: it is earned on the ambient stack,
    where the profile keeps only alpha-carrying stages and caps them at four.
    What it does buy on the lit surface is VRAM - 16,593 corpus materials ship
    a specular map, and this mode never loads one.
    """

    def lit_surface(self, b):
        iset = b.interactions.merged_set()
        if iset is None:
            return None
        bsdf = b.work('ShaderNodeBsdfDiffuse', 'interaction')
        self._link_diffuse(b, iset, bsdf.inputs['Color'])
        b.link(self._normal(b, iset), bsdf.inputs['Normal'])
        return bsdf.outputs['BSDF']


class UnlitDiffuseSurface(SurfaceStrategy):
    """SIMPLE - the diffuse texture on a lit Diffuse BSDF. Nothing else.

    Named for what it drops, not for its shading: it IS lit. An Emission
    version measured the same (0.183s against 0.187s, inside noise) and reads
    wrongly against scene lighting, which defeats the point of a mode whose
    job is to make geometry placement verifiable.

    No normals and no heightmaps, so two fewer textures resident per material
    on top of Basic's specular saving.
    """

    def lit_surface(self, b):
        iset = b.interactions.merged_set()
        if iset is None and b.ir.ambient_stages:
            # Nothing lights this surface and it has ambient stages of its
            # own, so the ambient path IS the material - a decal, a glow, a
            # sky. Inventing a lit surface here put a Diffuse BSDF underneath
            # a filter decal, which then had something to multiply and fell
            # out of its exact path into the luminance approximation.
            return None
        color = self._diffuse_color(b, iset) if iset is not None else None
        if color is None:
            color = self._standin_color(b)
        if color is None:
            return None
        bsdf = b.work('ShaderNodeBsdfDiffuse', 'diffuse')
        b.link(color, bsdf.inputs['Color'])
        return bsdf.outputs['BSDF']

    def _standin_color(self, b):
        """Simple's own fallback: the editor image, when it is preferred or
        when there is no diffuse stage to draw."""
        path = b.ir.editor_image if b.profile.prefer_editor_image else None
        if not path:
            diffuse = b.ir.stage_by_lighting('diffuse')
            if diffuse is not None and diffuse.image is not None:
                path = diffuse.image.base_path()
        path = path or b.ir.first_image_path()
        if not path:
            return None
        req = b.plan.want_image(b.assets.resolve(path, USE_DIFFUSE), 'sRGB',
                                DEFAULT_SAMPLING, (), path)
        return b.graph.image(req).outputs['Color']


STRATEGIES = {
    'EnginePassSurface': EnginePassSurface,
    'MergedPassSurface': MergedPassSurface,
    'FlatLitSurface': FlatLitSurface,
    'UnlitDiffuseSurface': UnlitDiffuseSurface,
}


def make_strategy(profile):
    return STRATEGIES[profile.surface]()


# ===========================================================================
# END SURFACE STRATEGIES
# ===========================================================================


# ===========================================================================
# BEGIN VISIBILITY GUARANTEE
#
# No material may build to an empty or non-visible result, in any mode, under
# any parameter policy - so that geometry placement can always be verified by
# looking at it. Skip and Baked can both empty a material outright, and so can
# the ambient cap; that is not a bug, it is what this exists for.
#
# Nothing degrades silently. Every fallback records WHY, and an empty report
# means every material was built exactly as its .mtr wrote it.
# ===========================================================================

# The fallback swatch: flat dark grey, mostly transparent. The brief asks for
# alpha 0.20; the constant is 0.22 and stays there, because a wall of these
# has to stay readable as scaffolding rather than fading out of the viewport.
_TRANSPARENT_BSDF = 'ShaderNodeBsdfTransparent'


def _output_node(nt):
    for node in nt.nodes:
        if node.type == 'OUTPUT_MATERIAL' and node.is_active_output:
            return node
    for node in nt.nodes:
        if node.type == 'OUTPUT_MATERIAL':
            return node
    return None


def surface_draws(nt):
    """Walk back from Material Output and answer: does this draw anything?

    A path is visible if it reaches an Emission with non-zero Strength and a
    non-black Colour, a TINTED Transparent BSDF, or any other BSDF. A material
    is invisible if there is no Material Output link at all, if the tree has
    no shader nodes, if every path terminates in an untinted Transparent, or
    if every Mix Shader Fac is pinned such that only the transparent branch
    can contribute.

    The tinted case is the one worth spelling out. `blend filter`
    (gl_dst_color, gl_zero) means result = dst * src, and a Transparent BSDF
    tinted with src is exactly that - it passes the background through
    multiplied per channel by its Colour. So a skid mark, a wall stain or a
    logo decal is CORRECTLY built as two nodes ending in a Transparent BSDF,
    and reading "ends in Transparent" as "draws nothing" threw 198 exactly
    reproduced Doom 3 materials away and redrew them as bright emissive
    quads. Only an UNTINTED Transparent - white, unlinked - is nothing.

    Reads node_tree.links once. NodeSocket.links walks the whole tree per
    socket, which is quadratic and is the accessor behind the intermittent
    'NodeLink object has no attribute to_socket' failures on large materials.
    """
    output = _output_node(nt)
    if output is None:
        return False

    incoming = {}
    for link in nt.links:
        incoming.setdefault((link.to_node, link.to_socket.identifier),
                            link.from_node)

    root = incoming.get((output, output.inputs['Surface'].identifier))
    if root is None:
        return False

    seen = set()

    def visible(node):
        if node is None or node.name in seen:
            return False
        seen.add(node.name)
        kind = node.bl_idname

        if kind == _TRANSPARENT_BSDF:
            colour = node.inputs['Color']
            if colour.is_linked:
                return True
            return any(c < 0.999 for c in colour.default_value[:3])

        if kind == 'ShaderNodeEmission':
            strength = node.inputs['Strength']
            colour = node.inputs['Color']
            lit = (strength.is_linked or strength.default_value > 0.0)
            coloured = (colour.is_linked or
                        any(c > 0.0 for c in colour.default_value[:3]))
            return bool(lit and coloured)

        if kind == 'ShaderNodeMixShader':
            fac = node.inputs['Fac']
            lower = incoming.get((node, node.inputs[1].identifier))
            upper = incoming.get((node, node.inputs[2].identifier))
            if not fac.is_linked:
                # A pinned Fac really does select one branch: at 0 only the
                # lower one can contribute, at 1 only the upper.
                if fac.default_value <= 0.0:
                    return visible(lower)
                if fac.default_value >= 1.0:
                    return visible(upper)
            return visible(lower) or visible(upper)

        if kind == 'ShaderNodeAddShader':
            return (visible(incoming.get((node, node.inputs[0].identifier))) or
                    visible(incoming.get((node, node.inputs[1].identifier))))

        if kind in ('ShaderNodeGroup', 'NodeReroute'):
            return any(visible(incoming.get((node, s.identifier)))
                       for s in node.inputs)

        # Any other shader node - Principled, Diffuse, Glossy, Volume - draws.
        return node.type.startswith('BSDF_') or 'Bsdf' in kind or \
            kind.startswith('ShaderNodeVolume') or kind == 'ShaderNodeHoldout'

    return visible(root)


def _fallback_image_path(builder):
    """The image the fallback should show, in the brief's order of preference.

    Diffuse first. That differs from the older placeholder path, which reached
    for qer_editorimage before anything else - but the diffuse stage is what
    the surface actually looks like in game, and the editor image is a stand-in
    the level editor uses precisely because it is cheaper to draw.
    """
    ir = builder.ir
    if ir is None:
        return None, None
    diffuse = ir.stage_by_lighting('diffuse')
    if diffuse is not None and diffuse.image is not None:
        path = diffuse.image.base_path()
        if path:
            return path, 'diffuse stage'
    for stage in ir.stages:
        if stage.tex_kind == TEX_FILE and stage.image is not None:
            path = stage.image.base_path()
            if path:
                return path, 'first stage image'
    if ir.editor_image:
        return ir.editor_image, 'qer_editorimage'
    return None, None


def _diagnose_emptiness(builder):
    """Say why, in the terms the .mtr would recognise."""
    ir = builder.ir
    if ir is None:
        return 'no declaration'
    if ir.failed:
        return 'the declaration is broken; the engine would MakeDefault() it'
    if not ir.stages:
        return 'no stages declared'
    if ir.is_light:
        return 'light shader, not a surface shader'
    if not builder.plan.stages:
        conditioned = sum(1 for s in ir.stages if s.condition is not None)
        if conditioned == len(ir.stages):
            return ('every stage is conditional and the %s parameter policy '
                    'does not build those' % builder.params.name.lower())
        # Say which of the several quite different reasons it was. "Dropped
        # by this mode" is actively wrong for a videoMap, which no mode
        # drops - there is simply no still frame of a video to sample.
        kinds = {s.tex_kind for s in ir.stages}
        if kinds and kinds <= {TEX_VIDEO, TEX_DYNAMIC, TEX_NONE}:
            what = 'video' if TEX_VIDEO in kinds else 'render-target'
            return ('every stage is a %s stage with no still image behind it '
                    '(no fragmentMap and no qer_editorimage to stand in)'
                    % what)
        if all(s.has_custom_program for s in ir.stages):
            return 'every stage is a screen-space program; nothing to sample'
        return 'every stage was dropped by this mode or policy'
    if ir.gui_surf:
        return 'guiSurf - the GUI itself is not reproducible'
    if all(s.writes_no_color for s, _v in builder.plan.stages):
        return 'every stage is maskColor, which writes no visible colour'
    if ir.deform:
        return 'deform %s - the geometry the engine generates is not built' \
            % (ir.deform[0] if ir.deform else '')
    if any(s.has_custom_program for s in ir.stages):
        return 'screen-space fragment program; nothing to sample'
    return 'nothing in the built graph reaches the output'


def _deliberately_hidden(builder):
    """True when this material is invisible because it is SUPPOSED to be.

    A `spectrum N` surface is lit only by a matching light, so with the scene
    Spectrum on anything else it draws nothing - that is the whole feature.
    Rescuing it with a placeholder makes the invisible writing visible, which
    is precisely backwards. 45 materials in the corpus declare a spectrum.
    """
    if not builder.ir or not builder.ir.spectrum:
        return False
    scene_spectrum = int(getattr(builder.settings, 'spectrum', 0)
                         if builder.settings is not None else 0)
    return scene_spectrum != builder.ir.spectrum


def ensure_visible(builder):
    """Return the fallback reason, or None if the material already draws.

    Replaces, rather than mixes into, whatever was built: the brief is that
    the rest of the node structure can go except as needed to control the
    editor image's transparency. So a fallback material is exactly a texture,
    an Emission and a Mix Shader against Transparent - and the mix is named
    IDTECH4_PlaceholderMix so the Editor Textures opacity slider keeps working
    on it.
    """
    if surface_draws(builder.nt):
        return None
    if _deliberately_hidden(builder):
        return None

    reason = _diagnose_emptiness(builder)
    path, source = _fallback_image_path(builder)

    builder.nt.nodes.clear()
    builder.static_expressions = []
    output = builder.new('ShaderNodeOutputMaterial', '', COL_OUTPUT, 0)

    if path:
        req = builder.plan.want_image(builder.assets.resolve(path,
                                                             USE_DIFFUSE),
                                      'sRGB', DEFAULT_SAMPLING, (), path)
        tex = builder.graph.image(req)
        colour = tex.outputs['Color']
        alpha = tex.outputs['Alpha']
        builder.note(DIAG_UNSUPPORTED, 'fallback',
                     '%s; drawn with its %s (%s) instead'
                     % (reason, source, path))
    else:
        tex = builder.new('ShaderNodeTexImage', 'placeholder', COL_IMAGE, 0)
        tex.image = builder.assets.placeholder()
        colour = tex.outputs['Color']
        alpha = tex.outputs['Alpha']
        builder.note(DIAG_UNSUPPORTED, 'fallback',
                     '%s; drawn with the placeholder swatch' % reason)

    emission = builder.new('ShaderNodeEmission', 'fallback', COL_WORK, 0)
    emission.inputs['Strength'].default_value = 1.0
    builder.link(colour, emission.inputs['Color'])

    transparent = builder.new(_TRANSPARENT_BSDF, 'fallback',
                              COL_WORK, -200)
    mix = builder.new('ShaderNodeMixShader', 'fallback', COL_WORK + 400, 0)
    mix.name = PLACEHOLDER_MIX_NAME
    builder.link(alpha, mix.inputs['Fac'])
    builder.link(transparent.outputs['BSDF'], mix.inputs[1])
    builder.link(emission.outputs['Emission'], mix.inputs[2])
    builder.link(mix.outputs['Shader'], output.inputs['Surface'])

    _set_render_method(builder.mat, 'BLENDED')
    try:
        builder.mat.blend_method = 'BLEND'
    except (TypeError, AttributeError):
        pass
    return reason


# ===========================================================================
# END VISIBILITY GUARANTEE
# ===========================================================================


def _topology_signature(mat):
    """
    Return a stable hash describing a material's node-graph *structure*,
    ignoring node names/positions/labels and the specific image/table data
    referenced. Two materials with the same signature should compile to
    the same (or a very similar) shader variant.
    """
    if not mat.use_nodes or not mat.node_tree:
        return "NO_NODES"

    nt = mat.node_tree
    nodes = sorted(nt.nodes, key=lambda n: n.bl_idname)
    node_sig = [n.bl_idname for n in nodes]

    # Node settings that change codegen even when connectivity is identical.
    for n in nodes:
        if n.bl_idname == 'ShaderNodeMath':
            node_sig.append(f"math:{n.operation}")
        elif n.bl_idname == 'ShaderNodeVectorMath':
            node_sig.append(f"vecmath:{n.operation}")
        elif n.bl_idname == 'ShaderNodeMixShader':
            linked_fac = n.inputs['Fac'].is_linked
            node_sig.append(f"mixshader:fac_linked={linked_fac}")
        elif n.bl_idname == 'ShaderNodeMix':
            node_sig.append(f"mix:{n.data_type}:{getattr(n, 'blend_type', '')}")
        elif n.bl_idname == 'ShaderNodeSeparateColor':
            node_sig.append(f"sepcolor:{n.mode}")
        elif n.bl_idname == 'ShaderNodeCombineColor':
            node_sig.append(f"combcolor:{n.mode}")
        elif n.bl_idname == 'ShaderNodeBsdfDiffuse':
            linked = sorted(s.identifier for s in n.inputs if s.is_linked)
            node_sig.append(f"diffuse_linked:{linked}")
        elif n.bl_idname == 'ShaderNodeBsdfPrincipled':
            linked = sorted(s.identifier for s in n.inputs if s.is_linked)
            node_sig.append(f"principled_linked:{linked}")
        elif n.bl_idname == 'ShaderNodeGroup' and n.node_tree:
            node_sig.append(f"group:{n.node_tree.name}")

    link_sig = sorted(
        f"{l.from_node.bl_idname}.{l.from_socket.identifier}"
        f"->{l.to_node.bl_idname}.{l.to_socket.identifier}"
        for l in nt.links
    )

    mat_sig = [
        getattr(mat, 'blend_method', None),
        getattr(mat, 'surface_render_method', None),
        getattr(mat, 'use_backface_culling', None),
    ]

    raw = repr((sorted(node_sig), link_sig, mat_sig))
    return hashlib.md5(raw.encode()).hexdigest()


def count_material_topologies(materials):
    """
    Group the given materials by topology signature.
    Returns a list of (signature, [material_names]) sorted by group size
    descending (largest/most-shared topology first).
    """
    groups = {}
    for mat in materials:
        sig = _topology_signature(mat)
        groups.setdefault(sig, []).append(mat.name)
    return sorted(groups.items(), key=lambda kv: -len(kv[1]))


# ─────────────────────────────────────────────────────────────────────
#  SHARED idTech4 SOURCES (Base Directory / Materials Source). Duplicated
#  identically across every idTech4 Blender addon (map import, .ase/.lwo
#  import, MD5 tools, materials) so Base Directory/Materials Source work
#  no matter which subset of these addons happens to be installed — they
#  used to live only in this addon's own AddonPreferences, so nothing
#  could read OR save them unless this specific addon was enabled.
#  Backed here by a small JSON file under Blender's per-user config
#  directory instead, shared by plain file path rather than by any
#  addon's registration state. The panel and its two operators are
#  registered at most once regardless of how many of these addons are
#  enabled at once — see _register_shared_ui.
# ─────────────────────────────────────────────────────────────────────


# ===========================================================================
# BEGIN REPORT
#
# The contract is that nothing degrades silently. Every departure from "we
# understood this and built it properly" produces a diagnostic, and this is
# where they are grouped for the panel and for an external importer's log.
# An empty report means every material was built exactly as its .mtr wrote it.
# ===========================================================================

IMPLICIT_DIAG_KIND = 'implicit image material'

_REPORT_HEADINGS = [
    (DIAG_PARSE_ERROR, 'Failed to parse'),
    (DIAG_UNKNOWN_KW, 'Unknown keywords'),
    (DIAG_UNSUPPORTED, 'Not reproducible in Blender'),
    (DIAG_APPROXIMATED, 'Approximated'),
    (DIAG_CONTENT, 'Problems in the .mtr itself'),
]


def summarise_diagnostics(diagnostics, per_kind=3):
    """Group diagnostics into report lines, most severe first.

    per_kind caps how many individual entries are named under each kind; pass
    None to name every one of them. The cap exists for the material panel's
    report, which is a fixed-height UIList - a report that ends up in an
    editable, saveable, copyable text block wants the whole list instead.

    That same uncapped form also spells out each entry's message, which is
    where a diagnostic says WHY - which stage was dropped and what dropped
    it, rather than only that something under this kind happened somewhere in
    this material. The panel's capped form still omits it: a UIList row is
    one fixed-width line and a sentence would be cut off mid-word.
    """
    buckets = {}
    for diag in diagnostics:
        buckets.setdefault(diag.level, {}).setdefault(diag.kind,
                                                      []).append(diag)

    lines = []
    for level, heading in _REPORT_HEADINGS:
        kinds = buckets.get(level)
        if not kinds:
            continue
        total = sum(len(v) for v in kinds.values())
        lines.append('%s (%d)' % (heading, total))
        for kind in sorted(kinds, key=lambda k: (-len(kinds[k]), k)):
            entries = kinds[kind]
            lines.append('    %-28s %4d' % (kind, len(entries)))
            shown = entries if per_kind is None else entries[:per_kind]
            for diag in shown:
                # " : " between the parts, and none of them printed when it
                # is empty - a diagnostic about a name in use has no source
                # position, and one about a whole file has no material. The
                # message comes last and only in the uncapped form; see the
                # docstring.
                parts = [diag.location(), diag.material]
                if per_kind is None:
                    parts.append(diag.message)
                lines.append('        %s' % ' : '.join(
                    part for part in parts if part))
            if per_kind is not None and len(entries) > per_kind:
                lines.append('        ... and %d more'
                             % (len(entries) - per_kind))
    return lines


class BuildStatus(object):
    """What happened to one material."""

    __slots__ = ('name', 'material', 'ok', 'mode', 'params', 'node_count',
                 'diagnostics', 'fallback_reason', 'driver_count')

    def __init__(self, name, material=None, ok=False, mode='', params='',
                 node_count=0, diagnostics=(), fallback_reason=None,
                 driver_count=0):
        self.name = name
        self.material = material
        self.ok = ok
        self.mode = mode
        self.params = params
        self.node_count = node_count
        self.diagnostics = list(diagnostics)
        self.fallback_reason = fallback_reason
        self.driver_count = driver_count

    @property
    def status(self):
        if not self.ok:
            return 'failed'
        if self.fallback_reason:
            return 'fallback: %s' % self.fallback_reason
        return 'built'


class BuildSummary(object):
    """The accounting for one material-build pass.

    Two separate diagnostic piles, on purpose. source_diagnostics is
    everything the parser said about the WHOLE .mtr tree - 438 of them for
    Quake 4's 5,138 materials. diagnostics is only what the materials actually
    built produced.

    The parse is necessarily global: name resolution needs an index of every
    decl, later files override earlier ones, tables and guides are scattered,
    and `inherit` is prefix-matched, so there is no way to read one material
    without scanning the tree. The REPORT is not global, and that distinction
    is what scoped_source_diagnostics() exists to hold. Building five scene
    materials against Quake 4 used to produce a 481-line report of which 471
    lines were about 380 materials the user had not asked for and could not
    act on - the parse scope leaking into a build-scoped report.

    A material the pass did not touch is therefore not reported on at all. The
    single exception is a parse error that names no material: that means a file
    failed structurally rather than a decl inside it, which can silently drop
    the very decls the pass went looking for, so it is reported however narrow
    the scope. Every diagnostic the shipping trees actually produce carries a
    material name, so in practice the exception costs nothing and exists only
    so a genuinely broken tree cannot fail quietly.
    """

    def __init__(self, db=None):
        self.db = db
        self.source_diagnostics = list(db.diagnostics) if db is not None else []
        self.diagnostics = []
        self.built = 0
        self.skipped = 0
        self.failed = 0
        self.fallbacks = 0
        self.not_found = []
        self.implicit = []
        self.errors = []
        self.log = []
        # Lowercased decl names this pass actually looked at, which is what
        # scopes the source diagnostics. Filled from the resolved decl rather
        # than the requested name so an alias, a `.tga` suffix or a case
        # difference between the mesh's slot and the .mtr still matches.
        self.touched = set()

    def note_touched(self, *names):
        for name in names:
            if name:
                self.touched.add(engine_canonical_decl(str(name)))

    def scoped_source_diagnostics(self):
        """The parser's notes, cut down to the materials this pass touched."""
        scoped = []
        for diag in self.source_diagnostics:
            material = engine_canonical_decl(diag.material or '')
            if material:
                if material in self.touched:
                    scoped.append(diag)
            elif diag.level == DIAG_PARSE_ERROR:
                # A structural failure with no decl to pin it on - the file
                # itself did not load, so a name this pass wanted may be
                # missing for that reason and nothing else would say so.
                scoped.append(diag)
        return scoped

    def record(self, status):
        self.diagnostics.extend(status.diagnostics)
        if not status.ok:
            self.failed += 1
        else:
            self.built += 1
        if status.fallback_reason:
            self.fallbacks += 1
        self.log.append('%-52s %s' % (status.name, status.status))

    def record_skipped(self, mat_name, mat_ir, reason):
        self.skipped += 1
        self.diagnostics.append(MtrDiagnostic(
            DIAG_UNSUPPORTED, 'light material', reason,
            mat_ir.filename, mat_ir.line, mat_ir.name))

    def record_implicit(self, mat_name, image_path, mode):
        """A name with no .mtr declaration that resolved to an image on disk.

        Not a failure - the engine generates a material from the name itself,
        so the surface works and always did. Still worth its own line, because
        "no decl anywhere" is just as often a materials tree that did not get
        loaded as it is a deliberate bare-image reference, and the two look
        identical in the viewport once both build.
        """
        self.implicit.append(mat_name)
        if mode == MODE_MAXIMUM:
            # Maximum builds the engine's literal generated decl, so there is
            # nothing approximate about it; the note is about the assets.
            level, detail = DIAG_CONTENT, (
                'not declared in any .mtr; built from %s the way the engine '
                'generates a material from the name'
                % os.path.basename(image_path))
        else:
            level, detail = DIAG_APPROXIMATED, (
                'not declared in any .mtr; built from %s as a lit diffuse '
                'surface - the engine would generate an unlit clamped blend '
                'stage instead (Maximum mode does)'
                % os.path.basename(image_path))
        self.diagnostics.append(MtrDiagnostic(
            level, IMPLICIT_DIAG_KIND, detail, '', 0, mat_name))

    def record_not_found(self, mat_name, searched=None):
        """A name in use by geometry that resolves to nothing at all.

        No .mtr declaration AND no image of that name on disk - if either
        existed the name would have been built by now. Worth reporting rather
        than passing over: the surface keeps whatever placeholder it already
        had, which renders as flat white with no nodes and gives no clue why.

        `searched` is every path that was actually tried. This one is a
        MISSING ASSET rather than a fault inside a file, so the useful thing
        to print is where it was looked for: with a Mod Base configured that
        is the only way to tell "neither tree ships it" apart from "one of
        the two roots is pointing somewhere wrong".
        """
        self.not_found.append(mat_name)
        detail = ('in use by geometry, not declared in any .mtr, and no image '
                  'of that name on disk - the surface keeps its blank '
                  'placeholder')
        if searched:
            detail += '; searched:\n        ' + '\n        '.join(searched)
        self.diagnostics.append(MtrDiagnostic(
            DIAG_UNSUPPORTED, 'material not declared', detail,
            '', 0, mat_name))

    def record_error(self, mat_name, exc):
        self.errors.append('%s: %s' % (mat_name, exc))
        self.diagnostics.append(MtrDiagnostic(
            DIAG_PARSE_ERROR, 'builder',
            'exception while building: %s' % exc, '', 0, mat_name))

    def publish(self, context):
        """Populate the material panel's report list, as the button does.

        Scoped, like every other view of this pass. The panel's list has no
        room to say which pile a row came from, so an unscoped source
        diagnostic here was indistinguishable from something wrong with a
        material the user had just asked for.
        """
        settings = getattr(context.scene, 'idtech4_settings', None) \
            if context else None
        if settings is None or self.db is None:
            return
        _store_report(settings, self.db,
                      self.scoped_source_diagnostics() + self.diagnostics,
                      self.built, self.skipped, self.failed)

    def lines(self):
        """Human-readable summary, for an external importer's own report.

        Everything here is scoped to the materials THIS pass touched. What the
        .mtr tree as a whole contains belongs to the material panel's own
        report, not to a .map import: a map that builds forty materials
        against Doom 3's 6,267 learns nothing from being told the other 6,227
        exist.

        Nothing is truncated either. The caller writes this into a text
        datablock the person can scroll, edit, copy and save, so a name
        omitted behind "... and 30 more" is a name they cannot act on.

        Implicit image materials appear nowhere in here, neither as a count
        nor as a list: a name with no .mtr decl is ordinary engine behaviour,
        the surface works, and there is nothing for anyone to do about it,
        which makes a paragraph of report about it purely noise. They are
        still counted in `built`, and self.implicit still holds the names.
        """
        out = []
        roots = list(getattr(self.db, 'roots', ()) or ())
        if len(roots) > 1:
            # Stated once, up front, instead of a diagnostic per overridden
            # material: with two roots configured, "which tree did this come
            # from" is the question behind almost every surprising result
            # here, and the answer is the same for all of them.
            out.append('Search order (first match wins, per file):')
            for i, root in enumerate(roots):
                out.append('    %d. %s' % (i + 1, root))
            overrides = getattr(self.db, 'mod_overrides', 0)
            if overrides:
                out.append('    %d material(s) declared in more than one of '
                           'these; the higher entry was used.' % overrides)
        out.append('%d built, %d skipped, %d failed to parse'
                   % (self.built, self.skipped, self.failed))
        if self.fallbacks:
            out.append('%d material(s) could not be drawn as declared and '
                       'fell back to an image or the placeholder swatch'
                       % self.fallbacks)
        if self.not_found:
            out.append('%d name(s) in use with no .mtr declaration and '
                       'no image on disk:' % len(self.not_found))
            for name in self.not_found:
                out.append('    %s' % name)
        if self.errors:
            out.append('%d build error(s):' % len(self.errors))
            for line in self.errors:
                out.append('    %s' % line)
        reportable = [d for d in self.diagnostics
                      if d.kind != IMPLICIT_DIAG_KIND]
        out.extend(summarise_diagnostics(reportable, per_kind=None))
        return out


REPORT_TEXT_NAME = 'idTech4 Material Report'


def write_report_text(summary, name=REPORT_TEXT_NAME):
    """Write the full report into a bpy.data.texts datablock and return it.

    The panel's report list is a fixed-height UIList that shows at most three
    entries per diagnostic kind, which is the right shape for a glance and the
    wrong one for acting on anything: a name behind "... and 47 more" is a
    name nobody can do anything about. This leaves the untruncated version in
    a Text datablock, reopenable from any Text Editor long after the import,
    exactly as the .map importer already does for its own report.
    """
    text = bpy.data.texts.get(name)
    if text is None:
        text = bpy.data.texts.new(name)
    text.clear()
    lines = [name, '=' * len(name), '']
    if summary.db is not None:
        lines.append('source: %d materials, %d tables, %d guides in %d files'
                     % (len(summary.db.materials), len(summary.db.tables),
                        len(summary.db.guides), len(summary.db.files)))
        lines.append('')
    lines.extend(summary.lines())
    scoped = summary.scoped_source_diagnostics()
    if scoped:
        lines.append('')
        lines.append('-- what the parser said about the materials built '
                     'above --')
        lines.extend(summarise_diagnostics(scoped, per_kind=None))
    withheld = len(summary.source_diagnostics) - len(scoped)
    if withheld > 0:
        lines.append('')
        lines.append('(%d further parser note%s about materials this run did '
                     'not build %s not listed)'
                     % (withheld, '' if withheld == 1 else 's',
                        'is' if withheld == 1 else 'are'))
    text.write('\n'.join(lines) + '\n')
    return text


def _store_report(settings, db, diagnostics, built, skipped, failed):
    """Fill the panel's report list and the Created Tables list."""
    settings.report_entries.clear()
    for line in summarise_diagnostics(diagnostics):
        item = settings.report_entries.add()
        item.text = line
    settings.report_summary = (
        '%d built, %d skipped, %d failed to parse - %d materials, %d tables, '
        '%d guides in %d files'
        % (built, skipped, failed, len(db.materials), len(db.tables),
           len(db.guides), len(db.files)))

    # The tables actually referenced by a driver, rather than every table in
    # the tree - both for the Created Tables list and for the live registry
    # those drivers resolve through, so the two can never disagree about which
    # tables this scene uses. This list is also what _rehydrate_table_registry
    # reads back after a file load; it is the only copy that survives.
    used = _tables_referenced_by_expressions()
    _prune_table_registry(used)
    settings.created_tables.clear()
    for name in sorted(used):
        table = db.tables.get(name)
        if table is None:
            continue
        item = settings.created_tables.add()
        item.table_name = name
        item.entry_count = len(table.values)
        item.entries_json = json.dumps([round(v, 6) for v in table.values])
        item.is_clamp = bool(table.clamp)
        item.is_snap = bool(table.snap)
    settings.created_tables_index = 0


# ---------------------------------------------------------------------------
# Implicit image materials
# ---------------------------------------------------------------------------
# A material name with no .mtr declaration is not automatically a mistake. The
# engine makes one up from the name itself: idDeclLocal::ParseLocal() calls
# SetDefaultText() before it gives up, and for materials that is
# idMaterial::SetDefaultText() (renderer/Material.cpp), which returns the decl
# name used verbatim as a texture path. mars_city1 leans on this for five
# surfaces that exist only as .tga files.
#
# One deliberate divergence. id commented the existence test out -
# `if ( 1 ) { //fileSystem->ReadFile( GetName(), NULL ) != -1 )` - so the
# engine generates the implicit material for ANY undeclared name and lets the
# image loader fall back to the _default checkerboard. We put that test back.
# A name with no image behind it is a real authoring problem the report should
# name, not paper over with a blank texture.

_IMPLICIT_ENGINE_TEXT = (
    'material %s // IMPLICITLY GENERATED\n'
    '{\n{\nblend blend\ncolored\nmap "%s"\nclamp\n}\n}\n')

# The lower modes read the same image as an ordinary lit diffuse surface. The
# decl above is unlit, alpha-blended and UV-clamped, which is right for a
# flashlight beam but leaves a table cart rendering fullbright and clamped
# beside its properly declared neighbours. Maximum keeps the engine's literal
# answer because reproducing the renderer is the point of that mode.
#
# Written with the material-level `diffusemap` shortcut rather than a stage
# block, because that is the form idTech4 defines it in: inside a stage the
# equivalent is `blend diffusemap` + `map`, and a bare `diffusemap` there is
# not a stage keyword at all - it parses, and silently yields a stage with no
# image, which is exactly what the first version of this did.
_IMPLICIT_DIFFUSE_TEXT = (
    'material %s // IMPLICITLY GENERATED\n'
    '{\ndiffusemap "%s"\n}\n')


def find_implicit_image_source(base_dir, name, mod_dir=''):
    """The on-disk image an undeclared material name refers to, or None.

    Probes the name as given and with a texture extension stripped, because
    model formats routinely carry an extension the decl side does not. Plain
    source images first - _plain_source_path covers R_LoadImage's own search,
    where an extensionless name means .tga falling back to .jpg - then
    base/dds/<name>.dds, then the compiled generated/images cache. An install
    can legitimately ship any one of the three and none of the others, so a
    miss on the first two is not evidence the image is absent.

    Detection only: this never feeds a texture node. The synthesised decl
    carries the bare name and the builder resolves it like any other map.
    """
    if not name:
        return None
    roots_abs = _abs_roots(base_dir, mod_dir)
    if not roots_abs:
        return None
    candidates = [name]
    stripped = strip_material_extension(name)
    if stripped and stripped != name:
        candidates.append(stripped)
    for rel in candidates:
        rel = rel.replace('\\', '/')
        found = _plain_source_path(roots_abs, rel)
        if found:
            return found
        # base/dds/ is per-root like everything else: a mod ships its own
        # dds/ tree or none at all. Shares _dds_cache_path with
        # resolve_image_path so detection and resolution cannot disagree -
        # they used to, and the version that mattered was the one without
        # the rule.
        found = _dds_cache_path(roots_abs, rel)
        if found:
            return found
        found = _find_bimage_fallback(roots_abs, rel, None)
        if found:
            return found
    return None


def build_implicit_material(name, mode=MODE_GOOD, diags=None):
    """Parse the decl the engine would have generated for `name`.

    Goes through the ordinary lexer, parser and finish_material() rather than
    hand-assembling an MtrMaterial, so an implicit material picks up the same
    implicit stages, stage sort, classification and coverage as a declared one
    and needs no special case anywhere in the builder.
    """
    text = (_IMPLICIT_ENGINE_TEXT if mode == MODE_MAXIMUM
            else _IMPLICIT_DIFFUSE_TEXT) % (name, name)
    local = []
    toks = _lex(text)
    for decl in _scan_decls(toks, '<implicitly generated>', local):
        if decl.type != 'material':
            continue
        mat = _parse_material(toks, decl, {}, local)
        finish_material(mat, local)
        mat.diagnostics = list(local)
        if diags is not None:
            diags.extend(local)
        return mat
    return None


# ===========================================================================
# END REPORT
# ===========================================================================


# ===========================================================================
# BEGIN PUBLIC API
#
# The documented surface the three sibling addons use. Everything else in this
# file is private and may be renamed without notice; nothing named here may
# be, without bumping API_VERSION.
#
# API_VERSION 2 replaces the v1.3.2 arrangement, where the siblings reached
# into _strip_mat_ext, _load_tables_into_cache, _TABLE_CACHE, _LAST_DATABASE,
# _refresh_editor_texture_list, MaterialBuildSummary, iter_build_materials and
# build_material_qer/simple/standard.
# ===========================================================================

API_VERSION = 2

# Parsed databases are cached per (source path, base dir). Parsing a whole
# game base is ~0.5s and every caller wants the same one.
_DATABASE_CACHE = {}
_KEYWORD_SCAN_CACHE = {}


# ---- discovery ------------------------------------------------------------

def _cache_key_part(value):
    """One cache-key component for a source/root that may be a sequence."""
    if not value:
        return ''
    if isinstance(value, str):
        return os.path.normpath(value)
    return tuple(os.path.normpath(v) for v in value)


def load_database(source=None, base_dir=None, mod_dir=None):
    """Parse (or return the cached) MtrDatabase for a materials source.

    Any argument omitted falls back to the shared config, so a caller that
    just wants "whatever the user configured" can pass nothing. mod_dir is
    the optional Mod Base searched ahead of base_dir; pass '' for a build
    that deliberately has no mod root.
    """
    shared_base, shared_mod, shared_source = resolve_shared_paths()
    if source is None:
        source = shared_source
    if base_dir is None:
        base_dir = shared_base
    if mod_dir is None:
        mod_dir = shared_mod
    if not source:
        return None
    # The mod root is part of the key, not just the base: it changes which
    # images every material in the parsed tree resolves to, so a database
    # parsed without one must not be handed back to a build that has one.
    key = (_cache_key_part(source), _cache_key_part(base_dir),
           _cache_key_part(mod_dir))
    db = _DATABASE_CACHE.get(key)
    if db is None:
        db = load_mtr_database(source, base_dir or None, mod_dir or '')
        _DATABASE_CACHE[key] = db
        _load_tables_into_registry(db)
    return db


def clear_database_cache():
    """Drop parsed databases and keyword scans; call when a source changes."""
    _DATABASE_CACHE.clear()
    _KEYWORD_SCAN_CACHE.clear()


def material_names(source=None, base_dir=None, mod_dir=None):
    """Every material name the source declares, canonical (lowercase)."""
    db = load_database(source, base_dir, mod_dir)
    return sorted(db.materials) if db else []


# ---- querying, BEFORE anything is built -----------------------------------
#
# MtrMaterial.raw records every keyword the decl carried, so these answer from
# the parse and never build a node. That is the point: idTech4_ase_lwo_io.py
# needs to know which materials declare `renderbump` or `unsmoothedTangents`
# before it decides how to shade a mesh, and building 6,267 materials to find
# out would be absurd.

def _material_raw(name, source=None, base_dir=None, mod_dir=None):
    db = load_database(source, base_dir, mod_dir)
    if db is None:
        return None
    mat = db.find(name)
    return mat


def material_keywords(name, source=None, base_dir=None, mod_dir=None):
    """Every keyword one material's declaration carries, or None.

    Values are whatever the keyword took: a string for `renderbump`, True for
    a valueless flag like `twosided`, a number for `spectrum`.
    """
    mat = _material_raw(name, source, base_dir, mod_dir)
    if mat is None:
        return None
    out = dict(mat.raw)
    if mat.renderbump:
        out['renderbump'] = mat.renderbump
    for flag in mat.flags:
        out.setdefault(flag, True)
    if mat.editor_image:
        out['qer_editorimage'] = mat.editor_image
    if mat.spectrum:
        out['spectrum'] = mat.spectrum
    if mat.cull != 'front':
        out['cull'] = mat.cull
    return out


def material_has_keyword(name, keyword, source=None, base_dir=None,
                         mod_dir=None):
    kws = material_keywords(name, source, base_dir, mod_dir)
    return bool(kws) and keyword.lower() in kws


def material_keyword_value(name, keyword, source=None, base_dir=None,
                           mod_dir=None):
    kws = material_keywords(name, source, base_dir, mod_dir)
    return kws.get(keyword.lower()) if kws else None


def scan_keywords(keywords, source=None, base_dir=None, mod_dir=None):
    """{canonical name: {keyword: value}} for every material declaring any.

    ONE pass over the whole tree, cached per (source, keyword set). This is
    the seam idTech4_ase_lwo_io.py's _mtr_shading_flags_via_parser sits on.
    """
    wanted = frozenset(k.lower() for k in keywords)
    db = load_database(source, base_dir, mod_dir)
    if db is None:
        return {}
    cache_key = (id(db), wanted)
    hit = _KEYWORD_SCAN_CACHE.get(cache_key)
    if hit is not None:
        return hit
    out = {}
    for name, mat in db.materials.items():
        found = {}
        for keyword in wanted:
            if keyword == 'renderbump':
                if mat.renderbump:
                    found['renderbump'] = mat.renderbump
                continue
            if keyword in mat.flags:
                found[keyword] = True
                continue
            value = mat.raw.get(keyword)
            if value is not None:
                found[keyword] = value
        if found:
            out[name] = found
    _KEYWORD_SCAN_CACHE[cache_key] = out
    return out


# ---- building -------------------------------------------------------------

def material_targets(objects):
    """{canonical name: [Material datablocks]} for everything on *objects*.

    Two datablocks can share one decl. A model's surface name is whatever the
    file wrote, and engine_canonical_decl is what the engine turns it into, so
    `textures/hell/wood1`, `textures/hell/wood1.tga` and
    `textures\\hell\\Wood1` are all one material to the engine and three
    datablocks in Blender. Building by NAME alone then fills a fourth
    datablock named `textures/hell/wood1` and leaves the ones the mesh
    actually points at as blank placeholders - which renders flat white and
    looks, correctly, like a broken duplicate.

    The datablocks themselves are never renamed. The key is the identity, the
    datablock keeps whatever the mesh calls it, and build_material fills each
    one in place through `material=`, so meshes keep their slots.

    Scans BOTH the object's slots and the mesh's own material list, because a
    .skin swap sets slot.link = 'OBJECT' and points the SLOT at the
    replacement while deliberately leaving the shared mesh alone. The two sets
    are genuinely disjoint once a skin is involved, and the union is the set
    actually in use.
    """
    targets = {}
    for obj in objects or ():
        if getattr(obj, 'type', None) != 'MESH':
            continue
        found = [slot.material for slot in obj.material_slots if slot.material]
        data = getattr(obj, 'data', None)
        found.extend(mat for mat in getattr(data, 'materials', ()) or ()
                     if mat is not None)
        for mat in found:
            bucket = targets.setdefault(engine_canonical_decl(mat.name), [])
            if mat not in bucket:
                bucket.append(mat)
    return targets


def _resolve_options(context, mode, params, base_dir, source, mod_dir=None):
    """Fill in whatever the caller did not pass from the scene settings.

    mod_dir is the optional Mod Base searched ahead of base_dir. Like the
    others, None means "take the configured one"; pass '' to say this build
    deliberately has no mod root - which is what the sibling importers do
    when their Base came from a gate override or from the imported file's
    own location rather than from the shared config.
    """
    settings = getattr(context.scene, 'idtech4_settings', None) \
        if context is not None else None
    shared_base, shared_mod, shared_source = resolve_shared_paths()
    if mode is None:
        mode = getattr(settings, 'generation_mode', MODE_GOOD) if settings \
            else MODE_GOOD
    if params is None:
        params = getattr(settings, 'parameter_policy', PARAMS_BAKED) \
            if settings else PARAMS_BAKED
    if base_dir is None:
        base_dir = shared_base
    if source is None:
        source = shared_source
    if mod_dir is None:
        mod_dir = shared_mod
    cap = getattr(settings, 'ambient_cap_override', -1) if settings else -1
    prefer_editor = getattr(settings, 'prefer_editor_image', None) \
        if settings else None
    profile = get_profile(mode,
                          ambient_cap=None if cap < 0 else cap,
                          prefer_editor_image=prefer_editor)
    return settings, profile, params, base_dir, source, mod_dir


def _make_policy_for(settings, params_name, db, context):
    scene = getattr(context, 'scene', None) if context is not None else None
    if scene is None:
        scene = getattr(bpy.context, 'scene', None)
    fps = 24.0
    frame = 0
    if scene is not None:
        try:
            fps = scene.render.fps / max(1.0, scene.render.fps_base)
        except (AttributeError, ZeroDivisionError):
            fps = 24.0
        frame = scene.frame_current
    parms = list(settings.shader_parms) if settings is not None \
        else list(DEFAULT_PARMS)
    globals_ = list(settings.global_parms) if settings is not None \
        else [0.0] * 8
    ctx = MtrEvalContext(
        tables=db.tables if db is not None else {},
        time=(frame / fps) if fps else 0.0,
        parms=parms, globals_=globals_,
        sound=getattr(settings, 'sound_amplitude', 0.0) if settings else 0.0)
    return make_policy(params_name, ctx, fps)


def build_material(name, mode=None, params=None, context=None, base_dir=None,
                   source=None, summary=None, material=None, mod_dir=None):
    """Build one material and return its BuildStatus.

    `material` lets a caller hand in the datablock to fill - which is what the
    model importers do, because the mesh already carries a slot pointing at it.
    """
    settings, profile, params_name, base_dir, source, mod_dir = _resolve_options(
        context, mode, params, base_dir, source, mod_dir)
    db = load_database(source, base_dir, mod_dir=mod_dir)
    canonical = engine_canonical_decl(name)
    mat_ir = db.find(name) if db is not None else None

    if summary is not None:
        # Both spellings: the name the caller asked for and the decl it
        # resolved to. This is what scopes the report - anything the pass
        # never asked about is not reported on.
        summary.note_touched(name, canonical,
                             mat_ir.name if mat_ir is not None else None)

    if mat_ir is None:
        # Not a failure by itself: idTech4 generates a material from any
        # undeclared name by treating it as a texture path.
        image = find_implicit_image_source(base_dir, name, mod_dir=mod_dir)
        if image is not None:
            try:
                mat_ir = build_implicit_material(canonical, profile.name)
            except Exception:                               # noqa: BLE001
                mat_ir = None
            if mat_ir is not None and not mat_ir.failed and summary is not None:
                summary.record_implicit(canonical, image, profile.name)
    if mat_ir is None:
        if summary is not None:
            # Every path find_implicit_image_source would have tried, so the
            # report says where it looked rather than only that it failed.
            summary.record_not_found(
                canonical,
                image_search_paths(base_dir, canonical, mod_dir=mod_dir))
        return BuildStatus(canonical, ok=False, mode=profile.name,
                           params=params_name)

    if mat_ir.is_light and settings is not None and \
            getattr(settings, 'skip_light_materials', False):
        if summary is not None:
            summary.record_skipped(
                canonical, mat_ir,
                'light shader skipped - Skip Light Materials is on')
        return BuildStatus(canonical, ok=True, mode=profile.name,
                           params=params_name)

    policy = _make_policy_for(settings, params_name, db, context)
    assets = AssetResolver(base_dir, mod_dir)
    blender_mat = material if material is not None else \
        (bpy.data.materials.get(canonical) or
         bpy.data.materials.new(canonical))

    try:
        builder = MaterialBuilder(mat_ir, blender_mat, profile, policy, assets,
                                  settings).build()
    except Exception as exc:                                # noqa: BLE001
        if summary is not None:
            summary.record_error(canonical, exc)
        return BuildStatus(canonical, material=blender_mat, ok=False,
                           mode=profile.name, params=params_name)

    status = BuildStatus(canonical, material=blender_mat, ok=not mat_ir.failed,
                         mode=profile.name, params=params_name,
                         node_count=len(builder.nt.nodes),
                         diagnostics=builder.notes,
                         fallback_reason=builder.fallback_reason,
                         driver_count=builder.driver_count)
    if summary is not None:
        summary.record(status)
    return status


def build_materials(names, mode=None, params=None, context=None,
                    base_dir=None, source=None, progress=False, summary=None,
                    targets=None, mod_dir=None):
    """Build many materials.

    progress=False returns a BuildSummary. progress=True returns a generator
    yielding (done, total, name) and holding the summary on `.summary`, for a
    caller inside a modal operator that has to interleave its own yields.

    targets, from material_targets(), maps each canonical name to the
    datablocks that name is actually used under. Pass it whenever the
    materials belong to objects already in the scene: without it a name whose
    datablock is spelled differently - a model surface carrying a `.tga` the
    decl does not - gets built into a fresh datablock while the mesh keeps the
    blank one.
    """
    (_settings, _profile, _params, resolved_base, resolved_source,
     resolved_mod) = _resolve_options(context, mode, params, base_dir,
                                      source, mod_dir)
    db = load_database(resolved_source, resolved_base, mod_dir=resolved_mod)
    if summary is None:
        summary = BuildSummary(db)
    wanted = list(names)

    def run():
        for index, name in enumerate(wanted):
            for datablock in (targets or {}).get(name) or [None]:
                build_material(name, mode=mode, params=params,
                               context=context, base_dir=base_dir,
                               source=source, summary=summary,
                               material=datablock, mod_dir=mod_dir)
            yield index + 1, len(wanted), name

    if progress:
        return _ProgressBuild(run(), summary)
    for _ in run():
        pass
    return summary


class _ProgressBuild(object):
    """An iterator of (done, total, name) that also carries the summary.

    A generator cannot hold an attribute, and the caller inside a modal
    operator needs both: the yields to drive its progress bar, and the summary
    to write its report when the loop ends.
    """

    __slots__ = ('_gen', 'summary')

    def __init__(self, gen, summary):
        self._gen = gen
        self.summary = summary

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._gen)

    next = __next__


# ---- reporting ------------------------------------------------------------

def publish_report(summary, context):
    """Publish one build pass's report: the panel's list, and the full text.

    The Generate Materials operator has always done this; the model importers
    never did. They call build_materials() through exactly the same seam, get
    the same BuildSummary back, and then dropped it - so importing a mesh with
    Generate Materials ticked built the materials and reported nothing, and
    the Report panel sat on "No materials generated yet." next to a scene full
    of freshly generated materials. Their only output was the operator's own
    NOTICE lines, which are driven by summary.errors and so say nothing at all
    when everything builds.

    Returns the Text datablock, so a caller can name it in its own result.
    One entry point rather than two calls because publishing to the panel
    without writing the text leaves the panel's "Open Full Report" button
    pointing at the previous run's report.
    """
    if summary is None:
        return None
    summary.publish(context)
    try:
        return write_report_text(summary)
    except (AttributeError, TypeError):
        return None


# ---- scene maintenance ----------------------------------------------------
# refresh_parameters() is defined with the driver infrastructure it belongs to;
# refresh_editor_textures() with the Editor Textures feature.


# ===========================================================================
# END PUBLIC API
# ===========================================================================


# ===========================================================================
# BEGIN EDITOR TEXTURES
#
# Materials that are nothing but a qer_editorimage - clip volumes, triggers,
# origins, the rest of textures/common, guisurf and flare markup - plus any
# the user adds by hand. The engine draws nothing at all for these, so the
# editor image is the only thing there is to show, and being able to hide
# them wholesale is the difference between a readable map and a fog of clip
# brushes.
#
# Visibility works by hiding OBJECTS, never by touching the shader or the
# face data: object.hide_viewport is free whatever the polycount, and it
# declutters Object Mode as well as Edit Mode. Opacity works through the
# shader, on the IDTECH4_EditorMix node.
# ===========================================================================


def _split_editor_texture_objects(context):
    """
    Ensure every editor-texture material (as currently listed in the
    Editor Textures panel) has a dedicated, independently-hideable
    object, so visibility toggling can hide whole objects — which is
    cheap regardless of polycount — instead of touching mesh/face data
    or the shader.

      - If an object's faces are ALL one material, and that material is
        an editor texture, the object is used directly: no duplication,
        no new object, it's just tagged in place. This always happens,
        regardless of the "Split Mixed-Material Objects" checkbox.
      - If an object mixes an editor-texture material with other
        materials, it's only touched when "Split Mixed-Material Objects"
        (settings.editor_textures_split_faces) is enabled — off by
        default. When enabled, the faces using that editor material are
        split off into a new object named '<original>_edittex' (or
        '<original>_edittex_<matname>' if more than one distinct editor
        material lives on the same object), placed in the *same*
        collection(s) as the source object — not a new sub-collection —
        so it's easy to find next to its origin and merge back later if
        needed. The source object's now-empty material slot is removed
        so it isn't reconsidered on the next pass. When the checkbox is
        off, mixed-material objects are left completely alone, so only
        objects that are wholly one editor texture can be hidden.

    Every object that ends up wholly one editor-texture material (pure
    originals and freshly split pieces alike) is tagged with a custom
    property, idtech4_editor_object_material = <material name>, which
    is what the visibility toggles key off of. Already-tagged objects
    are skipped, so this is safe and cheap to re-run every time the
    Editor Textures list changes (new material generated, or added via
    "Add Selected Material") — only genuinely new mixed-material cases
    do any actual mesh work.

    IMPORTANT: this works entirely at the mesh-data level (bmesh +
    Mesh.copy()) and never calls bpy.ops (no mode_set, no mesh.separate,
    no material_slot_remove). Those operators each push a full undo step
    and force a screen redraw; calling them once per editor material per
    mixed-material object made this scale extremely badly on large
    imported meshes with many distinct editor materials — easily
    stalling Blender's UI thread for minutes and looking like a hang.
    Pure data manipulation has none of that overhead.
    """
    import bmesh

    settings = context.scene.idtech4_settings
    editor_mat_names = {item.material_name for item in settings.editor_textures}
    if not editor_mat_names:
        return

    split_faces = settings.editor_textures_split_faces

    scene_objects = [o for o in context.scene.objects if o.type == 'MESH']

    for obj in scene_objects:
        # Already resolved to a single editor-texture material (either a
        # naturally pure object, or a previous split result) — nothing
        # further to do.
        if obj.get('idtech4_editor_object_material'):
            continue

        if not obj.data or not obj.material_slots:
            continue

        # If this object happens to be the one currently being edited,
        # make sure its mesh data reflects the live edit-mesh before we
        # read polygons/material indices from it.
        if obj.mode == 'EDIT':
            obj.update_from_editmode()

        # Which material slots on this object are currently editor
        # textures? Cheap slot-level check before touching face data.
        target_slot_idxs = {
            slot_idx for slot_idx, slot in enumerate(obj.material_slots)
            if slot.material and slot.material.name in editor_mat_names
        }
        if not target_slot_idxs:
            continue

        mesh = obj.data

        # Single O(faces) pass: which slots are actually used, and — only
        # needed when splitting is enabled — which faces (by index)
        # belong to each target material.
        buckets = {}
        used_slots = set()
        for poly in mesh.polygons:
            mi = poly.material_index
            used_slots.add(mi)
            if split_faces and mi in target_slot_idxs:
                buckets.setdefault(mi, []).append(poly.index)

        # Whole object already only ever uses one material slot, and
        # that material is an editor texture: use it in place, no split.
        # This applies regardless of the checkbox — it isn't a split.
        if len(used_slots) <= 1:
            only_idx = next(iter(used_slots), 0)
            only_mat = (obj.material_slots[only_idx].material
                        if only_idx < len(obj.material_slots) else None)
            if only_mat and only_mat.name in editor_mat_names:
                obj['idtech4_editor_object_material'] = only_mat.name
            continue

        # Mixed-material object: only split it off if the checkbox is
        # enabled. Off by default — leave it alone entirely, since it
        # can't be safely hidden as a whole object without also hiding
        # its other, non-editor-texture faces.
        if not split_faces or not buckets:
            continue

        parent_collections = list(obj.users_collection) or [context.scene.collection]
        disambiguate = len(buckets) > 1

        # ── Build each split-off object from a full copy of the source
        # mesh (a fast C-level operation that automatically preserves
        # UVs/vertex colors/etc.), trimmed down to just that material's
        # faces with bmesh. Pure data — no bpy.ops involved.
        all_target_face_idxs = set()
        for slot_idx, face_idxs in sorted(buckets.items()):
            all_target_face_idxs.update(face_idxs)
            mat = obj.material_slots[slot_idx].material
            keep_idx_set = set(face_idxs)

            new_mesh = mesh.copy()
            bm = bmesh.new()
            bm.from_mesh(new_mesh)
            bm.faces.ensure_lookup_table()
            drop_faces = [f for f in bm.faces if f.index not in keep_idx_set]
            if drop_faces:
                bmesh.ops.delete(bm, geom=drop_faces, context='FACES')
            loose_verts = [v for v in bm.verts if not v.link_faces]
            if loose_verts:
                bmesh.ops.delete(bm, geom=loose_verts, context='VERTS')
            bm.to_mesh(new_mesh)
            bm.free()
            new_mesh.update()

            base_name = f"{obj.name}_edittex"
            new_name = base_name
            if disambiguate:
                mat_tag = re.sub(r'[^A-Za-z0-9_]+', '_', mat.name.split('/')[-1])[:24]
                new_name = f"{base_name}_{mat_tag}"
            new_mesh.name = new_name

            # Reduce to a single material slot referencing just this material.
            new_mesh.materials.clear()
            new_mesh.materials.append(mat)
            for poly in new_mesh.polygons:
                poly.material_index = 0

            new_obj = bpy.data.objects.new(new_name, new_mesh)
            new_obj.matrix_world = obj.matrix_world.copy()
            for parent_coll in parent_collections:
                parent_coll.objects.link(new_obj)

            new_obj['idtech4_editor_object_material'] = mat.name

        # ── Trim the SOURCE mesh: remove every split-off face in one
        # single combined pass (not one pass per material).
        src_bm = bmesh.new()
        src_bm.from_mesh(mesh)
        src_bm.faces.ensure_lookup_table()
        remove_faces = [f for f in src_bm.faces if f.index in all_target_face_idxs]
        if remove_faces:
            bmesh.ops.delete(src_bm, geom=remove_faces, context='FACES')
        loose_verts = [v for v in src_bm.verts if not v.link_faces]
        if loose_verts:
            bmesh.ops.delete(src_bm, geom=loose_verts, context='VERTS')
        src_bm.to_mesh(mesh)
        src_bm.free()
        mesh.update()

        # Drop the now-unused material slots from the source object and
        # remap the remaining polygons' material_index to match — pure
        # data manipulation, equivalent to repeated
        # bpy.ops.object.material_slot_remove() calls but done once.
        keep_slots = [i for i in range(len(mesh.materials)) if i not in target_slot_idxs]
        remap = {old: new for new, old in enumerate(keep_slots)}
        kept_materials = [mesh.materials[i] for i in keep_slots]
        for poly in mesh.polygons:
            poly.material_index = remap.get(poly.material_index, 0)
        mesh.materials.clear()
        for m in kept_materials:
            mesh.materials.append(m)


def _set_material_objects_hidden(mat_name, hidden):
    """Hide or show every object across the scene that _split_editor_
    texture_objects() has tagged as wholly the material named mat_name
    (a naturally pure single-material object, or a split-off
    '<object>_edittex' piece). This only ever toggles object.hide_viewport
    — no mesh, face, or shader data is touched — so it's cheap no matter
    how large the object's polycount is, and it declutters Object Mode
    viewports as well as Edit Mode, unlike a per-face hide flag."""
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.get('idtech4_editor_object_material') == mat_name:
            obj.hide_viewport = hidden


def _apply_global_editor_settings(context):
    """Bring every editor-texture material in line with the panel:
    opacity is purely the shared Global Opacity value pushed straight
    onto each material's Mix Shader Fac, while visibility is handled
    independently by hiding/showing the dedicated object(s) for that
    material — an object ends up hidden if either the Global Visibility
    switch or that row's own eye toggle is off."""
    settings = context.scene.idtech4_settings
    for item in settings.editor_textures:
        mat = bpy.data.materials.get(item.material_name)
        if mat and mat.use_nodes and mat.node_tree:
            node = mat.node_tree.nodes.get('IDTECH4_EditorMix')
            if node:
                node.inputs['Fac'].default_value = settings.editor_textures_global_opacity

        hidden = not (settings.editor_textures_global_visible and item.visible)
        _set_material_objects_hidden(item.material_name, hidden)


def _refresh_editor_texture_list(context):
    """Scan all materials for ones flagged as editor-texture-only (built
    from nothing but a qer_editorimage stage, or manually added) and
    (re)populate the Editor Textures listbox, seeding each row's
    individual visibility toggle from the material itself (so a refresh
    never forgets which ones were hidden). Then re-run the object split
    (cheap — it skips anything already resolved, so only genuinely new
    editor-texture materials do real work) and bring every one of them
    in line with the current Global Opacity / Global Visibility
    settings — this is what keeps newly (re)generated or newly-added
    materials in sync with whatever the panel is currently set to."""
    settings = context.scene.idtech4_settings
    settings.editor_textures.clear()
    for mat in sorted(bpy.data.materials, key=lambda m: m.name.lower()):
        if not mat.get('idtech4_editor_texture'):
            continue
        if not (mat.use_nodes and mat.node_tree):
            continue
        if not mat.node_tree.nodes.get('IDTECH4_EditorMix'):
            continue
        item = settings.editor_textures.add()
        item.material_name = mat.name
        # Set via ID-property access to seed the row without
        # re-triggering the update callback.
        item["visible"] = mat.get("idtech4_visible", True)
    settings.editor_textures_index = 0
    _split_editor_texture_objects(context)
    _apply_global_editor_settings(context)


class IDTECH4_OT_RefreshEditorTextures(Operator):
    """Rescan all materials and refresh the Editor Textures list"""
    bl_idname = "idtech4.refresh_editor_textures"
    bl_label = "Refresh Editor Textures"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        _refresh_editor_texture_list(context)
        settings = context.scene.idtech4_settings
        self.report({'INFO'},
                    f"{len(settings.editor_textures)} editor texture material(s) found")
        return {'FINISHED'}


def _splice_editor_mix_into_material(mat):
    """Insert an 'IDTECH4_EditorMix' Mix Shader into an arbitrary
    material's existing node graph so it gains the same Global/individual
    opacity and visibility control as materials the importer classifies
    as editor-texture-only, without discarding whatever shader the
    material already has.

    Whatever currently feeds the Material Output's Surface socket is
    kept as the "shown" side of the mix; a new Transparent BSDF becomes
    the "hidden" side. If the material has no nodes yet, or nothing is
    wired to Surface, a plain gray Diffuse BSDF fallback is used instead
    so there's still something to fade between.

    Returns True if the material was (or already is) wired up; False if
    it has no usable node tree.
    """
    if not mat.use_nodes:
        mat.use_nodes = True
    nt = mat.node_tree
    if nt is None:
        return False

    # Already wired up by a previous "Add Selected Material" or by the
    # generator itself — just make sure it's tagged and stop.
    if nt.nodes.get('IDTECH4_EditorMix'):
        mat['idtech4_editor_texture'] = True
        if 'idtech4_visible' not in mat.keys():
            mat['idtech4_visible'] = True
        return True

    output = next((n for n in nt.nodes if n.bl_idname == 'ShaderNodeOutputMaterial'), None)
    if output is None:
        output = nt.nodes.new('ShaderNodeOutputMaterial')
        output.location = (600, 0)

    surface_input = output.inputs.get('Surface')
    ox, oy = output.location

    existing_socket = None
    if surface_input and surface_input.links:
        existing_socket = surface_input.links[0].from_socket
        for lnk in list(surface_input.links):
            nt.links.remove(lnk)

    if existing_socket is None:
        fallback = nt.nodes.new('ShaderNodeBsdfDiffuse')
        fallback.label = 'Editor Fallback'
        fallback.location = (ox - 300, oy + 150)
        fallback.inputs['Color'].default_value = (0.15, 0.15, 0.15, 1.0)
        existing_socket = fallback.outputs['BSDF']

    transp_bsdf = nt.nodes.new('ShaderNodeBsdfTransparent')
    transp_bsdf.location = (ox - 300, oy - 150)

    mix = nt.nodes.new('ShaderNodeMixShader')
    mix.name  = 'IDTECH4_EditorMix'
    mix.label = 'Editor Texture Opacity'
    mix.location = (ox - 100, oy)
    mix.inputs['Fac'].default_value = 1.0
    nt.links.new(transp_bsdf.outputs['BSDF'], mix.inputs[1])
    nt.links.new(existing_socket,             mix.inputs[2])
    if surface_input is not None:
        nt.links.new(mix.outputs['Shader'], surface_input)

    mat.blend_method = 'BLEND'
    mat['idtech4_editor_texture'] = True
    mat['idtech4_visible'] = True
    return True


class IDTECH4_OT_AddSelectedEditorTexture(Operator):
    """Add the material in the selected object's active material slot
    to the Editor Textures list, splicing in the node setup needed for
    the Global/individual opacity and visibility toggles to work on it"""
    bl_idname = "idtech4.add_selected_editor_texture"
    bl_label = "Add Selected Material"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        obj = context.object
        mat = obj.active_material if obj else None
        if mat is None:
            self.report({'WARNING'},
                        "No active material on the selected object.")
            return {'CANCELLED'}

        if not _splice_editor_mix_into_material(mat):
            self.report({'ERROR'},
                        f"Could not set up '{mat.name}' (no usable node tree).")
            return {'CANCELLED'}

        _refresh_editor_texture_list(context)
        self.report({'INFO'}, f"Added '{mat.name}' to Editor Textures.")
        return {'FINISHED'}


class IDTECH4_UL_EditorTextureList(bpy.types.UIList):
    """Listbox row showing an editor-texture-only material's name and
    an eye-icon toggle for hiding/showing just that one material,
    independent of the Global Visibility switch above the list."""
    bl_idname = "IDTECH4_UL_EditorTextureList"

    def draw_item(self, context, layout, data, item, icon,
                  active_data, active_propname, index):
        row = layout.row(align=True)
        row.label(text=item.material_name, icon='MATERIAL')
        row.prop(item, "visible", text="",
                  icon='HIDE_OFF' if item.visible else 'HIDE_ON',
                  emboss=False)


# ===========================================================================
# END EDITOR TEXTURES
# ===========================================================================


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
    # The refcount lives in bpy.app.driver_namespace, which does NOT survive
    # a file load - the same wipe that used to take the driver functions with
    # it. A count of 0 therefore does not prove the classes are unregistered,
    # so ask Blender rather than trusting the tally: without this, enabling a
    # second idTech4 addon after opening a .blend raises on a bl_idname that
    # is already taken.
    if count == 0 and not hasattr(bpy.types, IDTECH4_PT_sources.__name__):
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


# ===========================================================================
# BEGIN UI
# ===========================================================================

def _shader_parm_update(self, _context):
    """A slider moved, so re-fold every socket that records an expression.

    This is the whole reason those sockets carry a value plus a record
    instead of a driver: parm0..11 change here and at no other moment, so
    re-folding here is exactly as correct as a driver that re-ran every frame
    to produce the same number, and costs nothing on the frames between.
    """
    refresh_parameters(bpy.context)


def _editor_texture_visible_update(self, context):
    _apply_global_editor_settings(context)


def _editor_textures_global_update(self, context):
    _apply_global_editor_settings(context)


def _editor_textures_split_faces_update(self, context):
    if self.editor_textures_split_faces:
        _split_editor_texture_objects(context)
    _apply_global_editor_settings(context)


def _get_display_base_directory(self):
    return get_shared_paths()[0] or "(not set)"


def _get_display_mod_base_directory(self):
    """Blank is the normal state here and means "just use Base Directory",
    so it says that rather than "(not set)", which would read as something
    left misconfigured."""
    return get_shared_paths()[1] or "(none - using Base Directory only)"


def _get_display_materials_source(self):
    """What will actually be read, which is not always what is stored.

    With only a Base Directory set, <base>/materials is used - showing the
    stored blank would leave the panel claiming there is no source while the
    build works fine. With a Mod Base set as well that default is BOTH
    trees' materials folders, mod first, so all of them are named.
    """
    stored = get_shared_paths()[2]
    if stored:
        return stored
    resolved = resolve_shared_paths()[2]
    if not resolved:
        return "(not set)"
    if not isinstance(resolved, str):
        resolved = ', '.join(resolved)
    return '%s  (from Base)' % resolved


class IDTECH4_PG_TableEntry(PropertyGroup):
    """One row in the Created Tables listbox.

    entries_json is not decoration: it is the only copy of a table's values
    that survives into a saved .blend, and _rehydrate_table_registry reads it
    back on file load to put idtech4_tbl() back in business.
    """
    table_name: StringProperty(name="Table Name")
    entry_count: IntProperty(name="Entry Count")
    entries_json: StringProperty(name="Entries JSON")
    # These change how index values are looked up:
    #   clamp: an index outside [0,1] holds at the first/last entry instead of
    #          wrapping, so a time-driven expression ramps once and stops
    #          rather than repeating, with nothing else looking wrong.
    #   snap:  interpolation between entries is off - the lookup steps from
    #          one value to the next instead of blending.
    is_clamp: BoolProperty(name="Clamp", default=False)
    is_snap: BoolProperty(name="Snap", default=False)


class IDTECH4_PG_TopologyGroup(PropertyGroup):
    """One row in Material Topologies: a distinct node-graph structure and
    how many materials share it."""
    signature: StringProperty(name="Signature")
    material_count: IntProperty(name="Material Count")
    example_material: StringProperty(name="Example Material")
    node_count: IntProperty(name="Node Count")
    materials_json: StringProperty(name="Materials JSON")


class IDTECH4_PG_ReportEntry(PropertyGroup):
    text: StringProperty(name="Line")


class IDTECH4_PG_EditorTexture(PropertyGroup):
    material_name: StringProperty(name="Material")
    visible: BoolProperty(name="Visible", default=True,
                          update=_editor_texture_visible_update)


class IDTECH4_PG_Settings(PropertyGroup):

    # -- source paths (display only) ----------------------------------------
    # These mirror the shared idTech4 config file; the editable copies live in
    # the Sources panel. Both are get-only, which is what draws them as locked
    # text fields. Each getter re-reads the config (~50us), which is nothing
    # against the widgets the panel builds and is worth never showing a stale
    # path.
    display_base_directory: StringProperty(
        name="Base Directory",
        description="Base Directory from the shared idTech4 config. Read-only "
                    "here - set it in the idTech4 tab's Sources panel",
        get=_get_display_base_directory,
    )
    display_mod_base_directory: StringProperty(
        name="Mod Base",
        description="Optional Mod Base Directory from the shared idTech4 "
                    "config, searched before Base Directory with Base "
                    "Directory still searched behind it. Read-only here - "
                    "set it in the idTech4 tab's Sources panel",
        get=_get_display_mod_base_directory,
    )
    display_materials_source: StringProperty(
        name="Materials",
        description="The .mtr file or directory this addon reads. Search "
                    "order: this path if set, and nothing else; otherwise "
                    "<Mod Base>/materials if a Mod Base is set, then "
                    "<Base Directory>/materials. Set it with the folder "
                    "button, or clear it to fall back to that pair",
        get=_get_display_materials_source,
    )

    # -- what to build ------------------------------------------------------

    materials_to_create: EnumProperty(
        name="Materials to Create",
        description="Which materials this run builds",
        items=[
            ('ALL_SCENE', "All Scene Materials",
             "Every material already present in this .blend"),
            ('ALL_SOURCE', "All Source Materials",
             "Every material declared anywhere under the Materials Source"),
            ('SELECTED', "Selected Objects' Materials",
             "Only the materials used by the selected objects"),
        ],
        default='ALL_SCENE',
    )

    generation_mode: EnumProperty(
        name="Fidelity Mode",
        description="How much of the engine's shading is reproduced. Every "
                    "rung down renders measurably faster - the numbers are "
                    "from tests/tier_bench.py",
        items=[(p.name, p.label, p.description) for p in
               (PROFILES[k] for k in MODE_ORDER)],
        default=MODE_GOOD,
    )

    parameter_policy: EnumProperty(
        name="Parameters",
        description="What to do with time, parm0..11, global0..7, sound and "
                    "table lookups. Often a bigger performance lever than a "
                    "whole fidelity mode",
        items=[
            (PARAMS_BAKED, "Baked",
             "Fold every expression once, now, against the current frame and "
             "slider values. No drivers at all, so nothing re-runs per frame "
             "and nothing depends on the driver namespace surviving a file "
             "load. The mode to use for a whole-map import. Moving a slider "
             "afterwards needs a rebuild"),
            (PARAMS_DYNAMIC, "Dynamic",
             "Drivers for expressions that reach `time`, and a recorded "
             "expression re-folded on slider moves for the rest. Costs real "
             "frame time: on mars_city1, 219 drivers were 74ms of a 91ms "
             "frame, and evaluating all 219 expressions was 0.3ms of that - "
             "the rest was EEVEE rebuilding each material's GPU shader"),
            (PARAMS_SKIP, "Skip",
             "Refuse every parameter. Conditional stages are dropped without "
             "being evaluated and dynamic terms use their neutral defaults. "
             "The simplest and cheapest result, and the least faithful"),
        ],
        default=PARAMS_BAKED,
    )

    ambient_cap_override: IntProperty(
        name="Ambient Stage Cap",
        description="Maximum ambient stages built per material. 0 means "
                    "uncapped; -1 uses the fidelity mode's own value "
                    "(Maximum uncapped, Good 8, Basic 4, Simple 2). Four "
                    "additive stages measured +31% render time, and Prey "
                    "ships materials 200 stages deep",
        default=-1, min=-1, soft_max=32,
    )

    prefer_editor_image: BoolProperty(
        name="Prefer Editor Image",
        description="Simple mode only: draw the qer_editorimage instead of "
                    "the diffuse stage. Cheaper to load and closer to what "
                    "the level editor shows",
        default=False,
    )

    skip_light_materials: BoolProperty(
        name="Skip Light Materials",
        description="Do not build light shaders at all. They are not surface "
                    "shaders, so what gets built is only a preview of the "
                    "projection texture",
        default=False,
    )

    light_prefer_editor: BoolProperty(
        name="Lights Use Editor Image",
        description="Preview a light shader with its qer_editorimage rather "
                    "than its projection texture. Off by default because "
                    "1,267 of the corpus's 1,271 light materials have a stage "
                    "map and only 15 have an editor image",
        default=False,
    )

    # -- shader parms -------------------------------------------------------
    # One vector each, rather than 14 + 8 near-identical FloatProperty
    # definitions. These are preview values for a static import: idTech4 sets
    # parm0..11 per entity at runtime and global0..7 renderer-wide, and there
    # is no way to know either from a .mtr.

    shader_parms: FloatVectorProperty(
        name="Shader Parms",
        description="parm0..11, the per-entity shader parms. parm0..3 are "
                    "the entity colour and default to opaque white",
        size=12, default=tuple(DEFAULT_PARMS), soft_min=0.0, soft_max=1.0,
        update=_shader_parm_update,
    )
    global_parms: FloatVectorProperty(
        name="Global Parms",
        description="global0..7, the renderer-wide shader parms",
        size=8, default=(0.0,) * 8, soft_min=0.0, soft_max=1.0,
        update=_shader_parm_update,
    )
    sound_amplitude: FloatProperty(
        name="Sound",
        description="The `sound` expression term - an entity's current sound "
                    "amplitude. Roughly 500 light materials modulate their "
                    "brightness with it",
        default=0.0, min=0.0, max=1.0, update=_shader_parm_update,
    )
    spectrum: IntProperty(
        name="Spectrum",
        description="0 shows the ordinary world; N also reveals materials "
                    "declaring `spectrum N`. Only 45 materials in the whole "
                    "corpus, but they are invisible without it",
        default=0, min=0, max=16, update=_shader_parm_update,
    )

    # -- report -------------------------------------------------------------

    report_entries: CollectionProperty(type=IDTECH4_PG_ReportEntry)
    report_summary: StringProperty(name="Report Summary", default="")

    created_tables: CollectionProperty(type=IDTECH4_PG_TableEntry)
    created_tables_index: IntProperty(name="Active Table", default=0)

    topology_groups: CollectionProperty(type=IDTECH4_PG_TopologyGroup)
    topology_groups_index: IntProperty(name="Active Topology", default=0)
    topology_summary: StringProperty(name="Topology Summary", default="")

    # -- editor textures ----------------------------------------------------

    editor_textures: CollectionProperty(type=IDTECH4_PG_EditorTexture)
    editor_textures_index: IntProperty(name="Active Editor Texture", default=0)
    editor_textures_global_opacity: FloatProperty(
        name="Global Opacity", default=1.0, min=0.0, max=1.0,
        description="Opacity of every editor-texture material at once",
        update=_editor_textures_global_update,
    )
    editor_textures_global_visible: BoolProperty(
        name="Global Visibility", default=True,
        description="Show or hide every editor-texture object at once",
        update=_editor_textures_global_update,
    )
    editor_textures_split_faces: BoolProperty(
        name="Split Mixed-Material Objects", default=False,
        description="Split the faces using an editor-texture material off "
                    "into their own '<object>_edittex' object, so they can be "
                    "hidden independently. Off by default: it edits geometry",
        update=_editor_textures_split_faces_update,
    )


class IDTECH4_UL_TableList(bpy.types.UIList):
    """Listbox row showing table name, entry count, and any clamp/snap
    modifiers (these change how a table-driven expression behaves, so
    they need to be visible without opening the inspect popup)."""
    bl_idname = "IDTECH4_UL_TableList"

    def draw_item(self, context, layout, data, item, icon,
                  active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.label(text=item.table_name, icon='PRESET')
            row.label(text=f"{item.entry_count} entries")
            if item.is_clamp:
                row.label(text="", icon='CON_CLAMPTO')
            if item.is_snap:
                row.label(text="", icon='IPO_CONSTANT')
            if not item.is_clamp and not item.is_snap:
                row.label(text="loops")
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=item.table_name)


class IDTECH4_OT_ShowTable(bpy.types.Operator):
    """Show full contents of the selected table"""
    bl_idname  = "idtech4.show_table"
    bl_label   = "Table Contents"
    bl_options = {'INTERNAL'}

    def _get_active(self, context):
        settings = context.scene.idtech4_settings
        idx = settings.created_tables_index
        if 0 <= idx < len(settings.created_tables):
            return settings.created_tables[idx]
        return None

    def invoke(self, context, event):
        if self._get_active(context) is None:
            return {'CANCELLED'}
        return context.window_manager.invoke_popup(self, width=380)

    def execute(self, context):
        return {'FINISHED'}

    def draw(self, context):
        import json
        layout = self.layout
        item = self._get_active(context)
        if not item:
            layout.label(text="No table selected")
            return
        layout.label(text=f"Table: {item.table_name}", icon='PRESET')
        layout.label(text=f"Entries: {item.entry_count}")

        # ── Modifiers — clamp/snap change how the table is looked up, and
        # are easy to miss just by looking at the material text (they can
        # be declared on the `table` decl itself, far from wherever the
        # table is actually referenced in a stage). Spell out the effect
        # in plain language rather than just naming the keyword, since
        # that's the part that actually explains unexpected behaviour
        # (e.g. a table-driven `rotate`/`centerScale` that never loops).
        box = layout.box()
        col = box.column(align=True)
        if item.is_clamp:
            row = col.row(align=True)
            row.label(text="Clamp:", icon='CON_CLAMPTO')
            row.label(text="ON")
            col.label(text="Index holds at first/last entry — does NOT repeat.")
        else:
            row = col.row(align=True)
            row.label(text="Clamp:", icon='CON_CLAMPTO')
            row.label(text="off")
            col.label(text="Index wraps — a rising index (e.g. time) loops.")
        col.separator()
        if item.is_snap:
            row = col.row(align=True)
            row.label(text="Snap:", icon='IPO_CONSTANT')
            row.label(text="ON")
            col.label(text="No interpolation — jumps between entries (stepped).")
        else:
            row = col.row(align=True)
            row.label(text="Snap:", icon='IPO_CONSTANT')
            row.label(text="off")
            col.label(text="Entries blend smoothly (linear interpolation).")

        layout.separator()
        try:
            values = json.loads(item.entries_json)
        except Exception:
            values = []
        box = layout.box()
        col = box.column(align=True)
        for idx, val in enumerate(values):
            row = col.row(align=True)
            row.label(text=f"[{idx}]")
            row.label(text=f"{val:.6g}")


class IDTECH4_UL_TopologyList(bpy.types.UIList):
    """Listbox row showing a topology signature and how many materials
    share it, largest group first."""
    bl_idname = "IDTECH4_UL_TopologyList"

    def draw_item(self, context, layout, data, item, icon,
                  active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.label(text=item.signature[:10], icon='NODETREE')
            row.label(text=f"x{item.material_count}")
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text=item.signature[:10])


class IDTECH4_OT_ShowTopology(bpy.types.Operator):
    """Show which materials share the selected topology"""
    bl_idname  = "idtech4.show_topology"
    bl_label   = "Topology Members"
    bl_options = {'INTERNAL'}

    def _get_active(self, context):
        settings = context.scene.idtech4_settings
        idx = settings.topology_groups_index
        if 0 <= idx < len(settings.topology_groups):
            return settings.topology_groups[idx]
        return None

    def invoke(self, context, event):
        if self._get_active(context) is None:
            return {'CANCELLED'}
        return context.window_manager.invoke_popup(self, width=400)

    def execute(self, context):
        return {'FINISHED'}

    def draw(self, context):
        layout = self.layout
        item = self._get_active(context)
        if not item:
            layout.label(text="No topology selected")
            return
        layout.label(text=f"Signature: {item.signature[:10]}", icon='NODETREE')
        layout.label(text=f"{item.material_count} material(s) share this topology")
        layout.separator()
        box = layout.box()
        col = box.column(align=True)
        names = item.material_names.split(", ") if item.material_names else []
        for name in names:
            col.label(text=name)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def _target_material_names(context, settings):
    """The names this run should build, per Materials to Create."""
    which = settings.materials_to_create
    if which == 'ALL_SOURCE':
        return material_names()
    if which == 'SELECTED':
        names = []
        for obj in context.selected_objects:
            for slot in getattr(obj, 'material_slots', ()):
                if slot.material is not None:
                    names.append(slot.material.name)
            data = getattr(obj, 'data', None)
            for mat in getattr(data, 'materials', ()) or ():
                if mat is not None:
                    names.append(mat.name)
        return sorted(set(names))
    # ALL_SCENE. Grease Pencil materials are excluded and nothing else is.
    # The factory startup file ships one, "Dots Stroke", which does not
    # appear in the material list at all - so it was reported on every run
    # of a default scene as a material that could not be found in the .mtr,
    # naming a datablock the user cannot see. A zero-user REGULAR material
    # is still browsable in that list and stays in scope, and so does
    # Blender's own default "Material": both are things somebody could
    # plausibly mean to build.
    return sorted(m.name for m in bpy.data.materials
                  if not getattr(m, 'is_grease_pencil', False))


class IDTECH4_OT_GenerateMaterials(Operator):
    bl_idname = "idtech4.generate_materials"
    bl_label = "Generate Materials"
    bl_description = ("Build Blender materials from the .mtr source, using "
                      "the fidelity mode and parameter policy below")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.idtech4_settings
        base_dir, mod_dir, source = resolve_shared_paths()
        if not source:
            self.report({'ERROR'},
                        "No .mtr source. Set a Base Directory (its "
                        "materials/ folder is used automatically) or a "
                        "Materials path, using the folder buttons above.")
            return {'CANCELLED'}
        clear_database_cache()
        db = load_database(source, base_dir, mod_dir=mod_dir)
        if db is None:
            self.report({'ERROR'}, "Could not read the Materials Source.")
            return {'CANCELLED'}

        names = _target_material_names(context, settings)
        if not names:
            self.report({'WARNING'},
                        "Nothing selected to build - check 'Create'.")
            return {'CANCELLED'}
        summary = build_materials(names, context=context)
        summary.publish(context)
        refresh_editor_textures(context)
        text = write_report_text(summary)
        self.report({'INFO'}, '%s - full report in the Text Editor as "%s"'
                    % (summary.lines()[0], text.name))
        return {'FINISHED'}


class IDTECH4_OT_RefreshShaderParms(Operator):
    bl_idname = "idtech4.refresh_shader_parms"
    bl_label = "Refresh Parameters"
    bl_description = ("Re-fold every socket that records a parm/global/sound "
                      "expression. Slider moves do this automatically; this "
                      "is for a value changed from a script")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        materials, sockets = refresh_parameters(context)
        self.report({'INFO'}, "Refreshed %d socket(s) across %d material(s)."
                    % (sockets, materials))
        return {'FINISHED'}


class IDTECH4_OT_ShowReportText(Operator):
    bl_idname = "idtech4.show_report_text"
    bl_label = "Open Full Report"
    bl_description = ("Open the untruncated report in a Text Editor window. "
                      "The list above shows at most three entries per kind")
    bl_options = {'REGISTER'}

    def execute(self, context):
        text = bpy.data.texts.get(REPORT_TEXT_NAME)
        if text is None:
            self.report({'WARNING'}, "No report yet - generate materials "
                                     "first.")
            return {'CANCELLED'}
        try:
            bpy.ops.wm.window_new()
            area = context.window_manager.windows[-1].screen.areas[0]
            area.type = 'TEXT_EDITOR'
            area.spaces[0].text = text
            area.spaces[0].show_word_wrap = True
        except (RuntimeError, AttributeError, IndexError):
            self.report({'INFO'},
                        'Open "%s" in a Text Editor.' % REPORT_TEXT_NAME)
        return {'FINISHED'}


class IDTECH4_OT_CountTopologies(Operator):
    bl_idname = "idtech4.count_topologies"
    bl_label = "Count Topologies"
    bl_description = ("Group every idTech4 material in the scene by the "
                      "structure of its node graph")
    bl_options = {'REGISTER'}

    def execute(self, context):
        settings = context.scene.idtech4_settings
        materials = [m for m in bpy.data.materials
                     if m.get('idtech4_material')]
        groups = count_material_topologies(materials)
        settings.topology_groups.clear()
        for signature, entries in groups:
            item = settings.topology_groups.add()
            item.signature = signature
            item.material_count = len(entries)
            item.example_material = entries[0][0]
            item.node_count = entries[0][1]
            item.materials_json = json.dumps([e[0] for e in entries])
        settings.topology_groups_index = 0
        settings.topology_summary = ('%d distinct topologies across %d '
                                     'materials'
                                     % (len(groups), len(materials)))
        self.report({'INFO'}, settings.topology_summary)
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# The Materials panel
# ---------------------------------------------------------------------------

def _wrap_label(text, width):
    """Break a long line into label-sized pieces on word boundaries."""
    words = text.split()
    lines = []
    current = ''
    for word in words:
        candidate = (current + ' ' + word).strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


class IDTECH4_PT_materials(Panel):
    bl_label = "Materials"
    bl_idname = "IDTECH4_PT_materials"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "idTech4 Mtr"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.idtech4_settings
        profile = get_profile(settings.generation_mode)

        box = layout.box()
        box.label(text="Sources", icon='FILE_FOLDER')
        self._path_row(box, settings, "display_base_directory", "Base",
                       IDTECH4_OT_SelectBaseDirectory.bl_idname, 'BASE')
        self._path_row(box, settings, "display_mod_base_directory", "Mod Base",
                       IDTECH4_OT_SelectModBaseDirectory.bl_idname, 'MOD')
        self._path_row(box, settings, "display_materials_source", "Materials",
                       IDTECH4_OT_SelectSource.bl_idname, 'SOURCE')

        layout.prop(settings, "materials_to_create", text="Create")
        layout.prop(settings, "generation_mode", text="Fidelity")
        layout.prop(settings, "parameter_policy", text="Parameters")

        speed = layout.row()
        speed.enabled = False
        speed.label(text=("%.2fx faster than Maximum, measured"
                          % profile.est_speedup)
                    if profile.est_speedup > 1.0
                    else "the reference rung - most expensive")

        box = layout.box()
        box.label(text="Build Options", icon='OPTIONS')
        box.prop(settings, "ambient_cap_override", text="Ambient Cap")
        row = box.row()
        row.enabled = settings.generation_mode == MODE_SIMPLE
        row.prop(settings, "prefer_editor_image")
        box.prop(settings, "skip_light_materials")
        row = box.row()
        row.enabled = not settings.skip_light_materials
        row.prop(settings, "light_prefer_editor")

        layout.operator(IDTECH4_OT_GenerateMaterials.bl_idname,
                        icon='MATERIAL')

        self._draw_editor_textures(layout, settings)
        self._draw_parms(layout, settings)
        self._draw_tables(layout, settings)
        self._draw_topologies(layout, settings)
        self._draw_report(layout, settings)

    # -- sections -----------------------------------------------------------

    def _path_row(self, layout, settings, prop, label, browse, clear_target):
        """One shared path: a locked field, a browse button and a clear.

        The field itself is read-only - it mirrors the shared config file
        rather than a scene property, so a typed path would be silently
        discarded. The two buttons are the only way to change it, which is
        also how the Sources panel in the idTech4 tab does it.
        """
        row = layout.row(align=True)
        sub = row.row(align=True)
        sub.enabled = False
        sub.prop(settings, prop, text=label)
        row.operator(browse, text="", icon='FILE_FOLDER')
        row.operator(IDTECH4_OT_ClearSharedPath.bl_idname, text="",
                     icon='X').target = clear_target

    # -- collapsible sections -----------------------------------------------

    def _draw_parms(self, layout, settings):
        header, body = layout.panel("idtech4_panel_parms", default_closed=True)
        policy = settings.parameter_policy
        title = "Shader Parms"
        if policy == PARAMS_BAKED:
            title += "  (rebuild required)"
        header.label(text=title)
        if body is None:
            return
        body.enabled = policy != PARAMS_SKIP
        if policy == PARAMS_SKIP:
            body.label(text="Parameters are being skipped entirely.",
                       icon='INFO')
        elif policy == PARAMS_BAKED:
            body.label(text="Baked materials do not follow these; "
                            "rebuild to apply.", icon='INFO')
        col = body.column(align=True)
        for index in range(12):
            col.prop(settings, "shader_parms", index=index,
                     text="parm%d" % index)
        col = body.column(align=True)
        for index in range(8):
            col.prop(settings, "global_parms", index=index,
                     text="global%d" % index)
        body.prop(settings, "sound_amplitude")
        body.prop(settings, "spectrum")
        body.operator(IDTECH4_OT_RefreshShaderParms.bl_idname, icon='FILE_REFRESH')

    def _draw_editor_textures(self, layout, settings):
        header, body = layout.panel("idtech4_panel_editor_textures",
                                    default_closed=True)
        header.label(text="Editor Textures (%d)" % len(settings.editor_textures))
        if body is None:
            return
        row = body.row(align=True)
        row.prop(settings, "editor_textures_global_visible", text="",
                 icon='HIDE_OFF' if settings.editor_textures_global_visible
                 else 'HIDE_ON')
        row.prop(settings, "editor_textures_global_opacity", text="Opacity")
        body.prop(settings, "editor_textures_split_faces")
        body.template_list("IDTECH4_UL_EditorTextureList", "",
                           settings, "editor_textures",
                           settings, "editor_textures_index", rows=4)
        row = body.row(align=True)
        row.operator(IDTECH4_OT_RefreshEditorTextures.bl_idname,
                     icon='FILE_REFRESH')
        row.operator(IDTECH4_OT_AddSelectedEditorTexture.bl_idname,
                     icon='ADD')

    def _draw_report(self, layout, settings):
        # The row count is in the header because the section is collapsed by
        # default: "Report" alone gives no reason to open it, and a run that
        # produced 140 findings looks identical to one that produced none.
        header, body = layout.panel("idtech4_panel_report", default_closed=True)
        rows = len(settings.report_entries)
        header.label(text="Report (%d)" % rows if rows else "Report")
        if body is None:
            return
        if settings.report_summary:
            for line in _wrap_label(settings.report_summary, 42):
                body.label(text=line)
        if not rows:
            if not settings.report_summary:
                body.label(text="No materials generated yet.", icon='INFO')
                return
            body.label(text="Nothing to report - every material was built "
                            "as declared.", icon='CHECKMARK')
        # The findings themselves live only in the text datablock now. A
        # fixed-height list in a side panel could never show a whole run's
        # worth of them, and every row it did show was truncated to the
        # panel's width - the button opens the untruncated report instead.
        body.operator(IDTECH4_OT_ShowReportText.bl_idname, icon='WORDWRAP_ON')

    def _draw_tables(self, layout, settings):
        header, body = layout.panel("idtech4_panel_created_tables",
                                    default_closed=True)
        header.label(text="Created Tables (%d)" % len(settings.created_tables))
        if body is None:
            return
        if settings.created_tables:
            body.template_list("IDTECH4_UL_TableList", "",
                               settings, "created_tables",
                               settings, "created_tables_index", rows=5)
            body.operator(IDTECH4_OT_ShowTable.bl_idname, icon='TEXT')
        else:
            body.label(text="No driver in this scene looks up a table.")

    def _draw_topologies(self, layout, settings):
        header, body = layout.panel("idtech4_panel_topologies",
                                    default_closed=True)
        header.label(text="Material Topologies")
        if body is None:
            return
        body.operator(IDTECH4_OT_CountTopologies.bl_idname, icon='NODETREE')
        if settings.topology_summary:
            for line in _wrap_label(settings.topology_summary, 42):
                body.label(text=line)
        if settings.topology_groups:
            body.template_list("IDTECH4_UL_TopologyList", "",
                               settings, "topology_groups",
                               settings, "topology_groups_index", rows=5)
            body.operator(IDTECH4_OT_ShowTopology.bl_idname, icon='TEXT')


# ===========================================================================
# END UI
# ===========================================================================




# ===========================================================================
# register / unregister
# ===========================================================================

def refresh_editor_textures(context):
    """Public alias: rescan the scene for editor-texture-only materials."""
    _refresh_editor_texture_list(context)
    _apply_global_editor_settings(context)


_CLASSES = (
    IDTECH4_PG_TableEntry,
    IDTECH4_PG_TopologyGroup,
    IDTECH4_PG_ReportEntry,
    IDTECH4_PG_EditorTexture,
    IDTECH4_PG_Settings,
    IDTECH4_UL_TableList,
    IDTECH4_UL_TopologyList,
    IDTECH4_UL_EditorTextureList,
    IDTECH4_OT_GenerateMaterials,
    IDTECH4_OT_RefreshShaderParms,
    IDTECH4_OT_CountTopologies,
    IDTECH4_OT_ShowReportText,
    IDTECH4_OT_RefreshEditorTextures,
    IDTECH4_OT_AddSelectedEditorTexture,
    IDTECH4_OT_ShowTable,
    IDTECH4_OT_ShowTopology,
    IDTECH4_PT_materials,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.idtech4_settings = PointerProperty(type=IDTECH4_PG_Settings)
    _register_shared_ui()
    _register_driver_namespace()
    # bpy.app.driver_namespace does not survive a file load, and an
    # already-enabled addon never gets register() called again - see
    # _idtech4_load_post for what that cost.
    if _idtech4_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_idtech4_load_post)


def unregister():
    if _idtech4_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_idtech4_load_post)
    _unregister_driver_namespace()
    _unregister_shared_ui()
    del bpy.types.Scene.idtech4_settings
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
