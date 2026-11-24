"""Turn ⟦latex-subset⟧ markers in a pptx into native PowerPoint equations (OMML, a14:m).

Each text shape that contains a marker is wrapped in mc:AlternateContent: the Choice holds the
real equation objects, the Fallback keeps a plain-Unicode rendering for viewers without a14 support.
"""
import copy, re, sys, zipfile, shutil, os
from lxml import etree

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "a14": "http://schemas.microsoft.com/office/drawing/2010/main",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
}
A, P, M, A14, MC = (("{%s}" % NS[k]) for k in ("a", "p", "m", "a14", "mc"))

SYM = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε", "eta": "η",
    "theta": "θ", "lambda": "λ", "mu": "μ", "sigma": "σ", "tau": "τ", "phi": "φ", "omega": "ω", "Delta": "Δ",
    "nabla": "∇", "odot": "⊙", "cdot": "·", "times": "×", "in": "∈", "sim": "∼", "leftarrow": "←",
    "rightarrow": "→", "to": "→", "le": "≤", "ge": "≥", "approx": "≈", "neq": "≠", "infty": "∞",
    "emptyset": "∅", "langle": "⟨", "rangle": "⟩", "|": "‖", "partial": "∂", "pm": "±", "dots": "…",
    "lt": "<", "gt": ">", "prime": "′", "sum": "∑", "notin": "∉", "ldots": "…", "cdots": "⋯",
}
SPACE = {",": " ", ";": " ", "quad": " ", " ": " "}
FONT = {"mathbb": {"R": "ℝ", "N": "ℕ"}, "mathcal": {"L": "ℒ", "N": "𝒩", "E": "ℰ", "D": "𝒟"}}
ACC = {"hat": "̂", "tilde": "̃", "vec": "⃗"}


def tokenize(s):
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            m = re.match(r"\\([A-Za-z]+|.)", s[i:])
            out.append("\\" + m.group(1)); i += len(m.group(0))
            if out[-1] == "\\text" or out[-1] == "\\mathrm":
                # raw text argument
                j = s.index("}", i); out.append(("TEXT" if out[-1] == "\\text" else "ROMAN", s[i + 1:j])); out.pop(-2); i = j + 1
        elif c == " ":
            i += 1
        else:
            out.append(c); i += 1
    return out


class Parser:
    def __init__(self, toks): self.t, self.i = toks, 0
    def peek(self): return self.t[self.i] if self.i < len(self.t) else None
    def next(self): tok = self.peek(); self.i += 1; return tok

    def seq(self, stop=None):
        nodes = []
        while self.peek() is not None and self.peek() != stop:
            nodes.append(self.script(self.atom()))
        if stop: self.next()
        return nodes

    def group(self):
        if self.peek() == "{": self.next(); return self.seq("}")
        return [self.script(self.atom())] if False else [self.atom()]

    def atom(self):
        tok = self.next()
        if isinstance(tok, tuple): return ("text" if tok[0] == "TEXT" else "roman", tok[1])
        if tok == "{": return ("grp", self.seq("}"))
        if tok == "\\frac": return ("frac", self.group(), self.group())
        if tok == "\\sqrt": return ("sqrt", self.group())
        if tok == "\\bar": return ("bar", self.group())
        if tok[1:] in ACC: return ("acc", ACC[tok[1:]], self.group())
        if tok[1:] in FONT:
            g = self.group(); ch = g[0][1] if g and g[0][0] == "ch" else "?"
            return ("ch", FONT[tok[1:]].get(ch, ch), False)
        if tok == "\\sum": return ("nary", "∑")
        if tok in ("\\left", "\\right"): return ("ch", self.next(), False)
        if tok.startswith("\\") and tok[1:] in SPACE: return ("ch", SPACE[tok[1:]], False)
        if tok.startswith("\\") and tok[1:] in SYM: return ("ch", SYM[tok[1:]], False)
        if tok.startswith("\\"): return ("ch", tok[1:], False)
        return ("ch", tok, tok.isalpha())

    def script(self, base):
        sub = sup = None
        while self.peek() in ("_", "^"):
            k = self.next(); g = self.group()
            if k == "_": sub = g
            else: sup = g
        if base[0] == "nary":
            e = self.script(self.atom()) if self.peek() is not None else ("grp", [])
            return ("nary", base[1], sub, sup, e)
        if sub is not None and sup is not None: return ("subsup", base, sub, sup)
        if sub is not None: return ("sub", base, sub)
        if sup is not None: return ("sup", base, sup)
        return base


def parse(s): return Parser(tokenize(s)).seq()


# ---------------------------------------------------------------- OMML
def E(tag, *kids, **attrs):
    el = etree.Element(M + tag)
    for k, v in attrs.items(): el.set(M + k, v)
    for k in kids:
        if k is not None: el.append(k)
    return el


class Omml:
    def __init__(self, rpr): self.rpr = rpr  # template a:rPr (size/colour/bold)

    def run(self, text, italic, normal=False):
        r = etree.Element(M + "r")
        if normal:
            mp = etree.SubElement(r, M + "rPr"); etree.SubElement(mp, M + "nor")
        ap = copy.deepcopy(self.rpr) if self.rpr is not None else etree.Element(A + "rPr")
        ap.tag = A + "rPr"
        for ch in list(ap):
            if ch.tag in (A + "latin", A + "ea", A + "cs", A + "sym"): ap.remove(ch)
        ap.set("i", "1" if italic else "0")
        ap.set("lang", "en-US")
        lat = etree.SubElement(ap, A + "latin"); lat.set("typeface", "Arial" if normal else "Cambria Math")
        r.append(ap)
        t = etree.SubElement(r, M + "t"); t.text = text
        if text != text.strip(): t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        return r

    def nodes(self, ns, parent):
        buf = []  # consecutive upright characters share one run
        def flush():
            if buf: parent.append(self.run("".join(buf), False)); buf.clear()
        for n in ns:
            if n[0] == "ch" and not n[2]: buf.append(n[1]); continue
            flush(); self.node(n, parent)
        flush()
        return parent

    def node(self, n, parent):
        k = n[0]
        if k == "ch": parent.append(self.run(n[1], n[2]))
        elif k == "text": parent.append(self.run(n[1], False, normal=True))
        elif k == "roman": parent.append(self.run(n[1], False))
        elif k == "grp": self.nodes(n[1], parent)
        elif k == "frac": parent.append(E("f", self.nodes(n[1], E("num")), self.nodes(n[2], E("den"))))
        elif k == "sqrt": parent.append(E("rad", E("radPr", E("degHide", val="1")), E("deg"), self.nodes(n[1], E("e"))))
        elif k == "bar": parent.append(E("bar", E("barPr", E("pos", val="top")), self.nodes(n[1], E("e"))))
        elif k == "acc": parent.append(E("acc", E("accPr", E("chr", val=n[1])), self.nodes(n[2], E("e"))))
        elif k == "sub": parent.append(E("sSub", self.nodes([n[1]], E("e")), self.nodes(n[2], E("sub"))))
        elif k == "sup": parent.append(E("sSup", self.nodes([n[1]], E("e")), self.nodes(n[2], E("sup"))))
        elif k == "subsup": parent.append(E("sSubSup", self.nodes([n[1]], E("e")), self.nodes(n[2], E("sub")), self.nodes(n[3], E("sup"))))
        elif k == "nary":
            pr = E("naryPr", E("chr", val=n[1]), E("limLoc", val="undOvr"),
                   E("subHide", val="1") if n[2] is None else None, E("supHide", val="1") if n[3] is None else None)
            parent.append(E("nary", pr, self.nodes(n[2] or [], E("sub")), self.nodes(n[3] or [], E("sup")), self.nodes([n[4]], E("e"))))


# ---------------------------------------------------------------- Unicode fallback
SUBS = str.maketrans("0123456789+-=()aehijklmnoprstuvxβγφ", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓᵦᵧᵩ")
SUPS = str.maketrans("0123456789+-=()nirT", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱʳᵀ")


def lin(ns):
    out = ""
    for n in ns:
        k = n[0]
        if k in ("ch", "text", "roman"): out += n[1]
        elif k == "grp": out += lin(n[1])
        elif k == "frac":
            a, b = lin(n[1]), lin(n[2]); out += f"({a})/({b})" if len(a) > 1 or len(b) > 1 else f"{a}/{b}"
        elif k == "sqrt": out += "√" + (lin(n[1]) if len(lin(n[1])) == 1 else "(" + lin(n[1]) + ")")
        elif k == "bar": out += "".join(c + "̄" for c in lin(n[1]))
        elif k == "acc": out += lin(n[2]) + n[1]
        elif k in ("sub", "sup", "subsup"):
            out += lin([n[1]])
            def scr(g, tbl, mark):
                s = lin(g); t = s.translate(tbl)
                return t if all(ch in tbl.values() if False else ch != c or c in "" for ch, c in zip(t, s)) and t != s or (len(s) == 1 and t != s) else (mark + (s if len(s) == 1 else "(" + s + ")"))
            if k in ("sub", "subsup"): out += scr(n[2], SUBS, "_")
            if k == "sup": out += scr(n[2], SUPS, "^")
            if k == "subsup": out += scr(n[3], SUPS, "^")
        elif k == "nary":
            out += "Σ" + ("_" + lin(n[2]) if n[2] else "") + " " + lin([n[4]])
    return out


def scr_ok(s, tbl):
    t = s.translate(tbl)
    return all(a != b for a, b in zip(s, t))


def lin2(ns):  # cleaner subscript handling
    out = ""
    for n in ns:
        k = n[0]
        if k in ("ch", "text", "roman"): out += n[1]
        elif k == "grp": out += lin2(n[1])
        elif k == "frac":
            a, b = lin2(n[1]), lin2(n[2]); out += (f"({a})" if len(a) > 2 else a) + "/" + (f"({b})" if len(b) > 2 else b)
        elif k == "sqrt":
            a = lin2(n[1]); out += "√" + (a if len(a) <= 2 else "(" + a + ")")
        elif k == "bar": out += "".join(c + "̄" for c in lin2(n[1]))
        elif k == "acc": out += lin2(n[2]) + n[1]
        elif k in ("sub", "sup", "subsup"):
            out += lin2([n[1]])
            pieces = [(n[2], SUBS, "_")] if k == "sub" else [(n[2], SUPS, "^")] if k == "sup" else [(n[2], SUBS, "_"), (n[3], SUPS, "^")]
            for g, tbl, mark in pieces:
                s = lin2(g)
                out += s.translate(tbl) if scr_ok(s, tbl) else mark + (s if len(s) == 1 else "(" + s + ")")
        elif k == "nary":
            out += "Σ" + (lin2(n[2]).translate(SUBS) if n[2] and scr_ok(lin2(n[2]), SUBS) else ("_" + lin2(n[2]) if n[2] else "")) + " " + lin2([n[4]])
    return out


MARK = re.compile("⟦(.*?)⟧", re.S)


def process_slide(xml):
    root = etree.fromstring(xml)
    changed = False
    for sp in [e for e in root.iter(P + "sp", P + "graphicFrame")]:
        if sp.getparent().tag in (MC + "Choice", MC + "Fallback"): continue
        if sp.tag == P + "sp" and any(a.tag == P + "graphicFrame" for a in sp.iterancestors()): continue
        ts = [t for t in sp.iter(A + "t") if t.text and "⟦" in t.text]
        if not ts: continue
        changed = True
        fallback = copy.deepcopy(sp)
        for t in fallback.iter(A + "t"):
            if t.text and "⟦" in t.text: t.text = MARK.sub(lambda m: lin2(parse(m.group(1))), t.text)
        for r in [t.getparent() for t in ts]:
            rpr = r.find(A + "rPr"); par = r.getparent(); idx = par.index(r)
            parts = MARK.split(r.find(A + "t").text)
            new = []
            for j, seg in enumerate(parts):
                if not seg: continue
                if j % 2 == 0:
                    nr = copy.deepcopy(r); nr.find(A + "t").text = seg; new.append(nr)
                else:
                    a14m = etree.Element(A14 + "m", nsmap={"a14": NS["a14"]})
                    om = etree.SubElement(a14m, M + "oMath", nsmap={"m": NS["m"]})
                    Omml(rpr).nodes(parse(seg), om)
                    new.append(a14m)
            par.remove(r)
            for k, el in enumerate(new): par.insert(idx + k, el)
        alt = etree.Element(MC + "AlternateContent", nsmap={"mc": NS["mc"]})
        ch = etree.SubElement(alt, MC + "Choice", nsmap={"a14": NS["a14"]}); ch.set("Requires", "a14")
        fb = etree.SubElement(alt, MC + "Fallback")
        parent = sp.getparent(); pos = parent.index(sp); parent.remove(sp)
        ch.append(sp); fb.append(fallback); parent.insert(pos, alt)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True) if changed else None


def fill_refs(data, refs):
    t = data.decode("utf-8")
    t = t.replace("§T§", str(refs.get("__total", "?")))
    t = re.sub("§ref:([a-z0-9]+)§", lambda m: str(refs[m.group(1)]), t)
    return t.encode("utf-8")


def main(src, dst, fallback_only=False):
    import json, os
    rp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refs.json")
    refs = json.load(open(rp)) if os.path.exists(rp) else {}
    tmp = dst + ".tmp"
    with zipfile.ZipFile(src) as zi, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zo:
        for it in zi.infolist():
            data = zi.read(it.filename)
            if re.match(r"ppt/(slides|notesSlides)/\w+\.xml$", it.filename): data = fill_refs(data, refs)
            if re.match(r"ppt/slides/slide\d+\.xml$", it.filename):
                out = process_slide(data)
                if out is not None:
                    if fallback_only:  # for LibreOffice previews: keep only the fallback
                        r = etree.fromstring(out)
                        for alt in list(r.iter(MC + "AlternateContent")):
                            sp = alt.find(MC + "Fallback")[0]; alt.getparent().replace(alt, sp)
                        out = etree.tostring(r, xml_declaration=True, encoding="UTF-8", standalone=True)
                    data = out
            zo.writestr(it, data)
    shutil.move(tmp, dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--fallback" in sys.argv)
