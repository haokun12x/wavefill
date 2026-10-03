#!/usr/bin/env python3
"""Which device functions differ between two CUDA binaries, compared as SASS.

    python3 tools/wavefill/sass-diff.py OLD_BINARY NEW_BINARY [--cuobjdump PATH] [--cufilt PATH]

Runs `cuobjdump -sass` on both, hashes each function's instruction text with addresses, offsets
and relocation-dependent operands stripped, and compares the hash *multisets*: a template that
gained a parameter changes its mangled name, so names alone would report false changes. Prints
the number of functions per side, how many are unchanged, and the demangled names of functions
whose SASS exists on one side only (grouped by name).
"""
import argparse
import collections
import hashlib
import re
import shutil
import subprocess
import sys

FUNC = re.compile(r"^\s+Function : (\S+)")
ADDR = re.compile(r"/\*[0-9a-f]{4,}\*/")            # instruction address comments
ENC = re.compile(r"/\* 0x[0-9a-f]+ \*/")             # encoding words
HEX = re.compile(r"0x[0-9a-f]+")                     # absolute targets / constant bank offsets


def functions(binary, cuobjdump):
    out = subprocess.run([cuobjdump, "-sass", binary], capture_output=True, text=True, check=True)
    name, body = None, []
    for line in out.stdout.splitlines():
        m = FUNC.match(line)
        if m:
            if name is not None:
                yield name, body
            name, body = m.group(1), []
            continue
        if name is None or "/*" not in line:
            continue
        text = ENC.sub("", ADDR.sub("", line)).strip()
        if text:
            body.append(HEX.sub("X", text))
    if name is not None:
        yield name, body


def digest(binary, cuobjdump):
    table = collections.defaultdict(list)
    for name, body in functions(binary, cuobjdump):
        table[hashlib.sha1("\n".join(body).encode()).hexdigest()].append(name)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--cuobjdump", default=shutil.which("cuobjdump") or "cuobjdump")
    ap.add_argument("--cufilt", default=shutil.which("cu++filt") or "cu++filt")
    args = ap.parse_args()
    old, new = digest(args.old, args.cuobjdump), digest(args.new, args.cuobjdump)
    old_n = sum(len(v) for v in old.values())
    new_n = sum(len(v) for v in new.values())
    only_old = [n for h, names in old.items() for n in names[len(new.get(h, [])):]]
    only_new = [n for h, names in new.items() for n in names[len(old.get(h, [])):]]
    same = old_n - len(only_old)
    print(f"functions: old {old_n}, new {new_n}, identical SASS {same}")

    def show(title, names):
        print(f"{title}: {len(names)}")
        if not names:
            return
        demangled = subprocess.run([args.cufilt], input="\n".join(names), capture_output=True,
                                   text=True).stdout.splitlines()
        for name, count in collections.Counter(
                re.sub(r"<.*", "<...>", d) for d in demangled).most_common():
            print(f"  {count:4d}  {name}")

    show("only in old", only_old)
    show("only in new", only_new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
