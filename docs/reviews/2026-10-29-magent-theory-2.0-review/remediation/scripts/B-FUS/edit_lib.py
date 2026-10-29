"""Exact-match text replacement helper preserving CRLF line endings."""
import sys

ROOT = 'C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/'


class Doc:
    def __init__(self, rel: str) -> None:
        self.path = ROOT + rel
        raw = open(self.path, encoding='utf-8', newline='').read()
        self.crlf = '\r\n' in raw
        self.s = raw.replace('\r\n', '\n')

    def rep(self, old: str, new: str, count: int = 1) -> None:
        c = self.s.count(old)
        if c != count:
            sys.exit(f'MISMATCH ({c}) in {self.path}: {old[:90]!r}')
        self.s = self.s.replace(old, new)

    def save(self) -> None:
        out = self.s.replace('\n', '\r\n') if self.crlf else self.s
        open(self.path, 'w', encoding='utf-8', newline='').write(out)
        print('saved', self.path)
