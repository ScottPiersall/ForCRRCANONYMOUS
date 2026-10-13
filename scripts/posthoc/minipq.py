"""Minimal pure-Python Parquet reader (flat columns only).

Supports: Thrift compact footer, data page v1/v2, dictionary pages,
PLAIN / PLAIN_DICTIONARY / RLE_DICTIONARY encodings, optional (nullable)
top-level columns, UNCOMPRESSED / SNAPPY / GZIP codecs.
Physical types: BOOLEAN, INT32, INT64, FLOAT, DOUBLE, BYTE_ARRAY.
"""
import struct, zlib


# ---------------- Thrift compact protocol -----------------
class TReader:
    def __init__(self, buf, pos=0):
        self.b = buf; self.p = pos

    def byte(self):
        v = self.b[self.p]; self.p += 1; return v

    def varint(self):
        shift = 0; result = 0
        while True:
            x = self.byte(); result |= (x & 0x7F) << shift
            if not x & 0x80: return result
            shift += 7

    def zigzag(self):
        n = self.varint(); return (n >> 1) ^ -(n & 1)

    def binary(self):
        n = self.varint(); v = self.b[self.p:self.p + n]; self.p += n; return bytes(v)

    def value(self, t):
        if t == 1: return True
        if t == 2: return False
        if t == 3: v = struct.unpack('b', self.b[self.p:self.p + 1])[0]; self.p += 1; return v
        if t in (4, 5, 6): return self.zigzag()
        if t == 7: v = struct.unpack('<d', self.b[self.p:self.p + 8])[0]; self.p += 8; return v
        if t == 8: return self.binary()
        if t in (9, 10): return self.list_()
        if t == 11: return self.map_()
        if t == 12: return self.struct()
        raise ValueError('thrift type %d' % t)

    def list_(self):
        h = self.byte(); n = h >> 4; et = h & 0x0F
        if n == 15: n = self.varint()
        if et in (1, 2):  # bools in lists are one byte each
            return [self.byte() == 1 for _ in range(n)]
        return [self.value(et) for _ in range(n)]

    def map_(self):
        n = self.varint()
        if n == 0: return {}
        kv = self.byte(); kt, vt = kv >> 4, kv & 0x0F
        return {self.value(kt): self.value(vt) for _ in range(n)}

    def struct(self):
        out = {}; last = 0
        while True:
            h = self.byte()
            if h == 0: return out
            t = h & 0x0F; d = h >> 4
            fid = last + d if d else self.zigzag()
            out[fid] = self.value(t); last = fid


# ---------------- codecs -----------------
def snappy_decompress(src):
    r = TReader(src); n = r.varint(); out = bytearray(); p = r.p
    while p < len(src):
        tag = src[p]; p += 1; t = tag & 3
        if t == 0:
            ln = tag >> 2
            if ln >= 60:
                nb = ln - 59; ln = int.from_bytes(src[p:p + nb], 'little'); p += nb
            ln += 1; out += src[p:p + ln]; p += ln; continue
        if t == 1:
            ln = ((tag >> 2) & 7) + 4; off = ((tag >> 5) << 8) | src[p]; p += 1
        elif t == 2:
            ln = (tag >> 2) + 1; off = int.from_bytes(src[p:p + 2], 'little'); p += 2
        else:
            ln = (tag >> 2) + 1; off = int.from_bytes(src[p:p + 4], 'little'); p += 4
        start = len(out) - off
        for k in range(ln): out.append(out[start + k])
    assert len(out) == n, (len(out), n)
    return bytes(out)


def decompress(codec, data, usize):
    if codec == 0: return data
    if codec == 1: return snappy_decompress(data)
    if codec == 2: return zlib.decompress(data, 31)
    raise ValueError('unsupported codec %d' % codec)


# ---------------- encodings -----------------
def rle_hybrid(buf, pos, end, bw, count):
    out = []; r = TReader(buf, pos); nb = (bw + 7) // 8
    while len(out) < count and r.p < end:
        h = r.varint()
        if h & 1:
            groups = h >> 1; nbytes = groups * bw
            chunk = int.from_bytes(buf[r.p:r.p + nbytes], 'little'); r.p += nbytes
            mask = (1 << bw) - 1
            for k in range(groups * 8):
                out.append((chunk >> (k * bw)) & mask if bw else 0)
        else:
            run = h >> 1; v = int.from_bytes(buf[r.p:r.p + nb], 'little') if nb else 0; r.p += nb
            out.extend([v] * run)
    return out[:count]


def plain(buf, pos, ptype, n):
    vals = []
    if ptype == 1: vals = list(struct.unpack_from('<%di' % n, buf, pos))
    elif ptype == 2: vals = list(struct.unpack_from('<%dq' % n, buf, pos))
    elif ptype == 4: vals = list(struct.unpack_from('<%df' % n, buf, pos))
    elif ptype == 5: vals = list(struct.unpack_from('<%dd' % n, buf, pos))
    elif ptype == 0:
        for k in range(n): vals.append(bool((buf[pos + k // 8] >> (k % 8)) & 1))
    elif ptype == 6:
        p = pos
        for _ in range(n):
            ln = struct.unpack_from('<I', buf, p)[0]; p += 4
            vals.append(bytes(buf[p:p + ln]).decode('utf-8', 'replace')); p += ln
    else:
        raise ValueError('ptype %d' % ptype)
    return vals


# ---------------- file -----------------
def read_columns(path, wanted):
    b = open(path, 'rb').read()
    assert b[:4] == b'PAR1' and b[-4:] == b'PAR1'
    mlen = struct.unpack('<I', b[-8:-4])[0]
    meta = TReader(b, len(b) - 8 - mlen).struct()
    schema = meta[2]
    # top-level leaves: children of root (index 0) that have no children
    leaves = {}
    i = 1
    while i < len(schema):
        el = schema[i]; name = el[4].decode(); nchild = el.get(5, 0)
        if nchild:  # nested group: skip it and its subtree
            skip = nchild; i += 1
            while skip:
                sub = schema[i]; skip -= 1; skip += sub.get(5, 0); i += 1
            continue
        leaves[name] = (el.get(1), el.get(3, 0))
        i += 1
    out = {c: [] for c in wanted}
    for rg in meta[4]:
        for cc in rg[1]:
            md = cc[3]; name = md[3][0].decode()
            if name not in out or len(md[3]) != 1: continue
            ptype, rep = leaves[name]
            max_def = 1 if rep == 1 else 0
            codec = md[4]; total = md[5]
            off = md.get(11) or md[9]
            if md.get(11) is not None and md.get(11) < md[9]: off = md[11]
            r = TReader(b, off); dictvals = None; got = 0
            while got < total:
                ph = r.struct(); csize = ph[3]; usize = ph[2]
                raw = b[r.p:r.p + csize]; r.p += csize
                ptypeh = ph[1]
                if ptypeh == 2:  # dictionary page
                    page = decompress(codec, raw, usize)
                    dictvals = plain(page, 0, ptype, ph[7][1]); continue
                if ptypeh == 0:
                    dh = ph[5]; nvals = dh[1]; enc = dh[2]
                    page = decompress(codec, raw, usize); pos = 0
                    if max_def:
                        ln = struct.unpack_from('<I', page, 0)[0]
                        defs = rle_hybrid(page, 4, 4 + ln, 1, nvals); pos = 4 + ln
                    else:
                        defs = [1] * nvals
                elif ptypeh == 3:
                    dh = ph[8]; nvals = dh[1]; enc = dh[4]
                    dl = dh[5]; rl = dh[6]; comp = dh.get(7, True)
                    levels = raw[:rl + dl]; body = raw[rl + dl:]
                    body = decompress(codec, body, usize - rl - dl) if comp else body
                    defs = rle_hybrid(levels, rl, rl + dl, 1, nvals) if max_def else [1] * nvals
                    page = body; pos = 0
                else:
                    raise ValueError('page type %d' % ptypeh)
                nn = sum(defs)
                if enc == 0:
                    vals = plain(page, pos, ptype, nn)
                elif enc in (2, 8):
                    bw = page[pos]; idx = rle_hybrid(page, pos + 1, len(page), bw, nn)
                    vals = [dictvals[k] for k in idx]
                else:
                    raise ValueError('encoding %d' % enc)
                it = iter(vals)
                out[name].extend(next(it) if d else None for d in defs)
                got += nvals
    return out
