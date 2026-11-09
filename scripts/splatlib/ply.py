"""Gaussian-splat PLY files (binary little endian), read and write."""
import numpy as np

SH_C0 = 0.28209479177387814  # spherical-harmonic band-0 constant: colour = SH * SH_C0 + 0.5
_TYPES = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4"}
_NAMES = {"<f4": "float", "u1": "uchar", "<i4": "int"}


def read_ply(path):
    """Return (header lines, structured array of vertices)."""
    with open(path, "rb") as f:
        header = []
        while (line := f.readline().decode().strip()) != "end_header":
            header.append(line)
        if "format binary_little_endian 1.0" not in header:
            raise SystemExit("ERROR: expected a binary little-endian PLY")
        n = next(int(l.split()[2]) for l in header if l.startswith("element vertex"))
        dtype = [(l.split()[2], _TYPES[l.split()[1]]) for l in header if l.startswith("property")]
        data = np.frombuffer(f.read(n * np.dtype(dtype).itemsize), dtype=dtype, count=n).copy()
    return header, data


def write_ply(path, header, data):
    """Write data with the header; the vertex count in the header is set to len(data)."""
    header = [f"element vertex {len(data)}" if l.startswith("element vertex") else l for l in header]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\nend_header\n").encode())
        f.write(data.tobytes())


def ply_header(dtype):
    """Header lines for a structured dtype (float, uchar, int fields)."""
    return (["ply", "format binary_little_endian 1.0", "element vertex 0"]
            + [f"property {_NAMES[np.dtype(t).str]} {n}" for n, t in dtype])
