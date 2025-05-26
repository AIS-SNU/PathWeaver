import os
import argparse
import numpy as np

def read_u8bin(fname):
    with open(fname, "rb") as f:
        rows, cols = np.frombuffer(f.read(8), dtype=np.uint32)
        data = np.fromfile(f, dtype=np.uint8).reshape((rows, cols))
    return data, rows, cols

def write_fbin(fname, data):
    with open(fname, "wb") as f:
        np.asarray(data.shape, dtype=np.uint32).tofile(f)
        data.tofile(f)

def main():
    parser = argparse.ArgumentParser(description="Convert u8bin file to fbin format")
    parser.add_argument("input", type=str, help="Input u8bin file path")
    parser.add_argument("output", type=str, help="Output fbin file path")

    args = parser.parse_args()

    data, rows, cols = read_u8bin(args.input)
    print(f"Read u8bin file with shape ({rows}, {cols})")

    data = data.astype(np.float32)

    write_fbin(args.output, data)
    print(f"Data saved to fbin file: {args.output}")

if __name__ == "__main__":
    main()
