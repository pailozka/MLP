#!/usr/bin/env python3
"""Download official Fashion-MNIST IDX files and convert to the project's CSV format.

Python standard library only. Does not overwrite the legacy data/ files.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

BASE_URL = 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'
# Checksums published by the dataset authors. MD5 identifies the archive; HTTPS
# supplies transport authentication. SHA-256 of each generated CSV is recorded.
FILES = {
    'train-images-idx3-ubyte.gz': '8d4fb7e6c68d591d4c3dfef9ec88bf0d',
    'train-labels-idx1-ubyte.gz': '25c81989df183df01b3e8a0aad5dffbe',
    't10k-images-idx3-ubyte.gz': 'bef4ecab320f06d8554ea6380940ec79',
    't10k-labels-idx1-ubyte.gz': 'bb300cfdad3c16e7a12a480ee83cd310',
}


def download(name, cache):
    path = cache / name
    if not path.exists():
        print(f'Downloading {name}', flush=True)
        with urllib.request.urlopen(BASE_URL + name, timeout=60) as response:
            payload = response.read()
    else:
        payload = path.read_bytes()
    if hashlib.md5(payload).hexdigest() != FILES[name]:
        raise ValueError(f'Checksum mismatch: {path}; remove the damaged archive and retry')
    if not path.exists():
        temporary = path.with_suffix('.tmp')
        temporary.write_bytes(payload)
        temporary.replace(path)
    return gzip.decompress(payload)


def decode_images(payload):
    if len(payload) < 16:
        raise ValueError('Truncated image header')
    magic, count, rows, columns = struct.unpack('>IIII', payload[:16])
    if magic != 2051 or (rows, columns) != (28, 28) or count == 0:
        raise ValueError('Invalid image IDX header')
    if len(payload) != 16 + count * rows * columns:
        raise ValueError('Image IDX payload length mismatch')
    return memoryview(payload)[16:], count


def decode_labels(payload):
    if len(payload) < 8:
        raise ValueError('Truncated label header')
    magic, count = struct.unpack('>II', payload[:8])
    labels = payload[8:]
    if magic != 2049 or count == 0 or len(labels) != count or any(x > 9 for x in labels):
        raise ValueError('Invalid label IDX file')
    return labels


def convert(images, labels, prefix, output):
    pixels, count = decode_images(images)
    targets = decode_labels(labels)
    if count != len(targets):
        raise ValueError('Image and label counts differ')
    paths = [output / f'fashion_mnist_{prefix}_{kind}.csv' for kind in ('vectors', 'labels')]
    for path, kind in zip(paths, ('vectors', 'labels')):
        temporary = path.with_suffix('.csv.tmp')
        with temporary.open('w', encoding='ascii', newline='\n') as stream:
            if kind == 'vectors':
                for i in range(count):
                    stream.write(','.join(map(str, pixels[i * 784:(i + 1) * 784])) + '\n')
            else:
                stream.writelines(f'{value}\n' for value in targets)
        temporary.replace(path)
    return paths, count


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/prepared'))
    parser.add_argument('--cache', type=Path, default=Path('data/raw'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    manifest = {'source': BASE_URL, 'archive_md5': FILES, 'normalization': 'loader divides pixels by 255', 'files': {}}
    for source, split in [('train', 'train'), ('t10k', 'test')]:
        images = download(f'{source}-images-idx3-ubyte.gz', args.cache)
        labels = download(f'{source}-labels-idx1-ubyte.gz', args.cache)
        paths, count = convert(images, labels, split, args.output)
        for path in paths:
            manifest['files'][path.name] = {'rows': count, 'sha256': sha256(path)}
        print(f'Prepared {count} {split} samples', flush=True)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
