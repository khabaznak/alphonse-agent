"""Fail if any card crop differs from the matching approved mock crop."""

from itertools import product
from pathlib import Path
import sys

from PIL import Image, ImageChops


def main(directory: Path) -> int:
    states = ("panel", "gmail", "calendar", "gmail-selected")
    failures = 0
    for theme, width, state in product(("light", "dark"), (350, 670), states):
        stem = f"{theme}-{width}-{state}"
        paths = [directory / f"{stem}-{view}.png" for view in ("reference", "production")]
        if not all(path.is_file() for path in paths):
            print(f"{stem}: missing capture")
            failures += 1
            continue
        with Image.open(paths[0]) as mock, Image.open(paths[1]) as production:
            if mock.size != production.size:
                print(f"{stem}: dimensions differ ({mock.size} versus {production.size})")
                failures += 1
                continue
            if mock.info.get("icc_profile") != production.info.get("icc_profile"):
                print(f"{stem}: color profiles differ")
                failures += 1
                continue
            delta = ImageChops.difference(mock.convert("RGBA"), production.convert("RGBA"))
        red, green, blue, alpha = delta.split()
        mask = ImageChops.lighter(ImageChops.lighter(red, green), ImageChops.lighter(blue, alpha))
        changed = mask.width * mask.height - mask.histogram()[0]
        print(f"{stem}: {changed} differing pixels")
        failures += bool(changed)
    return int(failures > 0)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Provide the new capture directory printed by capture-parity.js.")
    raise SystemExit(main(Path(sys.argv[1])))
