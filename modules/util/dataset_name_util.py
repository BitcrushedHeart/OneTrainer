import hashlib
import os
from pathlib import Path

# Both WSL's drvfs view of a Windows drive (which exposes names as UTF-8) and RunPod's MooseFS
# /workspace volume cap a single path component at 255 *bytes*. NTFS instead counts 255 UTF-16
# units, so a caption-as-filename full of multi-byte characters (smart quotes, em-dashes, emoji)
# can be perfectly valid locally yet exceed 255 bytes once UTF-8 encoded - which aborts the rsync
# upload with ENAMETOOLONG ("File name too long"). On top of that, rsync transfers via a temp file
# named ".<name>.XXXXXX" (the original name plus 8 bytes) in the destination, so the final name has
# to leave room for that too. 240 bytes keeps both the name and its transfer temp under 255.
_MAX_NAME_BYTES = 240

# OneTrainer pairs an image with its caption (<base>.txt) and masks (<base>-masklabel.png /
# <base>-condlabel.png) by shared base name, so a whole group has to be renamed together or the
# pairing silently breaks. See modules/dataLoader/mixin/DataLoaderText2ImageMixin.py.
_GROUP_POSTFIXES = ("-masklabel", "-condlabel")


def _utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _image_base(filename: str) -> str:
    stem, _ext = os.path.splitext(filename)
    for postfix in _GROUP_POSTFIXES:
        if stem.endswith(postfix):
            return stem[: -len(postfix)]
    return stem


def _truncate_utf8(text: str, max_bytes: int) -> str:
    # Cut on a UTF-8 character boundary so we never emit a half-encoded code point.
    return text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


def normalize_long_filenames(
    root: Path, recursive: bool, max_name_bytes: int = _MAX_NAME_BYTES
) -> list[tuple[str, str]]:
    """Shorten dataset filenames whose UTF-8 length exceeds the 255-byte filesystem limit.

    Files are renamed in groups sharing an image base name (image + .txt caption + -masklabel /
    -condlabel masks) so OneTrainer's image/caption/mask pairing stays intact. Truncation appends a
    short hash of the original base to stay collision-free and idempotent (already-short names are
    left untouched on re-run). Returns the list of (old_name, new_name) renames performed.
    """
    if not root.is_dir():
        return []

    directories = [root]
    if recursive:
        directories = [p for p in root.rglob("*") if p.is_dir()]
        directories.append(root)

    renames: list[tuple[str, str]] = []
    for directory in directories:
        groups: dict[str, list[Path]] = {}
        for path in directory.iterdir():
            if path.is_file():
                groups.setdefault(_image_base(path.name), []).append(path)

        for base, members in groups.items():
            if all(_utf8_len(member.name) <= max_name_bytes for member in members):
                continue

            # Every member name starts with `base`; the remainder is its suffix (ext, or
            # -masklabel.png etc). Reserve room for the longest suffix in the group plus the hash.
            worst_suffix = max(_utf8_len(member.name[len(base) :]) for member in members)
            digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
            budget = max(1, max_name_bytes - worst_suffix - (len(digest) + 1))
            new_base = f"{_truncate_utf8(base, budget)}_{digest}"

            targets = {member: member.with_name(f"{new_base}{member.name[len(base) :]}") for member in members}
            # Skip the whole group on any collision so we never half-rename and break pairing.
            if any(target != member and target.exists() for member, target in targets.items()):
                continue

            for member, target in targets.items():
                if target == member:
                    continue
                member.rename(target)
                renames.append((member.name, target.name))

    return renames
