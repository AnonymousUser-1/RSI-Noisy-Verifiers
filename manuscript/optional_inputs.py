"""Resolve archived input paths without changing recorded provenance keys or digests."""
from pathlib import Path

BUNDLE = "RSI-Qwen3-4B-Graph-paired-inputs-20261004"


def add_argument(parser):
    parser.add_argument("--inputs", type=Path,
                        help="Extracted optional original Qwen3-4B input archive")


def resolve(code, relative, inputs=None):
    code = Path(code).resolve()
    relative = Path(relative)
    if relative.parts and relative.parts[0] == BUNDLE and inputs is not None:
        path = Path(inputs).resolve().joinpath(*relative.parts[1:])
    else:
        path = code / relative
    if not path.is_file():
        if not relative.parts or relative.parts[0] != BUNDLE:
            raise SystemExit("Missing source file: %s" % relative.as_posix())
        raise SystemExit("Missing original input: %s. This check uses the optional archived inputs; "
                         "pass --inputs /path/to/%s. Fresh experiment runs do not require this archive."
                         % (relative.as_posix(), BUNDLE))
    return path


def source_key(code, path, inputs=None):
    path, code = Path(path).resolve(), Path(code).resolve()
    # Preserve the logical keys used by the original source_files manifest even if the
    # archive is now outside the code checkout.
    if inputs is not None:
        try:
            return str(Path(BUNDLE) / path.relative_to(Path(inputs).resolve()))
        except ValueError:
            pass
    return str(path.relative_to(code))
