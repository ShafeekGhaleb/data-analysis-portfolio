"""
Tool 1: load_file. Reads a CSV or Excel file into a pandas DataFrame.

A DataFrame is pandas' name for a table: rows and named columns,
like a sheet in Excel.
"""

from pathlib import Path

import pandas as pd

from tools.errors import ToolError
from tools.utils import check_not_empty

# Constants (values that never change) are written in CAPITALS by convention.
SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}  # a "set": a collection with no duplicates
MAX_FILE_SIZE_MB = 50  # protects the program from huge files that would freeze it

# Text files can be saved with different "encodings" (ways of storing letters).
# utf-8 is the modern standard; "utf-8-sig" handles files saved by Excel that
# start with a hidden marker; latin-1 is an older European format that can
# read any byte, so we try it last as a safety net.
CSV_ENCODINGS = ["utf-8", "utf-8-sig", "latin-1"]


def load_file(path, sheet_name=0) -> pd.DataFrame:
    """
    Load a CSV or Excel file and return it as a DataFrame.

    Parameters
    ----------
    path : str or Path
        Location of the file, e.g. "sample_data/sales_data.csv".
    sheet_name : int or str
        For Excel files only: which sheet to read. 0 means "the first sheet".

    Raises
    ------
    ToolError
        With a clear message if the file is missing, too big, the wrong type,
        unreadable or empty.
    """
    # Path() turns a text path into a Path object, which has handy helpers
    # like .exists(), .suffix (the extension) and .stat() (file info).
    path = Path(path)

    # --- 1. Safety checks BEFORE reading anything ---
    if not path.exists():
        raise ToolError(f"File not found: {path}")
    if not path.is_file():
        raise ToolError(f"This is a folder, not a file: {path}")

    extension = path.suffix.lower()  # ".CSV" and ".csv" should both work
    if extension not in SUPPORTED_EXTENSIONS:
        raise ToolError(
            f"Unsupported file type '{extension}'. "
            f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    size_mb = path.stat().st_size / (1024 * 1024)  # bytes -> megabytes
    if size_mb > MAX_FILE_SIZE_MB:
        raise ToolError(
            f"File is {size_mb:.1f} MB; the limit is {MAX_FILE_SIZE_MB} MB."
        )

    # --- 2. Read the file with the right pandas function ---
    if extension == ".csv":
        df = _read_csv(path)
    else:
        df = _read_excel(path, sheet_name)

    # --- 3. Check the result is usable ---
    check_not_empty(df)
    return df


def _read_csv(path: Path) -> pd.DataFrame:
    """
    Read a CSV, trying several encodings until one works.

    The leading underscore in "_read_csv" is a Python convention meaning
    "internal helper: other files should not call this directly".
    """
    for encoding in CSV_ENCODINGS:
        try:
            # sep=None + engine="python" asks pandas to GUESS the separator,
            # so files using ";" (common in Europe) or tabs also work.
            return pd.read_csv(path, encoding=encoding, sep=None, engine="python")
        except UnicodeDecodeError:
            continue  # wrong encoding: try the next one in the list
        except pd.errors.EmptyDataError:
            raise ToolError(f"The file is empty: {path.name}")
        except Exception as error:  # any other reading problem
            raise ToolError(f"Could not read CSV file '{path.name}': {error}")
    # We only get here if every encoding failed.
    raise ToolError(f"Could not decode '{path.name}' with encodings {CSV_ENCODINGS}")


def _read_excel(path: Path, sheet_name) -> pd.DataFrame:
    """Read one sheet of an Excel (.xlsx) file."""
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except ValueError as error:
        # pandas raises ValueError when the requested sheet does not exist.
        raise ToolError(f"Could not read sheet {sheet_name!r} in '{path.name}': {error}")
    except Exception as error:
        raise ToolError(f"Could not read Excel file '{path.name}': {error}")
