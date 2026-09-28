#!/usr/bin/env bash
# Fetch and verify the two public expression deposits the external-anchor panel
# reads. The on-chip device records are not deposited by the article and are not
# fetched here; this script touches public accessions only.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "${HERE}")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEST="${1:-${ROOT}/runs/external}"

mkdir -p "${DEST}"

fetch() {
  local acc="$1"
  local url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=${acc}&targ=self&form=text&view=brief"
  "${PYTHON_BIN}" - "${url}" "${DEST}/${acc}.soft.txt" <<'PY'
import sys
import urllib.request

url, dest = sys.argv[1], sys.argv[2]
request = urllib.request.Request(url, headers={"User-Agent": "obscal-cohort-fetch"})
with urllib.request.urlopen(request, timeout=60) as response:
    payload = response.read()
if not payload.startswith(b"^SERIES"):
    raise SystemExit(f"{url} did not return a GEO series record")
with open(dest, "wb") as handle:
    handle.write(payload)
print(f"{acc if False else dest}: {len(payload)} bytes")
PY
}

fetch GSE175396
fetch GSE63859

"${PYTHON_BIN}" - "${DEST}" <<'PY'
import pathlib
import sys

dest = pathlib.Path(sys.argv[1])
for path in sorted(dest.glob("*.soft.txt")):
    lines = path.read_text().splitlines()
    title = next(line for line in lines if line.startswith("!Series_title"))
    accession = next(line for line in lines if line.startswith("!Series_geo_accession"))
    print(f"{path.name}\t{accession.split('=')[1].strip()}\t{title.split('=', 1)[1].strip()[:90]}")
PY
