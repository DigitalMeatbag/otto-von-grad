# Imported source metadata

This directory holds source manifests for external corpora.

- `philosophy.csv` lists philosophy works and their source text URIs.
- `raw/` is created by `scripts/download_import.ps1` and contains downloaded text files plus a `manifest.csv`.

Files under `raw/` are downloaded source data, not cleaned training text, and are ignored by git.

## Project Gutenberg usage

Before downloading, review Project Gutenberg's current policies:

- Terms of Use: <https://www.gutenberg.org/policy/terms_of_use.html>
- Robot access guidance: <https://www.gutenberg.org/policy/robot_access.html>

The downloader defaults to a 2-second delay between requests, skips cached files,
and refuses to fetch more than 100 uncached files from the main Gutenberg site in
one run. For larger imports, use a Project Gutenberg mirror or their official
robot/harvest workflow instead of the main site.

Example:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\download_import.ps1 `
  -AcceptProjectGutenbergTerms `
  -Contact "mailto:you@example.com"
```

After raw download, `scripts/prepare_philosophy_text.ps1` is the placeholder for
turning those source texts into `data/text/philosophy.txt`.
