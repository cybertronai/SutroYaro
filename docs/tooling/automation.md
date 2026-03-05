# Automation Scripts

## sync_google_docs.py

Pulls Google Docs to local markdown files and extracts all hyperlinks into a references page.

### Usage

```bash
python3 src/sync_google_docs.py              # sync all configured docs
python3 src/sync_google_docs.py --list       # show configured docs
python3 src/sync_google_docs.py --add URL NAME  # add a new doc
```

### Requirements

- `pandoc` (install with `brew install pandoc`)
- Google Doc must be publicly accessible (Share > Anyone with the link)

### How It Works

1. Downloads the Google Doc as HTML via the export URL (`/export?format=html`)
2. Converts HTML to markdown using pandoc
3. Cleans up the markdown (removes excessive whitespace, fixes link formatting)
4. Extracts all hyperlinks and saves them to `docs/references_auto.md`
5. Saves the cleaned markdown to `docs/google-docs/{name}.md`

### Configuration

Docs are configured in `src/docs_config.json`:

```json
[
  {
    "url": "https://docs.google.com/document/d/DOCID/edit",
    "name": "meeting-7-notes",
    "description": "Meeting #7: Sparse parity discussion (02 Mar 2026)"
  }
]
```

### Limitations

- Only works with Google Docs (not Sheets, Slides, or PDFs on Drive)
- Meetings 3-5 link to PDFs on Google Drive, so they can't be pulled this way
- Embedded images are lost (replaced with `[embedded image]`)
- Tables from Google Docs sometimes convert poorly

### After Syncing

After pulling new docs:

1. Add the new pages to `mkdocs.yml` nav
2. Add cross-reference headers (the `!!! info` admonition boxes)
3. Update `docs/meetings/index.md` and `docs/meetings/notes.md` with local links
4. Run `python3 -m mkdocs build` to verify

## References Auto-Extraction

The sync script also builds `docs/references_auto.md` — a flat list of all unique URLs found across all pulled Google Docs. This feeds into the curated `docs/references.md` page which organizes them by category (Google Docs, Drive, Colab, GitHub, Gemini, Other).
