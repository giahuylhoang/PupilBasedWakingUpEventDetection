# .baton: shared context for this repo

Every coding agent and every teammate reads this folder, so each new session starts warm.
Reading it needs no tool. baton (`uv tool install` it) writes notes and rebuilds the briefs;
without it you can still write a note by hand, as described below.

| Path | What it is |
|---|---|
| `BRIEF.md` | the repo brief, loaded at the start of every session (generated) |
| `PROJECT.md` | the project name and charter (hand-written) |
| `streams/<name>/STREAM.md` | a workstream's charter and path globs (hand-written) |
| `streams/<name>/BRIEF.md` | that stream's brief (generated) |
| `notes/YYYY/MM/*.md` | one note per session; only the session that wrote a note changes it |
| `graph.json` | the notes as a graphify graph (generated) |
| `config.toml` | settings (hand-written) |
| `local/` | this machine's cache (gitignored) |

Never edit generated files: `baton build` and the git hooks rebuild them.

## Writing a note with baton

Before you finish a piece of work or commit, pipe the note body into `baton note`:

    baton note --agent <agent> --stream <name> <<'EOF'
    # What this session did
    ...
    EOF

- `--agent` is claude-code, claude-deepseek, kiro, antigravity or other.
- Leave out `--stream` for repo-wide work; baton then picks the stream whose paths match the edited files.
- Running it again in the same session rewrites the same note.
- `--kind consolidation` writes a note whose items supersede or close many older ones.
  `baton show --in-force <stream>` lists them.
- baton adds the facts itself (branch, commits, edited files, times) and redacts secrets.

## The note body

    # A short title: what the session did

    ## Summary
    Two to four sentences: what this session did and why.

    ## State
    Where the stream stands now. The brief shows this as the current state.

    ## Decisions
    - One decision per item, at most 400 characters. {supersedes: k7f3qa.D1}

    ## Open threads
    - Something still open.
    - {closes: k7f3qa.T2} Something that is now done.

    ## Next
    - The next steps.

- The title and `## Summary` are required; the other sections are optional.
- Decisions and Open threads are lists of `- ` items; indent continuation lines by two spaces.
- The n-th decision of note `<id>` is `<id>.D<n>`, and the n-th new open thread is `<id>.T<n>`.
  An item with a `{closes: …}` tag records a closure and gets no number.
- A decision can carry `{supersedes: <id>.D<n>, …}`; an Open threads item can carry `{closes: <id>.T<n>, …}`.
- A note body is at most 12 KB.
- Never put secrets, credentials or personal data in a note.

## Writing a note by hand

Without baton, create `notes/YYYY/MM/YYYY-MM-DD-<agent>-<you>-<slug>-<id>.md` holding this
frontmatter followed by the body above:

    ---
    baton: 1
    id: k7f3qa                      # 6 random characters from a-z and 2-7
    kind: note
    date: 2026-09-25T14:32:00-04:00  # now, with your UTC offset
    agent: other                    # claude-code, claude-deepseek, kiro, antigravity or other
    author: yourname
    branch: main
    stream: stblink                 # leave this line out for repo-wide work
    ---

The briefs are rebuilt by `baton build`, or by the next commit of a teammate who has baton.

## Looking things up

- `baton search <words>`: notes containing every word, newest first.
- `baton show <id>` or `baton show <id>.D2`: a note or one item, with its status.
- `baton show --in-force <stream|repo>`: every decision in force and every open thread.
