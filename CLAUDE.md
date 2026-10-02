<!-- baton:start -->
## Project context (baton)
This repo keeps shared context in `.baton/`. The repo brief follows.
1. Before working in a stream's paths (see the Streams table), read that stream's brief:
   `.baton/streams/<name>/BRIEF.md`.
2. To look up history: `baton search <words>`, `baton show <id>`, or grep `.baton/notes/`.
3. Before you finish a piece of work or commit, write a session note:
   `baton note --agent claude-code --stream <name>` (leave out `--stream` for repo-wide work;
   format: `.baton/README.md`). If baton isn't installed, write the note file by hand,
   following that README.
Never put secrets, credentials or personal data in a note.

@.baton/BRIEF.md
<!-- baton:end -->
