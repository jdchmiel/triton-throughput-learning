# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd prime` for full workflow context.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work atomically
bd close <id>         # Complete work
```

## Beads Sync Model (this repo)

**Do not run `bd dolt push`.** This repo has no Dolt remote and needs none.

Issues are shared through the single flat file `.beads/issues.jsonl`, which is
tracked by git. A `pre-commit` hook exports the database into that file and
stages it on every commit, so **`git push` is the only sync step** — beads data
travels with the rest of the repo. There is no second datastore to keep in step.

Edit issues with `bd`, never by hand. `bd` writes its working database and the
hook regenerates the JSONL; hand-editing the JSONL will be overwritten by the
next `bd` write.

You will still see `bd dolt push` suggested by `bd prime`, `bd onboard`, and the
generated block at the bottom of this file. Those are bd's stock templates, not
this repo's configuration. Ignore the command; running it only prints a
"supported remote URLs" error, because the configured remote is the placeholder
`git+ssh://git@github.com/org/repo.git`.

If beads data ever looks out of sync, the fix is `bd export` followed by a
commit — not a dolt push.

## Non-Interactive Shell Commands

**ALWAYS use non-interactive flags** with file operations to avoid hanging on confirmation prompts.

Shell commands like `cp`, `mv`, and `rm` may be aliased to include `-i` (interactive) mode on some systems, causing the agent to hang indefinitely waiting for y/n input.

**Use these forms instead:**
```bash
# Force overwrite without prompting
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file

# For recursive operations
rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r source dest
```

**Other commands that may prompt:**
- `scp` - use `-o BatchMode=yes` for non-interactive
- `ssh` - use `-o BatchMode=yes` to fail instead of prompting
- `apt-get` - use `-y` flag
- `brew` - use `HOMEBREW_NO_AUTO_UPDATE=1` env var

## Progress Log (REQUIRED)

`PROGRESS_LOG.md` is the durable record of what this project has measured and
concluded. **Keep it up to date as you work — do not wait until the end of a
session to write it up.**

**Append-only.** New entries go at the **bottom**. Never rewrite or delete
anything above; if an earlier entry turns out to be wrong, add a new entry that
corrects it and says so explicitly. Corrected mistakes are the most valuable
content in this file.

**Append an entry when any of these happen:**

- A campaign or sweep finishes and produces numbers.
- A measurement settles, or overturns, an open question.
- A bug changes how existing results should be read.
- A Q&A or investigation establishes a fact worth keeping.
- You discover a measurement methodology trap that could mislead a future run.

**Each entry should contain:**

1. A `## YYYY-MM-DD — short descriptive title` heading.
2. **The numbers**, as a table, with the conditions that produced them
   (concurrency, instance count, payload, hardware). A number without its
   conditions is not reproducible and will be misread later.
3. **Why**, not just what. Prose over bullet-spam. A future reader needs the
   reasoning to know when the conclusion stops applying.
4. **What would mislead** — caveats, known-unrealistic parts of the setup, and
   any earlier belief this entry corrects.

**Do not** put task tracking in it. `bd` owns tasks and issues; this file owns
findings. A todo list in the progress log will go stale and contradict `bd`.

**During session close**, before pushing, confirm every finding from the session
is recorded. If you measured something worth keeping and it is not in
`PROGRESS_LOG.md`, write it up — that is part of "Hand off context for the next
session" in the workflow below.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push
   git status  # MUST show "up to date with origin"
   ```
   No `bd dolt push` — see "Beads Sync Model" above. The `pre-commit` hook has
   already exported beads into `.beads/issues.jsonl`, so this push carries it.
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
