# Claude Code Craft Journal

How to be more effective with Claude Code as a tool.

## Workflow Patterns That Work

### Project Setup
- A well-structured CLAUDE.md is the single highest-leverage thing. It's loaded
  every conversation, so every line earns its place or gets cut.
- Skill files (`.claude/skills/`) are great for repeatable workflows — they keep
  CLAUDE.md lean while giving detailed instructions on demand.
- Memory files (`memory/`) persist across conversations. Use MEMORY.md as an index
  (keep it under 200 lines!) and separate topic files for depth.

### Effective Tool Usage
- **Parallel tool calls**: When reading multiple files or running independent
  searches, do them all in one message. Huge time saver.
- **Right tool for the job**: Glob for finding files, Grep for searching content,
  Read for reading files. Don't use Bash for file operations.
- **Subagents**: Use Explore agent for broad codebase research. Don't duplicate
  work between main context and subagents.

### TDD with Claude Code
- Write the test file first, run it to see RED, then implement to GREEN.
- Run tests after every meaningful change — fast feedback loop.
- Use focused test runs during development, full suite before committing.

## Token Optimization

### High-Cost Operations (Avoid)
| Operation | Alternative |
|-----------|-------------|
| Reading entire large files | `Read(file, offset=X, limit=Y)` |
| Re-reading files already in context | Track what you've read |
| Long tool output from Bash | `| tail -n 30` or `head` |
| Running full test suite repeatedly | Run only the specific test file |

### Low-Cost Patterns (Do More)
- Parallel tool calls: 3 reads in one message = 1 round-trip
- Skill files over CLAUDE.md: details only loaded when relevant
- Targeted reads: `Read(file, offset=50, limit=30)` not the whole file
- Concise output: don't narrate what you're about to do in paragraphs

## Git Best Practices
- Don't `git add -A` — stage specific files
- Don't amend published commits
- Always check `git diff` before committing

## Mistakes & Lessons
- (Log entries as you encounter issues)
