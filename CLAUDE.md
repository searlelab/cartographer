# CLAUDE.md

## Collaboration rules (follow strictly)
1. Before making any major change (API changes, refactors, moving files, changing build/test wiring, modifying data formats), propose a plan in 3 to 7 bullet points and wait for confirmation.
2. Do not speculate about code you have not opened. If a conclusion depends on specific behavior, open and read the relevant files first (source, tests, build config, scripts).
3. Prefer the simplest solution that solves the problem. Avoid cleverness, deep refactors, and broad rewrites unless explicitly requested.
4. Minimize blast radius:
   - Change the fewest files possible.
   - Keep diffs small and easy to review.
   - Prefer additive changes over rewiring existing behavior.
5. When you make a claim about behavior, include the evidence path: file names and the exact methods or sections you relied on.

## How to investigate (do this before coding)
1. Identify the entry points and interfaces involved.
2. Read the current implementation and existing tests that exercise it.
3. Search for references (prefer LSP if available, otherwise grep/ripgrep).
4. If behavior is unclear, create or extend a small test that demonstrates current behavior before changing it.

## Coding standards (match the repository)
1. Preserve formatting and conventions already present in nearby files.
   - Follow existing indentation and brace style (this repo commonly uses tabs and K&R braces).
   - Do not reformat unrelated code.
2. Preserve and maintain comments:
   - Keep existing comments unless they are incorrect.
   - Update comments when behavior changes.
   - Use the same comment style as nearby code (JavaDoc blocks for public classes and core components, concise `//` comments for local intent).
3. Documentation:
   - If the repo has a documentation file or folder (README, docs, etc.), keep additions consistent in tone and structure.
   - When a change affects user-visible behavior, formats, CLI flags, configuration, or build steps, update documentation in the same style as existing docs.

## Change management expectations
1. Every change should include a rationale and a verification step.
2. Prefer tests that already exist. Add a new test only when necessary, keep it minimal and focused.
3. If a task spans multiple languages (Java plus native/.NET/Rust/scripts), keep the boundaries explicit:
   - Do not change cross-language interfaces without first showing the plan and getting confirmation.
   - Verify assumptions by reading the relevant bridge points (JNI bindings, CLI contracts, schemas, scripts).

## Communication conventions
- If you are blocked due to missing fixtures, missing external tools, or ambiguous requirements, explain exactly what you checked and what is missing, then offer the smallest next step.
- When proposing alternatives, list tradeoffs briefly and recommend the option that best matches existing patterns in the repo.
