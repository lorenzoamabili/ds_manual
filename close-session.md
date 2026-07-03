You are closing the development session. Execute these steps in order:

1. Write a detailed session log to .claude/sessions/[DATE].md with:
   - Everything built or changed today
   - Decisions made and why
   - Problems found and how they were solved
   - Current status of each part of the app

2. Update CLAUDE.md with only:
   - Current status of each feature (max 2 lines each)
   - Reference to today's session log
   - Next steps for the next session (max 5 items)
   - Keep it under 40 lines total

3. Update README.md if anything changed today that affects:
   - Setup or installation steps
   - Environment variables
   - Database schema
   - Architecture or folder structure
   - Available routes or features
   Only update sections that actually changed — do not rewrite the whole file.

4. **Sync Android app** — the Android app must mirror all web changes made this session:
   a. Run: npm run build
   b. Run: npx cap sync android
   c. Re-apply Java 17 patch to android/app/capacitor.build.gradle (both sourceCompatibility and
      targetCompatibility must be JavaVersion.VERSION_17 — cap sync overwrites this file with VERSION_21)
   d. Run: cd android && .\gradlew.bat bundleDebug
   e. Confirm BUILD SUCCESSFUL and note the AAB path in the session log
   f. Return to project root: cd ..

5. Scan today's session for strategic prompts the user gave that may not have been archived to .claude/prompts/.
  Identify candidates the same way /save-prompt does:
  - Multi-step instructions wrapped in code blocks
  - Messages with STEP markers and DO NOT guard rails
  - Messages over 500 characters describing a multi-component task
  For each candidate, check whether a corresponding file exists in .claude/prompts/ with today's date AND a matching slug or title in INDEX.md.
  If unarchived candidates are found, list them:
    "Found N strategic prompts from this session that don't appear to be archived:
    1. [first 80 chars of the prompt]…
    2. [first 80 chars]…
    Run /save-prompt for any you want to keep, then run /close-session again."

    If all candidates are archived OR no candidates were found:
      "✓ Prompt archive is up to date"
    and continue with the rest of close-session.
    Do NOT auto-save anything from close-session. Archiving is always explicit via /save-prompt — close-session only flags omissions.

6. Stage all changes:
   git add .

7. Commit with a conventional commit message summarising today's work:
   git commit -m "docs: end of session [DATE] — [one line summary]"

8. Push to the current branch:
   git push origin [current-branch-name]

Do not open a pull request — just push. PRs are created manually.
Tell me what was pushed and to which branch when done.
