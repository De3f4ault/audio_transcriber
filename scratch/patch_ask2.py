with open("src/audiobench/cli/commands/chat.py") as f:
    content = f.read()

# Fix the indentation issue for the block starting at line 322
lines = content.splitlines()
for i, line in enumerate(lines):
    if line.startswith("    if not transcript_id:") and lines[i - 1].strip() == "sys.exit(0)":
        # The preceding block has 8 spaces? Wait
        pass

# It's better to just re-apply the block with exact correct indentation
