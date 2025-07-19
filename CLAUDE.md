# CLAUDE.md - Arena Bot Project Rules

## 🗂️ DIRECTORY ACCESS FOR FUTURE CLAUDE INSTANCES

### **Primary Working Directory:**
```bash
cd "/mnt/d/cursor bots/arena_bot_project"
```

### **If Path Issues Occur:**
1. **Find the project:**
   ```bash
   find /mnt -name "arena_bot_project" -type d 2>/dev/null
   ```
2. **Verify correct directory by checking for:**
   - `CLAUDE_ARENA_BOT_CHECKPOINT.md` (main context file)
   - `integrated_arena_bot_gui.py` (primary bot)
   - `arena_bot/` directory
   - `assets/cards/` with 12,000+ images

### **Emergency File Location:**
```bash
find /mnt -name "*CHECKPOINT*" -type f 2>/dev/null
find /mnt -name "integrated_arena_bot*" -type f 2>/dev/null
```

## 🚨 CRITICAL PROJECT RULES


# Using Gemini CLI for Large Codebase Analysis

  When analyzing large codebases or multiple files that might exceed context limits, use the Gemini CLI with its massive
  context window. Use `gemini -p` to leverage Google Gemini's large context capacity.

  ## File and Directory Inclusion Syntax

  Use the `@` syntax to include files and directories in your Gemini prompts. The paths should be relative to WHERE you run the
   gemini command:

  ### Examples:

  **Single file analysis:**
  ```bash
  gemini -p "@src/main.py Explain this file's purpose and structure"

  Multiple files:
  gemini -p "@package.json @src/index.js Analyze the dependencies used in the code"

  Entire directory:
  gemini -p "@src/ Summarize the architecture of this codebase"

  Multiple directories:
  gemini -p "@src/ @tests/ Analyze test coverage for the source code"

  Current directory and subdirectories:
  gemini -p "@./ Give me an overview of this entire project"
  
#
 Or use --all_files flag:
  gemini --all_files -p "Analyze the project structure and dependencies"

  Implementation Verification Examples

  Check if a feature is implemented:
  gemini -p "@src/ @lib/ Has dark mode been implemented in this codebase? Show me the relevant files and functions"

  Verify authentication implementation:
  gemini -p "@src/ @middleware/ Is JWT authentication implemented? List all auth-related endpoints and middleware"

  Check for specific patterns:
  gemini -p "@src/ Are there any React hooks that handle WebSocket connections? List them with file paths"

  Verify error handling:
  gemini -p "@src/ @api/ Is proper error handling implemented for all API endpoints? Show examples of try-catch blocks"

  Check for rate limiting:
  gemini -p "@backend/ @middleware/ Is rate limiting implemented for the API? Show the implementation details"

  Verify caching strategy:
  gemini -p "@src/ @lib/ @services/ Is Redis caching implemented? List all cache-related functions and their usage"

  Check for specific security measures:
  gemini -p "@src/ @api/ Are SQL injection protections implemented? Show how user inputs are sanitized"

  Verify test coverage for features:
  gemini -p "@src/payment/ @tests/ Is the payment processing module fully tested? List all test cases"

  When to Use Gemini CLI

  Use gemini -p when:
  - Analyzing entire codebases or large directories
  - Comparing multiple large files
  - Need to understand project-wide patterns or architecture
  - Current context window is insufficient for the task
  - Working with files totaling more than 100KB
  - Verifying if specific features, patterns, or security measures are implemented
  - Checking for the presence of certain coding patterns across the entire codebase

  Important Notes

  - Paths in @ syntax are relative to your current working directory when invoking gemini
  - The CLI will include file contents directly in the context
  - No need for --yolo flag for read-only analysis
  - Gemini's context window can handle entire codebases that would overflow Claude's context
  - When checking implementations, be specific about what you're looking for to get accurate results # Using Gemini CLI for Large Codebase Analysis



### **BEFORE MAKING ANY CHANGES:**
1. **READ CLAUDE_ARENA_BOT_CHECKPOINT.md COMPLETELY** - This contains all project context and history
2. **NEVER simplify existing production modules** - They exist for good reasons
3. **NEVER create "basic" implementations** - Advanced versions already exist
4. **ALWAYS use existing production modules** - Don't reinvent the wheel
5. **CHECK the checkpoint for current status** - Understand what's already implemented

### **IF BOT "ISN'T WORKING":**
1. **Check if user is using correct launcher** (see Production Launchers section in checkpoint)
2. **Verify environment** (Windows native vs WSL vs GUI requirements)
3. **Check existing implementations** before creating new ones
4. **Read debug output carefully** - it shows what's actually happening

1. First think through the problem, read the codebase for relevant files, and write a plan to tasks/todo.md.
2. The plan should have a list of todo items that you can check off as you complete them
3. Before you begin working, check in with me and I will verify the plan.
4. Then, begin working on the todo items, marking them as complete as you go.
5. Please every step of the way just give me a high level explanation of what changes you made
6. Make every task and code change you do as simple as possible. We want to avoid making any massive or complex changes. Every change should impact as little code as possible. Everything is about simplicity.
7. Finally, add a review section to the [todo.md](http://todo.md/) file with a summary of the changes you made and any other relevant information.

