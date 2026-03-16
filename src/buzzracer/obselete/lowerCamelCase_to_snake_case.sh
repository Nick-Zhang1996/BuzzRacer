#!/bin/bash

# ==============================================================================
# PEP 8 Function Name Refactor Script
#
# Description:
# This script recursively finds all Python files (`.py`) in the current
# directory and its subdirectories. It identifies function names written in
# lowerCamelCase and refactors them to snake_case, as recommended by PEP 8.
# The script updates both the function definitions and all their call sites.
#
# Usage:
# 1. Save this script as `refactor.sh`.
# 2. Make it executable: `chmod +x refactor.sh`
# 3. Run it from the root directory of your Python project: `./refactor.sh`
#
# IMPORTANT:
# It's highly recommended to run this on a code base that is under version
# control (like Git), so you can easily review and revert changes if needed.
# ==============================================================================

echo "🚀 Starting PEP 8 function name refactoring..."

# --- Step 1: Find all unique lowerCamelCase function names ---
# We use `grep` with a Perl-compatible regular expression (PCRE) to find
# function definitions that follow the lowerCamelCase pattern.
#
# Regex breakdown:
#   -r : Recurse through directories.
#   -h : Suppress the prefixing of file names on output. This is the key fix.
#   -P : Use Perl-compatible regex for advanced features like lookarounds.
#   -o : Print only the matched parts of the lines.
#   --include="*.py" : Only search within Python files.
#
#   '(?<=def )' : Positive lookbehind. Asserts that "def " precedes the match,
#                 but doesn't include it in the result.
#   '[a-z][a-zA-Z0-9]*[A-Z]+[a-zA-Z0-9]*' : This is the core pattern for
#                 lowerCamelCase. It matches a string that starts with a
#                 lowercase letter and contains at least one uppercase letter.
#   '(?=\()' : Positive lookahead. Asserts that a '(' follows the match,A
#              but doesn't include it in the result.
#
# The final list is sorted and made unique with `sort -u`.
CAMEL_NAMES=$(grep -rhPo '(?<=def )[a-z][a-zA-Z0-9]*[A-Z]+[a-zA-Z0-9]*(?=\()' . --include="*.py" | sort -u)

# Check if any functions were found to refactor.
if [ -z "$CAMEL_NAMES" ]; then
    echo "✅ No lowerCamelCase function definitions found. Your code is already compliant!"
    exit 0
fi

echo "🔍 Found the following functions to rename:"
echo "$CAMEL_NAMES"
echo "-------------------------------------------"

# --- Step 2: Loop through each name and perform the replacement ---
for camel_name in $CAMEL_NAMES; do
    # Convert the camelCase name to snake_case.
    # 1. `sed -r 's/([A-Z])/_\1/g'` inserts an underscore before each capital letter.
    # 2. `tr '[:upper:]' '[:lower:]'` converts the entire string to lowercase.
    snake_name=$(echo "$camel_name" | sed -r 's/([A-Z])/_\1/g' | tr '[:upper:]' '[:lower:]')

    echo "🔧 Renaming: ${camel_name} -> ${snake_name}"

    # Find all Python files and perform an in-place replacement using `sed`.
    # `find . -type f -name "*.py" -print0 | xargs -0` is a robust way to
    # process files, correctly handling filenames with spaces or special characters.
    #
    # The `sed` command `s/\b${camel_name}\b/${snake_name}/g` replaces all occurrences.
    # `\b` are word boundaries, which are crucial to prevent partial replacements.
    # For example, it ensures that renaming `myFunction` doesn't accidentally
    # change a different function named `myFunctionExtended`.
    find . -type f -name "*.py" -print0 | xargs -0 sed -i "s/\b${camel_name}\b/${snake_name}/g"
done

echo "-------------------------------------------"
echo "✅ Refactoring complete!"
echo "✨ Your Python files now follow PEP 8 snake_case naming for functions."
