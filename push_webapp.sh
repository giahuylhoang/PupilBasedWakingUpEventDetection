#!/bin/bash

# Script to package and push webapp changes to GitHub
# This script handles everything needed to share the webapp with others

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Webapp Packaging and Push Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}Error: Not in a git repository!${NC}"
    echo "Please run this script from the project root directory."
    exit 1
fi

# Check if git remote is configured
if ! git remote get-url origin > /dev/null 2>&1; then
    echo -e "${YELLOW}Warning: No remote 'origin' configured.${NC}"
    echo "You'll need to set up a remote repository first:"
    echo "  git remote add origin <repository-url>"
    exit 1
fi

echo -e "${GREEN}Step 1: Checking git status...${NC}"
git status

echo ""
echo -e "${GREEN}Step 2: Ensuring all webapp files are tracked...${NC}"

# Files that must be tracked
REQUIRED_FILES=(
    "webapp.py"
    "run_webapp.sh"
    "templates/index.html"
    "templates/individual.html"
    "templates/batch.html"
    ".gitignore"
    "requirements.txt"
)

# Check if files exist and add if needed
MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
        echo -e "${YELLOW}Warning: $file not found!${NC}"
    else
        # Check if file is tracked or ignored
        if git check-ignore "$file" > /dev/null 2>&1; then
            echo -e "${YELLOW}Warning: $file is ignored by .gitignore${NC}"
        elif ! git ls-files --error-unmatch "$file" > /dev/null 2>&1; then
            echo -e "${BLUE}Adding $file to git...${NC}"
            git add "$file"
        else
            echo -e "${GREEN}✓ $file is tracked${NC}"
        fi
    fi
done

# Add templates directory if it exists
if [ -d "templates" ]; then
    echo -e "${BLUE}Adding templates directory...${NC}"
    git add templates/*.html 2>/dev/null || true
fi

# Add .gitignore if modified
if git diff --name-only | grep -q ".gitignore"; then
    echo -e "${BLUE}Adding .gitignore...${NC}"
    git add .gitignore
fi

# Add requirements.txt if modified
if git diff --name-only | grep -q "requirements.txt"; then
    echo -e "${BLUE}Adding requirements.txt...${NC}"
    git add requirements.txt
fi

echo ""
echo -e "${GREEN}Step 3: Checking for uncommitted changes...${NC}"
if git diff --staged --quiet && git diff --quiet; then
    echo -e "${YELLOW}No changes to commit. Everything is up to date!${NC}"
    echo ""
    read -p "Do you want to push existing commits? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
else
    echo -e "${BLUE}Files staged for commit:${NC}"
    git diff --staged --name-only
    
    echo ""
    read -p "Do you want to commit these changes? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    
    # Get commit message
    echo ""
    echo -e "${BLUE}Enter commit message (or press Enter for default):${NC}"
    read -r COMMIT_MSG
    if [ -z "$COMMIT_MSG" ]; then
        COMMIT_MSG="Add webapp UI for easy data processing
        
- Add Flask-based web interface
- Add interactive folder browser
- Add support for individual and batch processing
- Add run_webapp.sh script for easy startup"
    fi
    
    echo -e "${GREEN}Committing changes...${NC}"
    git commit -m "$COMMIT_MSG"
fi

echo ""
echo -e "${GREEN}Step 4: Checking current branch...${NC}"
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

echo ""
read -p "Push to '$CURRENT_BRANCH' branch? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo -e "${GREEN}Step 5: Pushing to GitHub...${NC}"
if git push origin "$CURRENT_BRANCH"; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Successfully pushed to GitHub!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${BLUE}Others can now use the webapp by:${NC}"
    echo "  1. git pull origin $CURRENT_BRANCH"
    echo "  2. ./setup.sh  (if first time)"
    echo "  3. ./run_webapp.sh"
    echo ""
else
    echo -e "${RED}Error: Failed to push to GitHub${NC}"
    echo "You may need to set up upstream tracking:"
    echo "  git push -u origin $CURRENT_BRANCH"
    exit 1
fi


