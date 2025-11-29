# Quick Start Guide

## For First-Time Setup (New Computer)

Copy and paste these commands:

```bash
# Make scripts executable
chmod +x setup.sh test_setup.sh init.sh run_individual.sh

# Run setup (takes 2-5 minutes)
./setup.sh

# Test everything works
./test_setup.sh
```

That's it! If all tests pass, you're ready to use the project.

## Quick Test on Existing Setup

To quickly verify everything still works:

```bash
./test_setup.sh
```

## Run Your First Analysis

```bash
./run_individual.sh -d "data/raw/test/cycle_9"
```

## One-Line Setup & Test

If you want to do everything in one command:

```bash
chmod +x *.sh && ./setup.sh && ./test_setup.sh
```

## What Each Script Does

- **`setup.sh`** - Initial setup: creates venv, installs dependencies
- **`test_setup.sh`** - Verifies everything is working correctly
- **`init.sh`** - Activates environment (run this before using Python scripts directly)
- **`run_individual.sh`** - Main script to run analyses

## Troubleshooting

If something fails:

1. **Python not found?**
   - macOS: `brew install python3`
   - Linux: `sudo apt-get install python3 python3-pip python3-venv`

2. **Permission denied?**
   ```bash
   chmod +x *.sh
   ```

3. **Dependencies not installing?**
   ```bash
   source init.sh
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Still having issues?**
   - Check the full [README.md](README.md) for detailed troubleshooting

