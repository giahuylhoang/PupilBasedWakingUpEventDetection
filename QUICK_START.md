# Quick Start Guide

## For First-Time Setup (New Computer)

**That's it! Just run one command:**

```bash
# Make scripts executable and run setup
chmod +x *.sh && ./setup.sh
```

The `setup.sh` script will:
1. ✓ Check if Python 3 is installed (if not, shows installation instructions)
2. ✓ Check if pip and venv are available
3. ✓ Create virtual environment
4. ✓ Install all required dependencies
5. ✓ Verify everything is working

**If Python is not installed**, the script will detect your OS and show you exactly how to install it. After installing Python, just run `./setup.sh` again and it will continue automatically.

For detailed Python installation help, see **[INSTALL_PYTHON.md](INSTALL_PYTHON.md)**.

### Optional: Test Setup

After setup completes, verify everything works:

```bash
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

Complete setup and testing in one command:

```bash
chmod +x *.sh && ./setup.sh && ./test_setup.sh
```

**Note:** `./setup.sh` handles everything automatically - it checks prerequisites, installs dependencies, and sets up the complete environment. If Python is missing, it will guide you through installation.

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

