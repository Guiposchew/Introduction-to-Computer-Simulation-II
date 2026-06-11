# LaTeX Compilation Guide

## How to Compile Your LaTeX Document Consistently

Your LaTeX document (`hw2-computer_simulation.tex`) compiles successfully, but here are the most reliable ways to run it:

### Method 1: Batch Script (Recommended for Windows)
Double-click `compile_latex.bat` in this folder, or run:
```cmd
.\compile_latex.bat
```

### Method 2: PowerShell Script
Run the PowerShell script:
```powershell
.\compile_latex.ps1
```
Or with cleaning option:
```powershell
.\compile_latex.ps1 -Clean
```

### Method 3: Manual Command
From this directory (`HW2/tex/`), run:
```cmd
pdflatex hw2-computer_simulation.tex
```

## Troubleshooting

### "PDF won't open" or "file is being used by another process"
- Close any PDF viewers that might have the file open
- The PDF is already compiled successfully, just can't be opened automatically

### "pdflatex command not found"
- Make sure TeX Live or MiKTeX is installed
- Add the LaTeX bin directory to your PATH

### "File not found" errors
- Make sure you're running from the `HW2/tex/` directory
- Check that all image files in `plots/` folder exist

### Inconsistent results
- Always run from the same directory (`HW2/tex/`)
- Use the provided scripts for consistency
- Check the `.log` file for detailed error messages

## File Structure
```
HW2/tex/
├── hw2-computer_simulation.tex    # Main LaTeX file
├── compile_latex.bat              # Windows batch script
├── compile_latex.ps1              # PowerShell script
├── plots/                         # Image files
│   ├── Random_0.pdf
│   ├── E_per_spin_T.pdf
│   └── ...
└── README.md                      # This file
```

The document compiles to `hw2-computer_simulation.pdf` (2 pages).