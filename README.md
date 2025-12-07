# Symmetry-Aware LLM-Driven Generation and Repair of Twine/Twee Interactive Fiction Graphs

This repository contains a research-oriented framework for **automated generation, structural analysis, and iterative repair** of interactive fiction (IF) graphs using **Large Language Models (LLMs)**.  
The system produces complete branching narratives in the **Twine/Twee** format and fixes structural inconsistencies through a hybrid method combining LLM rewriting with graph-theoretic validation.

A full scientific publication describing this method will be released soon.

---

## Citation (To Appear)

If you use this repository, algorithm, prompts, analysis tools, or experimental results in academic work, please cite the forthcoming paper (details will be added here).

---

## Authors and Affiliation

**Marcin Puchalski**  
**Bożena Woźna-Szcześniak**  
Faculty of Science and Technology  
Jan Dlugosz University in Czestochowa  
Czestochowa, Poland  

---

## Project Overview

This project implements:

- **LLM-based generation** of full Twine/Twee stories.
- **Automatic narrative-graph analysis**, detecting:
  - missing passages,
  - trap cycles (cycles without exits),
  - non-ending dead-end states,
  - naming-based state asymmetries,
  - optional symmetry analysis.
- **Iterative repair loop**, where:
  - the LLM generates missing passages,
  - or produces minimal structural patches fixing cycles, dead-ends, or asymmetry,
  - until the narrative becomes structurally sound.
- **CSV structural logging**, enabling scientific analysis of the convergence process.
- **Import-ready `.twee` output**, compatible with the Twine editor.

---

## Requirements

### 1. Python
Python **3.9+** recommended.

### 2. Ollama  
Install and run from:  
https://ollama.com

Start the Ollama server:

```bash
ollama serve
```

### 3. At Least One Installed Model

Example:

```bash
ollama pull llama3.1
```

---

## Installation

```bash
git clone https://github.com/nosferathoo/generate_twine
cd generate_twine
pip install -r requirements.txt
```

Ensure Ollama is running:

```bash
ollama serve
```

---

## Usage Examples
### Basic story generation

```bash
python3 generate_twine.py llama3.1 gamebook_description.txt -o output.twee
```

### Continue mode (skip initial generation)

```bash
python3 generate_twine.py llama3.1 gamebook_description.txt --continue
```

### Increase number of repair rounds

```bash
python3 generate_twine.py llama3.1 gamebook_description.txt --max-fix-rounds 10
```

### Debug mode — print only the prompt

```bash
python3 generate_twine.py llama3.1 gamebook_description.txt --print-prompt-only
```

---

## Help / Command-Line Arguments

Show help:

```bash
python3 generate_twine.py --help
```

or:

```bash
python3 generate_twine.py -h
```

---

## Output Files

### `output.twee`
Final story in valid Twee format.  
Can be imported and tested using:  
https://twinery.org  

### `log.csv`
A round-by-round structural report including counts of:
- passages  
- links  
- missing passages  
- endings  
- cycles  
- dead-end passages  
- asymmetries  

---

## License

...

---

## Acknowledgements

This research was conducted at the  
Faculty of Science and Technology,  
Jan Dlugosz University in Czestochowa.

