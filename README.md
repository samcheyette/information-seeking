---
editor_options: 
  markdown: 
    wrap: 72
---

# Information-seeking in puzzle games

This repository contains the implementation and experimental data for
our research on information-seeking in puzzle games, specifically
focusing on Mastermind and ButtonSet paradigms.

## Project Structure

The project is organized into the following directories:

- `Mastermind/`: Contains the implementation of the Mastermind model
- `ButtonSet/`: Contains the implementation of the ButtonSet model
- `Experiments/`: Contains the experimental data collected from human participants

## Models

The project implements two different concept learning paradigms:

### Mastermind Model

Located in `Mastermind/`, this model implements a grammar-based learning
system for the Mastermind game, where players must deduce a hidden code
through feedback on their guesses.

Key components: - Grammar implementation for representing game rules -
Logical expression generation and evaluation - Feedback-based learning
mechanisms

### ButtonSet Model

Located in `ButtonSet/`, this model implements a similar grammar-based
approach for a button selection task, where participants learn rules
about valid button combinations.

Key components: - Set-based grammar implementation - Possibility space
exploration - Rule learning from examples

## Experimental Data

The `Experiments/` directory contains experimental data collected from
human participants:

### Experiments

-   Data from participants solving both Mastermind and ButtonSet tasks
-   Includes trial-by-trial responses, timing information, and accuracy
    measures
-   Format: CSV files with detailed participant interaction data

## Usage

For researchers interested in replicating our results or using our
models:

``` python
from Mastermind.grammar import make_grammar as make_mastermind_grammar
from ButtonSet.grammar import make_grammar as make_buttonset_grammar

# Initialize Mastermind grammar
mm_grammar = make_mastermind_grammar(n_positions=4, n_colors=4)

# Initialize ButtonSet grammar
bs_grammar = make_buttonset_grammar(n_positions=4)
```

