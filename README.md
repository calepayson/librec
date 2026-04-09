# librec

## Getting Started

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) for package management. Install it, then run:

```bash
uv sync   # install dependencies
./run.sh  # run the project
```

## Logging

The project uses Python's built-in `logging` module. To add logging to a new module:

```python
import logging
logger = logging.getLogger(__name__)
```

Then use `logger.info()`, `logger.warning()`, `logger.error()`, etc. See the [Python logging docs](https://docs.python.org/3/library/logging.html) for more.

## Contributing

**Branches**
To make contributing easy we're going to use development branches then merge
to main. Workflow will look along the lines of:

```bash
git checkout -b <name>/<feature>
# write the code
git add .
git commit -m <description of work>
git push
```

We should be able to handle merges from the gh ui.

**Workflow**
Our goal is a single script that handles everything. Running the project should
be as simple as:

```bash
./run.sh
```

Probably this will just point to a main function. This function will handle 
data download, preprocessing, cross validation, model training, and evalution.
The broad philosophy is that each of these steps should produce an artifact
that serves as the input for the next step. When we run the main script it will
check each step and, if an artifact doesn't already exist, will produce that
artifact. This makes it super easy to work together as state will be entirely
contained in the build script. 
