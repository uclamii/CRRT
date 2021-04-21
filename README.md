# CRRT Preprocessing

**Formatting**: Using flake8, mypy, and black for formatting.

# Includes:

- A .gitignore file from

https://raw.githubusercontent.com/github/gitignore/master/Python.gitignore

- The nbstripout library that removes any jupyter notebook outputs to be avoided from versioning.

## Using nbstripout

- Follow instructions on official repo site

https://github.com/kynan/nbstripout


## Basic usage

- Use pip to install:

```
pip install --upgrade nbstripout
```
- Navigate to repository directory and run on the terminal

```
nbstripout --install
```

- Use the following to check the created filters.

```
nbstripout --status
```
