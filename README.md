# Demo algorithms for the Nova Rover
This is a repo to demonstrate some system control and filtering ideas for an autonomous rover. In general this repo is for concept development and demonstration.

```NOTE``` Main branch is develop, branch from that - and open a pull request once it is developed.

Master is for deployment ready code.

## Key features
* Path-finding algorithms
	* A*
	* Dijkstra
	* Best-First (greedy)
* Localization algorithms
	* EKF and pose manipulation  
* Mapping algorithms
	* Potential Field mapping

## Code Quality
Follow [PEP8](https://www.python.org/dev/peps/pep-0008/) style guide conventions

### Unit tests
Before a feature can be merged into master, unit tests must be written to validate the correctness of the feature. Tests should be included in the `tests/` sub-directory and be named `test_<feature name>.py`
Unit tests would be run by `pytest` in the clone directory, which can installed using
``` pip install -U pytest ```

### Python packages
To create a python package in a sub-directory, include a file in the directory called `__init__.py`. This defines the directory as a package and allows it to be directly imported as `from <package> import <module>`. When a regular package is imported, this ``__init__.py`` file is implicitly executed, and the objects it defines are bound to names in the package’s namespace.

## Git Conventions
### Branch Naming
* ```feature/<insert feature name here>```
	* for new features
* ```hotfix/<thing being fixed here>```
	* dedicated branches for squashing bugs that have been merged into in master
* ```refactor/<thing being refactored here>```
	* dedicated branch to cleaning up part of the code base

```NOTE``` Before merging code make sure that you haven't broken anything else in the repo.

### Feature Integration
* Once a feature is complete
 1. rebase onto master
 2. Squash all commits in the branch into one
 3. Rename commit to something meaningful
 	* e.g. `Added demo of A* algorithm`
 4. Create a pull request and nominate another member of the team (preferable someone with knowledge of the feature) as a reviewer

## TODO
* Test integration some examples [here](https://sourcery.ai/blog/python-best-practices/)
