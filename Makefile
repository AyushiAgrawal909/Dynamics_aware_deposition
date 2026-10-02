PYTHON ?= python3

.PHONY: verify tests code-tests bounded completion exact-width all checksums

verify:
	$(PYTHON) reproduce.py verify

tests:
	$(PYTHON) reproduce.py tests

code-tests:
	$(PYTHON) -m pytest pic_kde_2d/tests

bounded:
	$(PYTHON) reproduce.py bounded

completion:
	$(PYTHON) reproduce.py completion

exact-width:
	$(PYTHON) reproduce.py exact-width

all:
	$(PYTHON) reproduce.py all

checksums:
	$(PYTHON) tools/build_checksums.py
