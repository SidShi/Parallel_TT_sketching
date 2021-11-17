# Edit Makefile.in for compiler and package information
include ./Makefile.inc

# Compile code
.PHONY: all
all:
	$(MAKE) -C ./src/
	$(MAKE) -C ./test/

# Control how many cores the test runs with
THREADS?=1

# Run test
.PHONY: test
test:
	$(MAKE) -C ./test/ test THREADS=$(THREADS)

# Clean files
clean:
	$(MAKE) -C ./src/ clean
	$(MAKE) -C ./test/ clean