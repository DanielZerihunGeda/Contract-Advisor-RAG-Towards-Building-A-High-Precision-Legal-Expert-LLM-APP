# Variables
PYTHON = python3
SRC_DIR = scripts
APP_FILE = app.py
TEST_DIR = tests
BUILD_DIR = build
BIN_DIR = bin

# List of Python source files
SRCS = $(wildcard $(SRC_DIR)/*.py)

# List of test files
TESTS = $(wildcard $(TEST_DIR)/*.py)

# Executable name
EXEC = $(BIN_DIR)/my_app

# Default target
all: $(EXEC)

# Rule to create the build directory
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Rule to create the executable
$(EXEC): $(BUILD_DIR)
	@mkdir -p $(@D)
	echo "#!/usr/bin/env $(PYTHON)" > $(EXEC)
	cat $(SRCS) >> $(EXEC)
	echo "\n\n# Entry point" >> $(EXEC)
	cat $(APP_FILE) >> $(EXEC)
	chmod +x $(EXEC)

# Rule to run tests
test:
	$(PYTHON) -m unittest discover -s $(TEST_DIR)

# Clean target
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
