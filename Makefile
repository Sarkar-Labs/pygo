.PHONY: configure build test install python-wheel clean

BUILD ?= build
PREFIX ?= /usr/local

configure:
	cmake -S . -B $(BUILD) -DPYGO_BUILD_SHARED=ON -DPYGO_BUILD_TESTS=ON

build: configure
	cmake --build $(BUILD) --parallel

test: build
	ctest --test-dir $(BUILD) --output-on-failure

install: test
	cmake --install $(BUILD) --prefix $(PREFIX)

python-wheel:
	python -m build --wheel

clean:
	rm -rf $(BUILD) dist build/bdist.* *.egg-info
