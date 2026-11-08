# Build the C example against the rankwise shared library.
ROOT := ../..
CFLAGS ?= -O2 -Wall -Wextra
LIBDIR := $(ROOT)/target/release

stack: stack.c $(ROOT)/rankwise-capi/include/rankwise.h
	cargo build --release -p rankwise-capi --manifest-path $(ROOT)/Cargo.toml
	$(CC) $(CFLAGS) stack.c -I$(ROOT)/rankwise-capi/include -L$(LIBDIR) -lrankwise -Wl,-rpath,$(abspath $(LIBDIR)) -o $@

clean:
	rm -f stack
