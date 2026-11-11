// pygo::autograd::Node is fully defined in node.h (small, and every method
// needs to be visible for the compiler to inline apply() through the
// virtual call — splitting it here would add a translation unit boundary
// for no benefit). This file exists so the build system's expected
// node.cpp target has something to compile.
#include "pygo/node.h"
