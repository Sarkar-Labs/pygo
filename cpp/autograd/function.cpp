// pygo::autograd::Function<Derived> is a template — its definition must
// stay in the header (function.h) for callers to instantiate it with their
// own Derived type. grad_mode_enabled() (declared in function.h, used by
// Function<Derived>::apply_multi) is defined in engine.cpp alongside the
// rest of GradMode, since it's really just a read of GradMode's state.
#include "pygo/function.h"
