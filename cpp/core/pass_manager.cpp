#include "pygo/pass_manager.h"
#include <cstdio>

namespace pygo::ir {

void PassManager::run(IRGraph& graph, bool verbose) const {
    for (auto& pass : passes_) {
        pass->run(graph);
        if (verbose) {
            std::printf("=== after %s ===\n%s\n", pass->name().c_str(), graph.dump().c_str());
        }
    }
}

} // namespace pygo::ir
