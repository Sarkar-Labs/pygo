// ONNX export for PyGo models
#include "pygo/serialization.h"
#include <fstream>
#include <stdexcept>
namespace pygo::serialization {

ONNXExporter::ONNXExporter(ONNXExportConfig cfg) : cfg_(cfg) {}

void ONNXExporter::export_model(
    const std::string& path,
    std::function<Tensor(const TensorList&)> fwd,
    const TensorList& inputs)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write: " + path);

    // Minimal ONNX header
    f << "ir_version: 8
";
    f << "opset_import { domain: "" version: " << cfg_.opset_version << " }
";
    f << "graph { name: "pygo_model"
";
    f << "  input { name: "input_0" }
";
    f << "  output { name: "output_0" }
";
    f << "}
";
}

} // namespace pygo::serialization
