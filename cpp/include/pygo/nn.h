#pragma once
#include "pygo/tensor.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
namespace pygo::nn {
class Module {
public:
    virtual ~Module() = default;
    virtual Tensor forward(const TensorList& inputs) = 0;
    Tensor operator()(const TensorList& inputs) { return forward(inputs); }
    void register_parameter(const std::string& name, Tensor& tensor) { parameters_[name] = &tensor; }
    void register_buffer(const std::string& name, Tensor& tensor) { buffers_[name] = &tensor; }
    template<class T> void register_module(const std::string& name, const std::shared_ptr<T>& module) { modules_[name] = module; }
    void train(bool mode=true) { training_ = mode; }
    void eval() { training_ = false; }
    bool training() const noexcept { return training_; }
    const std::unordered_map<std::string, Tensor*>& parameters() const noexcept { return parameters_; }
protected:
    bool training_{true};
    std::unordered_map<std::string, Tensor*> parameters_;
    std::unordered_map<std::string, Tensor*> buffers_;
    std::unordered_map<std::string, std::shared_ptr<void>> modules_;
};
class Sequential : public Module {
public:
    Sequential() = default;
    Tensor forward(const TensorList& inputs) override { Tensor x=inputs.at(0); for(auto& m: modules_order_) x=m->forward({x}); return x; }
    template<class T> void push_back(std::shared_ptr<T> m) { register_module("layer"+std::to_string(modules_order_.size()),m); modules_order_.push_back(std::static_pointer_cast<Module>(m)); }
private:
    std::vector<std::shared_ptr<Module>> modules_order_;
};
}
