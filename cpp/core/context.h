#pragma once
namespace pygo::core {
class Context {
public:
 static Context& current();
private:
 Context() = default;
};
}
