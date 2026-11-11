package autograd

import "github.com/pygo/go/tensor"

type GradFn interface {
	Apply(grads []*tensor.Tensor) []*tensor.Tensor
	Name() string
}
type Engine struct{ numThreads int }

var defaultEngine = &Engine{numThreads: 4}

func Backward(root *tensor.Tensor, grad *tensor.Tensor, retainGraph bool) {
	defaultEngine.backward(root, grad, retainGraph)
}
func (e *Engine) backward(root *tensor.Tensor, grad *tensor.Tensor, retain bool) {}

var gradEnabled = true

func IsGradEnabled() bool   { return gradEnabled }
func SetGradEnabled(v bool) { gradEnabled = v }
func NoGrad(fn func())      { SetGradEnabled(false); fn(); SetGradEnabled(true) }

type NoGradGuard struct{ prev bool }

func NewNoGradGuard() *NoGradGuard { return &NoGradGuard{prev: IsGradEnabled()} }
func (g *NoGradGuard) Restore()    { SetGradEnabled(g.prev) }
