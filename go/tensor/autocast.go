package tensor

import "sync"

// AutocastMode tracks mixed-precision casting state
type AutocastMode struct {
	DType   DType
	Enabled bool
}

var (
	acMu    sync.Mutex
	acStack []AutocastMode
)

func BeginAutocast(dt DType) {
	acMu.Lock()
	defer acMu.Unlock()
	acStack = append(acStack, AutocastMode{dt, true})
}
func EndAutocast() {
	acMu.Lock()
	defer acMu.Unlock()
	if len(acStack) > 0 {
		acStack = acStack[:len(acStack)-1]
	}
}
func CurrentAutocastDType() (DType, bool) {
	acMu.Lock()
	defer acMu.Unlock()
	if len(acStack) == 0 {
		return Float32, false
	}
	top := acStack[len(acStack)-1]
	return top.DType, top.Enabled
}
func AutocastEnabled() bool { _, ok := CurrentAutocastDType(); return ok }

func (t *Tensor) MaybeAutocast() *Tensor {
	dt, ok := CurrentAutocastDType()
	if !ok || dt == t.DType() {
		return t
	}
	return t.CastTo(dt)
}
