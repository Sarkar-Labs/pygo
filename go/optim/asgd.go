package optim

import (
	"github.com/pygo/go/nn"
	"github.com/pygo/go/tensor"
	"sync"
)

type ASGD struct {
	mu                  sync.Mutex
	groups              []ParamGroup
	lambd, alpha, t0, t float64
	ax                  map[*nn.Parameter]*tensor.Tensor
	step                int64
}

func NewASGD(params []*nn.Parameter, lr, lambd, alpha, t0, wd float64) *ASGD {
	return &ASGD{
		groups: []ParamGroup{{Params: params, LR: lr, WD: wd}},
		lambd:  lambd, alpha: alpha, t0: t0,
		ax: make(map[*nn.Parameter]*tensor.Tensor),
	}
}

func (a *ASGD) Step() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.step++
	for _, g := range a.groups {
		for _, p := range g.Params {
			grad := p.Grad()
			if grad == nil {
				continue
			}
			if g.WD != 0 {
				grad = grad.Add(p.Tensor.MulScalar(g.WD))
			}
			p.Tensor = p.Tensor.Sub(grad.MulScalar(g.LR))
			if _, ok := a.ax[p]; !ok {
				a.ax[p] = p.Tensor.Clone()
			} else {
				mu := 1.0 / (1.0 + a.lambd*a.t)
				a.ax[p] = a.ax[p].MulScalar(1 - mu).Add(p.Tensor.MulScalar(mu))
			}
			a.t++
		}
	}
}
func (a *ASGD) ZeroGrad() {
	for _, g := range a.groups {
		for _, p := range g.Params {
			p.ZeroGrad()
		}
	}
}
func (a *ASGD) AddParamGroup(g ParamGroup)                 { a.groups = append(a.groups, g) }
func (a *ASGD) ParamGroups() []ParamGroup                  { return a.groups }
func (a *ASGD) StateDict() map[string]interface{}          { return nil }
func (a *ASGD) LoadStateDict(map[string]interface{}) error { return nil }
