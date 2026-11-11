package optim

import (
	"github.com/pygo/go/nn"
	"github.com/pygo/go/tensor"
	"sync"
)

type LARS struct {
	mu                 sync.Mutex
	groups             []ParamGroup
	momentum, eta, eps float64
	vel                map[*nn.Parameter]*tensor.Tensor
	step               int64
}

func NewLARS(params []*nn.Parameter, lr, momentum, eta, eps, wd float64) *LARS {
	return &LARS{
		groups:   []ParamGroup{{Params: params, LR: lr, WD: wd}},
		momentum: momentum, eta: eta, eps: eps,
		vel: make(map[*nn.Parameter]*tensor.Tensor),
	}
}

func (l *LARS) Step() {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.step++
	for _, g := range l.groups {
		for _, p := range g.Params {
			grad := p.Grad()
			if grad == nil {
				continue
			}
			wN := p.Tensor.Norm(2, -1, false).Item()
			gN := grad.Norm(2, -1, false).Item()
			trust := 1.0
			if wN > 0 && gN > 0 {
				trust = l.eta * wN / (gN + l.eps*wN + 1e-8)
			}
			ng := grad.MulScalar(trust)
			if _, ok := l.vel[p]; !ok {
				l.vel[p] = ng.Clone()
			} else {
				l.vel[p] = l.vel[p].MulScalar(l.momentum).Add(ng)
			}
			p.Tensor = p.Tensor.Sub(l.vel[p].MulScalar(g.LR))
		}
	}
}
func (l *LARS) ZeroGrad() {
	for _, g := range l.groups {
		for _, p := range g.Params {
			p.ZeroGrad()
		}
	}
}
func (l *LARS) AddParamGroup(g ParamGroup)                 { l.groups = append(l.groups, g) }
func (l *LARS) ParamGroups() []ParamGroup                  { return l.groups }
func (l *LARS) StateDict() map[string]interface{}          { return nil }
func (l *LARS) LoadStateDict(map[string]interface{}) error { return nil }
