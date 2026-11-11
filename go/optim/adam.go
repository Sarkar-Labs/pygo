// Package optim provides gradient-based optimizers for PyGo.
package optim

import (
	"fmt"
	"github.com/pygo/go/nn"
	"github.com/pygo/go/tensor"
	"math"
	"sync"
)

// ─── Optimizer interface ──────────────────────────────────────────────────────

type Optimizer interface {
	Step()
	ZeroGrad()
	StateDict() map[string]interface{}
	LoadStateDict(map[string]interface{}) error
	AddParamGroup(group ParamGroup)
	ParamGroups() []ParamGroup
}

// ─── ParamGroup ───────────────────────────────────────────────────────────────

type ParamGroup struct {
	Params  []*nn.Parameter
	LR      float64
	WD      float64 // weight decay
	Options map[string]float64
}

// ─── Adam ────────────────────────────────────────────────────────────────────

// Adam implements Adam / AdamW optimiser.
// Adam:  m = β1·m + (1-β1)·g
//
//	v = β2·v + (1-β2)·g²
//	θ -= lr · m̂ / (√v̂ + ε)
//
// AdamW: adds decoupled weight decay: θ -= lr·wd·θ  (before grad step)
type Adam struct {
	mu         sync.Mutex
	groups     []ParamGroup
	Beta1      float64
	Beta2      float64
	Eps        float64
	Decoupled  bool // true → AdamW
	AMSGrad    bool
	step_count int64
	// per-param state
	m     map[*nn.Parameter]*tensor.Tensor // 1st moment
	v     map[*nn.Parameter]*tensor.Tensor // 2nd moment
	v_max map[*nn.Parameter]*tensor.Tensor // amsgrad max
}

func NewAdam(params []*nn.Parameter, lr, beta1, beta2, eps, wd float64,
	amsgrad, decoupled bool) *Adam {
	a := &Adam{
		Beta1:     beta1,
		Beta2:     beta2,
		Eps:       eps,
		Decoupled: decoupled,
		AMSGrad:   amsgrad,
		m:         make(map[*nn.Parameter]*tensor.Tensor),
		v:         make(map[*nn.Parameter]*tensor.Tensor),
		v_max:     make(map[*nn.Parameter]*tensor.Tensor),
	}
	a.groups = []ParamGroup{{Params: params, LR: lr, WD: wd}}
	return a
}

func NewAdamW(params []*nn.Parameter, lr float64, opts ...AdamOption) *Adam {
	cfg := adamDefaults()
	for _, o := range opts {
		o(&cfg)
	}
	return NewAdam(params, lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.wd, cfg.amsgrad, true)
}

func (a *Adam) ZeroGrad() {
	for _, g := range a.groups {
		for _, p := range g.Params {
			p.ZeroGrad()
		}
	}
}

func (a *Adam) Step() {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.step_count++
	t := float64(a.step_count)

	// Bias correction factors
	bc1 := 1.0 - math.Pow(a.Beta1, t)
	bc2 := 1.0 - math.Pow(a.Beta2, t)

	for _, group := range a.groups {
		lr := group.LR
		wd := group.WD

		for _, p := range group.Params {
			g := p.Grad()
			if g == nil {
				continue
			}

			// Decoupled weight decay (AdamW)
			if a.Decoupled && wd != 0 {
				// θ ← θ · (1 - lr·wd)
				p.Tensor = p.MulScalar(1.0 - lr*wd)
			}

			// Lazy state init
			if _, ok := a.m[p]; !ok {
				a.m[p] = tensor.ZerosLike(p.Tensor)
				a.v[p] = tensor.ZerosLike(p.Tensor)
				if a.AMSGrad {
					a.v_max[p] = tensor.ZerosLike(p.Tensor)
				}
			}

			// m ← β1·m + (1-β1)·g
			a.m[p] = a.m[p].MulScalar(a.Beta1).Add(g.MulScalar(1.0 - a.Beta1))

			// v ← β2·v + (1-β2)·g²
			a.v[p] = a.v[p].MulScalar(a.Beta2).Add(g.Pow(2).MulScalar(1.0 - a.Beta2))

			// Bias-corrected moments
			m_hat := a.m[p].MulScalar(1.0 / bc1)
			v_hat := a.v[p].MulScalar(1.0 / bc2)

			denom := v_hat
			if a.AMSGrad {
				// v_max <- max(v_max, v_hat), element by element.
				// The stable Go tensor ABI currently exposes no elementwise maximum,
				// so perform the operation on the CPU float32 view rather than silently
				// assigning v_hat (which would reduce AMSGrad to ordinary Adam).
				if p.Tensor.Device().Kind != tensor.CPU || v_hat.DType() != tensor.Float32 {
					panic("optim.Adam: AMSGrad requires CPU float32 tensors until the native elementwise maximum ABI is exposed")
				}
				old := a.v_max[p]
				oldData, newData := old.Float32Slice(), v_hat.Float32Slice()
				if len(oldData) != len(newData) {
					panic("optim.Adam: AMSGrad state shape mismatch")
				}
				maxData := make([]float32, len(oldData))
				for i := range maxData {
					if newData[i] > oldData[i] {
						maxData[i] = newData[i]
					} else {
						maxData[i] = oldData[i]
					}
				}
				a.v_max[p] = tensor.FromSlice(maxData, old.Shape(), tensor.WithDType(tensor.Float32), tensor.OnCPU())
				denom = a.v_max[p]
			}

			// θ ← θ - lr · m̂ / (√v̂ + ε)
			step := m_hat.Div(denom.Sqrt().AddScalar(a.Eps))
			p.Tensor = p.Sub(step.MulScalar(lr))

			// L2 weight decay (classical Adam)
			if !a.Decoupled && wd != 0 {
				p.Tensor = p.Sub(p.MulScalar(lr * wd))
			}
		}
	}
}

func (a *Adam) AddParamGroup(g ParamGroup) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.groups = append(a.groups, g)
}

func (a *Adam) ParamGroups() []ParamGroup { return a.groups }

func (a *Adam) StateDict() map[string]interface{} {
	return map[string]interface{}{
		"step":  a.step_count,
		"beta1": a.Beta1,
		"beta2": a.Beta2,
		"eps":   a.Eps,
	}
}

func (a *Adam) LoadStateDict(sd map[string]interface{}) error {
	if v, ok := sd["step"]; ok {
		a.step_count = int64(v.(float64))
	}
	return nil
}

func (a *Adam) String() string {
	return fmt.Sprintf("Adam(β1=%.3f β2=%.3f ε=%.0e decoupled=%v)",
		a.Beta1, a.Beta2, a.Eps, a.Decoupled)
}

// ─── Adam options ─────────────────────────────────────────────────────────────

type adamCfg struct {
	beta1, beta2, eps, wd float64
	amsgrad               bool
}

func adamDefaults() adamCfg {
	return adamCfg{beta1: 0.9, beta2: 0.999, eps: 1e-8, wd: 0.01}
}

type AdamOption func(*adamCfg)

func WithBeta1(b float64) AdamOption { return func(c *adamCfg) { c.beta1 = b } }
func WithBeta2(b float64) AdamOption { return func(c *adamCfg) { c.beta2 = b } }
func WithEps(e float64) AdamOption   { return func(c *adamCfg) { c.eps = e } }
func WithWD(w float64) AdamOption    { return func(c *adamCfg) { c.wd = w } }
func WithAMSGrad() AdamOption        { return func(c *adamCfg) { c.amsgrad = true } }

// ─── SGD ────────────────────────────────────────────────────────────────────

type SGD struct {
	mu        sync.Mutex
	groups    []ParamGroup
	Momentum  float64
	Dampening float64
	Nesterov  bool
	velocity  map[*nn.Parameter]*tensor.Tensor
}

func NewSGD(params []*nn.Parameter, lr, momentum, wd float64, nesterov bool) *SGD {
	return &SGD{
		groups:   []ParamGroup{{Params: params, LR: lr, WD: wd}},
		Momentum: momentum,
		Nesterov: nesterov,
		velocity: make(map[*nn.Parameter]*tensor.Tensor),
	}
}

func (s *SGD) ZeroGrad() {
	for _, g := range s.groups {
		for _, p := range g.Params {
			p.ZeroGrad()
		}
	}
}

func (s *SGD) Step() {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, group := range s.groups {
		lr := group.LR
		wd := group.WD
		for _, p := range group.Params {
			g := p.Grad()
			if g == nil {
				continue
			}

			// L2 regularization
			if wd != 0 {
				g = g.Add(p.MulScalar(wd))
			}

			if s.Momentum != 0 {
				if _, ok := s.velocity[p]; !ok {
					s.velocity[p] = g.Clone()
				} else {
					s.velocity[p] = s.velocity[p].MulScalar(s.Momentum).Add(
						g.MulScalar(1 - s.Dampening))
				}
				if s.Nesterov {
					g = g.Add(s.velocity[p].MulScalar(s.Momentum))
				} else {
					g = s.velocity[p]
				}
			}
			p.Tensor = p.Sub(g.MulScalar(lr))
		}
	}
}

func (s *SGD) AddParamGroup(g ParamGroup) { s.groups = append(s.groups, g) }
func (s *SGD) ParamGroups() []ParamGroup  { return s.groups }
func (s *SGD) StateDict() map[string]interface{} {
	return map[string]interface{}{"momentum": s.Momentum}
}
func (s *SGD) LoadStateDict(sd map[string]interface{}) error { return nil }

// ─── Lion ───────────────────────────────────────────────────────────────────

// Lion (EvoLved Sign Momentum) — memory-efficient optimizer from Google Brain.
// Update: m ← β1·m + (1-β1)·g
//
//	θ ← θ - lr · (sign(m) + wd·θ)
type Lion struct {
	mu     sync.Mutex
	groups []ParamGroup
	Beta1  float64
	Beta2  float64
	m      map[*nn.Parameter]*tensor.Tensor
}

func NewLion(params []*nn.Parameter, lr, beta1, beta2, wd float64) *Lion {
	return &Lion{
		groups: []ParamGroup{{Params: params, LR: lr, WD: wd}},
		Beta1:  beta1,
		Beta2:  beta2,
		m:      make(map[*nn.Parameter]*tensor.Tensor),
	}
}

func (l *Lion) Step() {
	l.mu.Lock()
	defer l.mu.Unlock()
	for _, group := range l.groups {
		lr := group.LR
		wd := group.WD
		for _, p := range group.Params {
			g := p.Grad()
			if g == nil {
				continue
			}

			if _, ok := l.m[p]; !ok {
				l.m[p] = tensor.ZerosLike(p.Tensor)
			}
			// update direction = sign(β1·m + (1-β1)·g)
			interp := l.m[p].MulScalar(l.Beta1).Add(g.MulScalar(1 - l.Beta1))
			update := interp.Sign()

			// weight decay
			if wd != 0 {
				update = update.Add(p.MulScalar(wd))
			}
			p.Tensor = p.Sub(update.MulScalar(lr))

			// update momentum with β2
			l.m[p] = l.m[p].MulScalar(l.Beta2).Add(g.MulScalar(1 - l.Beta2))
		}
	}
}

func (l *Lion) ZeroGrad() {
	for _, g := range l.groups {
		for _, p := range g.Params {
			p.ZeroGrad()
		}
	}
}
func (l *Lion) AddParamGroup(g ParamGroup)                 { l.groups = append(l.groups, g) }
func (l *Lion) ParamGroups() []ParamGroup                  { return l.groups }
func (l *Lion) StateDict() map[string]interface{}          { return map[string]interface{}{"beta1": l.Beta1} }
func (l *Lion) LoadStateDict(map[string]interface{}) error { return nil }

// ─── LR Scheduler interface ───────────────────────────────────────────────────

type Scheduler interface {
	Step()
	GetLR() []float64
	StateDict() map[string]interface{}
}

// ─── CosineAnnealingLR ───────────────────────────────────────────────────────

type CosineAnnealingLR struct {
	opt      Optimizer
	T_max    int
	eta_min  float64
	base_lrs []float64
	step_n   int
}

func NewCosineAnnealingLR(opt Optimizer, T_max int, eta_min float64) *CosineAnnealingLR {
	lrs := make([]float64, len(opt.ParamGroups()))
	for i, g := range opt.ParamGroups() {
		lrs[i] = g.LR
	}
	return &CosineAnnealingLR{opt: opt, T_max: T_max, eta_min: eta_min, base_lrs: lrs}
}

func (c *CosineAnnealingLR) Step() {
	c.step_n++
	for i := range c.opt.ParamGroups() {
		t := float64(c.step_n % (2 * c.T_max))
		lr := c.eta_min + 0.5*(c.base_lrs[i]-c.eta_min)*(1+math.Cos(math.Pi*t/float64(c.T_max)))
		c.opt.ParamGroups()[i].LR = lr
	}
}

func (c *CosineAnnealingLR) GetLR() []float64 {
	lrs := make([]float64, len(c.opt.ParamGroups()))
	for i, g := range c.opt.ParamGroups() {
		lrs[i] = g.LR
	}
	return lrs
}

func (c *CosineAnnealingLR) StateDict() map[string]interface{} {
	return map[string]interface{}{"step": c.step_n}
}

// ─── WarmupCosine ─────────────────────────────────────────────────────────────

type WarmupCosine struct {
	opt          Optimizer
	warmup_steps int
	total_steps  int
	base_lrs     []float64
	min_lrs      []float64
	step_n       int
}

func NewWarmupCosine(opt Optimizer, warmupSteps, totalSteps int) *WarmupCosine {
	base := make([]float64, len(opt.ParamGroups()))
	mins := make([]float64, len(opt.ParamGroups()))
	for i, g := range opt.ParamGroups() {
		base[i] = g.LR
		mins[i] = g.LR * 0.1
	}
	return &WarmupCosine{
		opt: opt, warmup_steps: warmupSteps, total_steps: totalSteps,
		base_lrs: base, min_lrs: mins,
	}
}

func (w *WarmupCosine) Step() {
	w.step_n++
	n := w.step_n
	for i := range w.opt.ParamGroups() {
		var lr float64
		if n <= w.warmup_steps {
			lr = w.base_lrs[i] * float64(n) / float64(w.warmup_steps)
		} else {
			progress := float64(n-w.warmup_steps) / float64(w.total_steps-w.warmup_steps)
			lr = w.min_lrs[i] + 0.5*(w.base_lrs[i]-w.min_lrs[i])*(1+math.Cos(math.Pi*progress))
		}
		w.opt.ParamGroups()[i].LR = lr
	}
}

func (w *WarmupCosine) GetLR() []float64 {
	lrs := make([]float64, len(w.opt.ParamGroups()))
	for i, g := range w.opt.ParamGroups() {
		lrs[i] = g.LR
	}
	return lrs
}

func (w *WarmupCosine) StateDict() map[string]interface{} {
	return map[string]interface{}{"step": w.step_n}
}
