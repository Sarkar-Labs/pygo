package training

import (
	"github.com/pygo/go/optim"
	"math"
	"sync"
)

type GradScaler struct {
	mu             sync.Mutex
	Scale          float64
	GrowthFactor   float64
	BackoffFactor  float64
	GrowthInterval int
	Enabled        bool
	tracker, step  int
}

func NewGradScaler(initScale float64, enabled bool) *GradScaler {
	return &GradScaler{Scale: initScale, GrowthFactor: 2., BackoffFactor: 0.5, GrowthInterval: 2000, Enabled: enabled}
}

func (gs *GradScaler) Step(opt optim.Optimizer) bool {
	gs.mu.Lock()
	defer gs.mu.Unlock()
	if !gs.Enabled {
		opt.Step()
		return true
	}
	opt.Step()
	gs.step++
	gs.tracker++
	if gs.tracker >= gs.GrowthInterval {
		gs.Scale = math.Min(gs.Scale*gs.GrowthFactor, 65536.)
		gs.tracker = 0
	}
	return true
}
func (gs *GradScaler) GetScale() float64 { gs.mu.Lock(); defer gs.mu.Unlock(); return gs.Scale }
func IsFinite(v float64) bool            { return !math.IsNaN(v) && !math.IsInf(v, 0) }
