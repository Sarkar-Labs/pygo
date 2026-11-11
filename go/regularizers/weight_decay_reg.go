package regularizers

import "time"

// GeneratedWeightDecayReg records execution state for the weight decay reg component.
type GeneratedWeightDecayReg struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWeightDecayReg() *GeneratedWeightDecayReg {
	return &GeneratedWeightDecayReg{Name: "weight_decay_reg", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWeightDecayReg) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWeightDecayReg) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
