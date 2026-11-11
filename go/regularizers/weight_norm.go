package regularizers

import "time"

// GeneratedWeightNorm records execution state for the weight norm component.
type GeneratedWeightNorm struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWeightNorm() *GeneratedWeightNorm {
	return &GeneratedWeightNorm{Name: "weight_norm", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWeightNorm) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWeightNorm) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
