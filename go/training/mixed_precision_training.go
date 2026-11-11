package training

import "time"

// GeneratedMixedPrecisionTraining records execution state for the mixed precision training component.
type GeneratedMixedPrecisionTraining struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMixedPrecisionTraining() *GeneratedMixedPrecisionTraining {
	return &GeneratedMixedPrecisionTraining{Name: "mixed_precision_training", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMixedPrecisionTraining) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMixedPrecisionTraining) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
