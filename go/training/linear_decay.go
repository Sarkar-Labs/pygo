package training

import "time"

// GeneratedLinearDecay records execution state for the linear decay component.
type GeneratedLinearDecay struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLinearDecay() *GeneratedLinearDecay {
	return &GeneratedLinearDecay{Name: "linear_decay", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLinearDecay) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLinearDecay) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
