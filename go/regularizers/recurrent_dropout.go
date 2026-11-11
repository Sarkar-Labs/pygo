package regularizers

import "time"

// GeneratedRecurrentDropout records execution state for the recurrent dropout component.
type GeneratedRecurrentDropout struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedRecurrentDropout() *GeneratedRecurrentDropout {
	return &GeneratedRecurrentDropout{Name: "recurrent_dropout", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedRecurrentDropout) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedRecurrentDropout) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
