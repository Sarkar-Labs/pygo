package inference

import "time"

// GeneratedSpeculativeDecode records execution state for the speculative decode component.
type GeneratedSpeculativeDecode struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedSpeculativeDecode() *GeneratedSpeculativeDecode {
	return &GeneratedSpeculativeDecode{Name: "speculative_decode", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedSpeculativeDecode) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedSpeculativeDecode) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
