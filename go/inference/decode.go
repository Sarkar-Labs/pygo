package inference

import "time"

// GeneratedDecode records execution state for the decode component.
type GeneratedDecode struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDecode() *GeneratedDecode {
	return &GeneratedDecode{Name: "decode", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDecode) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDecode) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
