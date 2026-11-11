package inference

import "time"

// GeneratedEagleDecode records execution state for the eagle decode component.
type GeneratedEagleDecode struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedEagleDecode() *GeneratedEagleDecode {
	return &GeneratedEagleDecode{Name: "eagle_decode", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedEagleDecode) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedEagleDecode) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
