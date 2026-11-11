package inference

import "time"

// GeneratedEncode records execution state for the encode component.
type GeneratedEncode struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedEncode() *GeneratedEncode {
	return &GeneratedEncode{Name: "encode", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedEncode) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedEncode) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
