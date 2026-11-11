package training

import "time"

// GeneratedForwardPass records execution state for the forward pass component.
type GeneratedForwardPass struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedForwardPass() *GeneratedForwardPass {
	return &GeneratedForwardPass{Name: "forward_pass", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedForwardPass) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedForwardPass) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
