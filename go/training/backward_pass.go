package training

import "time"

// GeneratedBackwardPass records execution state for the backward pass component.
type GeneratedBackwardPass struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBackwardPass() *GeneratedBackwardPass {
	return &GeneratedBackwardPass{Name: "backward_pass", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBackwardPass) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBackwardPass) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
