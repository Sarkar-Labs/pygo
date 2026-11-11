package training

import "time"

// GeneratedGradientClipping records execution state for the gradient clipping component.
type GeneratedGradientClipping struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGradientClipping() *GeneratedGradientClipping {
	return &GeneratedGradientClipping{Name: "gradient_clipping", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGradientClipping) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGradientClipping) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
