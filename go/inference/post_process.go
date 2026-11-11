package inference

import "time"

// GeneratedPostProcess records execution state for the post process component.
type GeneratedPostProcess struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedPostProcess() *GeneratedPostProcess {
	return &GeneratedPostProcess{Name: "post_process", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedPostProcess) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedPostProcess) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
