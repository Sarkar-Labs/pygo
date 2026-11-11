package training

import "time"

// GeneratedTestLoop records execution state for the test loop component.
type GeneratedTestLoop struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedTestLoop() *GeneratedTestLoop {
	return &GeneratedTestLoop{Name: "test_loop", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedTestLoop) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedTestLoop) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
