package transforms

import "time"

// GeneratedResize records execution state for the resize component.
type GeneratedResize struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedResize() *GeneratedResize {
	return &GeneratedResize{Name: "resize", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedResize) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedResize) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
