package transforms

import "time"

// GeneratedGridmask records execution state for the gridmask component.
type GeneratedGridmask struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGridmask() *GeneratedGridmask {
	return &GeneratedGridmask{Name: "gridmask", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGridmask) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGridmask) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
