package callbacks

import "time"

// GeneratedComet records execution state for the comet component.
type GeneratedComet struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedComet() *GeneratedComet {
	return &GeneratedComet{Name: "comet", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedComet) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedComet) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
