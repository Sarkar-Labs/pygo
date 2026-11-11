package callbacks

import "time"

// GeneratedNeptune records execution state for the neptune component.
type GeneratedNeptune struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedNeptune() *GeneratedNeptune {
	return &GeneratedNeptune{Name: "neptune", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedNeptune) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedNeptune) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
