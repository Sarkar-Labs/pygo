package transforms

import "time"

// GeneratedCutout records execution state for the cutout component.
type GeneratedCutout struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedCutout() *GeneratedCutout {
	return &GeneratedCutout{Name: "cutout", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedCutout) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedCutout) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
