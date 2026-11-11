package transforms

import "time"

// GeneratedCutmix records execution state for the cutmix component.
type GeneratedCutmix struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedCutmix() *GeneratedCutmix {
	return &GeneratedCutmix{Name: "cutmix", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedCutmix) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedCutmix) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
