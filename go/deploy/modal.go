package deploy

import "time"

// GeneratedModal records execution state for the modal component.
type GeneratedModal struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedModal() *GeneratedModal {
	return &GeneratedModal{Name: "modal", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedModal) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedModal) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
