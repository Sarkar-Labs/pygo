package transforms

import "time"

// GeneratedCenterCrop records execution state for the center crop component.
type GeneratedCenterCrop struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedCenterCrop() *GeneratedCenterCrop {
	return &GeneratedCenterCrop{Name: "center_crop", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedCenterCrop) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedCenterCrop) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
