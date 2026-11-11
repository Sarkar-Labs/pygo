package transforms

import "time"

// GeneratedCenterPad records execution state for the center pad component.
type GeneratedCenterPad struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedCenterPad() *GeneratedCenterPad {
	return &GeneratedCenterPad{Name: "center_pad", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedCenterPad) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedCenterPad) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
