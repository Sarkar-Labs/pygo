package regularizers

import "time"

// GeneratedShakedrop records execution state for the shakedrop component.
type GeneratedShakedrop struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedShakedrop() *GeneratedShakedrop {
	return &GeneratedShakedrop{Name: "shakedrop", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedShakedrop) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedShakedrop) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
