package transforms

import "time"

// GeneratedToPil records execution state for the to pil component.
type GeneratedToPil struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedToPil() *GeneratedToPil {
	return &GeneratedToPil{Name: "to_pil", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedToPil) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedToPil) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
