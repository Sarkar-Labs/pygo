package inference

import "time"

// GeneratedBeamSearch records execution state for the beam search component.
type GeneratedBeamSearch struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBeamSearch() *GeneratedBeamSearch {
	return &GeneratedBeamSearch{Name: "beam_search", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBeamSearch) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBeamSearch) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
