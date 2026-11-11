package transforms

import "time"

// GeneratedStandardize records execution state for the standardize component.
type GeneratedStandardize struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedStandardize() *GeneratedStandardize {
	return &GeneratedStandardize{Name: "standardize", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedStandardize) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedStandardize) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
