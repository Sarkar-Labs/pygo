package training

import "time"

// GeneratedFit records execution state for the fit component.
type GeneratedFit struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedFit() *GeneratedFit { return &GeneratedFit{Name: "fit", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedFit) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedFit) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
