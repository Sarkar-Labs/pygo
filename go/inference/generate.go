package inference

import "time"

// GeneratedGenerate records execution state for the generate component.
type GeneratedGenerate struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedGenerate() *GeneratedGenerate {
	return &GeneratedGenerate{Name: "generate", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedGenerate) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedGenerate) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
