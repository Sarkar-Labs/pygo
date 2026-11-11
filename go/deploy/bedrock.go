package deploy

import "time"

// GeneratedBedrock records execution state for the bedrock component.
type GeneratedBedrock struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBedrock() *GeneratedBedrock {
	return &GeneratedBedrock{Name: "bedrock", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBedrock) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBedrock) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
