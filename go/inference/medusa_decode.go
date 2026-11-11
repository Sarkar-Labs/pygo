package inference

import "time"

// GeneratedMedusaDecode records execution state for the medusa decode component.
type GeneratedMedusaDecode struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMedusaDecode() *GeneratedMedusaDecode {
	return &GeneratedMedusaDecode{Name: "medusa_decode", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMedusaDecode) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMedusaDecode) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
