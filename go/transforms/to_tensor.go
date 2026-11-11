package transforms

import "time"

// GeneratedToTensor records execution state for the to tensor component.
type GeneratedToTensor struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedToTensor() *GeneratedToTensor {
	return &GeneratedToTensor{Name: "to_tensor", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedToTensor) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedToTensor) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
