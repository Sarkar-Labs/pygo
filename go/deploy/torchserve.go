package deploy

import "time"

// GeneratedTorchserve records execution state for the torchserve component.
type GeneratedTorchserve struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedTorchserve() *GeneratedTorchserve {
	return &GeneratedTorchserve{Name: "torchserve", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedTorchserve) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedTorchserve) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
