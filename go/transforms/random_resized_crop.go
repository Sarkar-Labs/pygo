package transforms

import "time"

// GeneratedRandomResizedCrop records execution state for the random resized crop component.
type GeneratedRandomResizedCrop struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedRandomResizedCrop() *GeneratedRandomResizedCrop {
	return &GeneratedRandomResizedCrop{Name: "random_resized_crop", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedRandomResizedCrop) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedRandomResizedCrop) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
