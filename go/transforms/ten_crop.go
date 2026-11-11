package transforms

import "time"

// GeneratedTenCrop records execution state for the ten crop component.
type GeneratedTenCrop struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedTenCrop() *GeneratedTenCrop {
	return &GeneratedTenCrop{Name: "ten_crop", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedTenCrop) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedTenCrop) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
