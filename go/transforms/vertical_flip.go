package transforms

import "time"

// GeneratedVerticalFlip records execution state for the vertical flip component.
type GeneratedVerticalFlip struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedVerticalFlip() *GeneratedVerticalFlip {
	return &GeneratedVerticalFlip{Name: "vertical_flip", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedVerticalFlip) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedVerticalFlip) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
