package transforms

import "time"

// GeneratedColorJitter records execution state for the color jitter component.
type GeneratedColorJitter struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedColorJitter() *GeneratedColorJitter {
	return &GeneratedColorJitter{Name: "color_jitter", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedColorJitter) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedColorJitter) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
