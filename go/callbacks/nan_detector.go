package callbacks

import "time"

// GeneratedNanDetector records execution state for the nan detector component.
type GeneratedNanDetector struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedNanDetector() *GeneratedNanDetector {
	return &GeneratedNanDetector{Name: "nan_detector", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedNanDetector) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedNanDetector) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
