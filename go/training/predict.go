package training

import "time"

// GeneratedPredict records execution state for the predict component.
type GeneratedPredict struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedPredict() *GeneratedPredict {
	return &GeneratedPredict{Name: "predict", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedPredict) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedPredict) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
