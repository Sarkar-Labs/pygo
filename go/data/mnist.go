package data

import "time"

// GeneratedMnist records execution state for the mnist component.
type GeneratedMnist struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMnist() *GeneratedMnist {
	return &GeneratedMnist{Name: "mnist", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMnist) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMnist) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
