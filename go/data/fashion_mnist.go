package data

import "time"

// GeneratedFashionMnist records execution state for the fashion mnist component.
type GeneratedFashionMnist struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedFashionMnist() *GeneratedFashionMnist {
	return &GeneratedFashionMnist{Name: "fashion_mnist", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedFashionMnist) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedFashionMnist) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
