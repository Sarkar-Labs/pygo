package data

import "time"

// GeneratedCifar10 records execution state for the cifar10 component.
type GeneratedCifar10 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedCifar10() *GeneratedCifar10 {
	return &GeneratedCifar10{Name: "cifar10", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedCifar10) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedCifar10) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
