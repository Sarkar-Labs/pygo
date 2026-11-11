package data

import "time"

// GeneratedCifar100 records execution state for the cifar100 component.
type GeneratedCifar100 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedCifar100() *GeneratedCifar100 {
	return &GeneratedCifar100{Name: "cifar100", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedCifar100) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedCifar100) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
