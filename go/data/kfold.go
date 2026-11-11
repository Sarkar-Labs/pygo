package data

import "time"

// GeneratedKfold records execution state for the kfold component.
type GeneratedKfold struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedKfold() *GeneratedKfold {
	return &GeneratedKfold{Name: "kfold", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedKfold) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedKfold) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
