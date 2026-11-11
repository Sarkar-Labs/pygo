package data

import "time"

// GeneratedHdf5 records execution state for the hdf5 component.
type GeneratedHdf5 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedHdf5() *GeneratedHdf5 { return &GeneratedHdf5{Name: "hdf5", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedHdf5) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedHdf5) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
