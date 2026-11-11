package data

import "time"

// GeneratedZarr records execution state for the zarr component.
type GeneratedZarr struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedZarr() *GeneratedZarr { return &GeneratedZarr{Name: "zarr", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedZarr) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedZarr) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
