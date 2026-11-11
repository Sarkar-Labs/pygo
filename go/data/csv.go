package data

import "time"

// GeneratedCsv records execution state for the csv component.
type GeneratedCsv struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedCsv() *GeneratedCsv { return &GeneratedCsv{Name: "csv", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedCsv) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedCsv) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
