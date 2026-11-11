package data

import "time"

// GeneratedParquet records execution state for the parquet component.
type GeneratedParquet struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedParquet() *GeneratedParquet {
	return &GeneratedParquet{Name: "parquet", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedParquet) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedParquet) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
