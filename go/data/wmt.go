package data

import "time"

// GeneratedWmt records execution state for the wmt component.
type GeneratedWmt struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedWmt() *GeneratedWmt { return &GeneratedWmt{Name: "wmt", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedWmt) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedWmt) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
