package config

import "time"

// GeneratedIni records execution state for the ini component.
type GeneratedIni struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedIni() *GeneratedIni { return &GeneratedIni{Name: "ini", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedIni) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedIni) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
