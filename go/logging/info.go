package logging

import "time"

// GeneratedInfo records execution state for the info component.
type GeneratedInfo struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedInfo() *GeneratedInfo { return &GeneratedInfo{Name: "info", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedInfo) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedInfo) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
