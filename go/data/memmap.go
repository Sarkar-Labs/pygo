package data

import "time"

// GeneratedMemmap records execution state for the memmap component.
type GeneratedMemmap struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMemmap() *GeneratedMemmap {
	return &GeneratedMemmap{Name: "memmap", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMemmap) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMemmap) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
