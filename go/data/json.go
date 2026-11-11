package data

import "time"

// GeneratedJson records execution state for the json component.
type GeneratedJson struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedJson() *GeneratedJson { return &GeneratedJson{Name: "json", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedJson) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedJson) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
