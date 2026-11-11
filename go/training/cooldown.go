package training

import "time"

// GeneratedCooldown records execution state for the cooldown component.
type GeneratedCooldown struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedCooldown() *GeneratedCooldown {
	return &GeneratedCooldown{Name: "cooldown", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedCooldown) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedCooldown) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
