package training

import "time"

// GeneratedActivationCheckpointing records execution state for the activation checkpointing component.
type GeneratedActivationCheckpointing struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedActivationCheckpointing() *GeneratedActivationCheckpointing {
	return &GeneratedActivationCheckpointing{Name: "activation_checkpointing", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedActivationCheckpointing) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedActivationCheckpointing) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
