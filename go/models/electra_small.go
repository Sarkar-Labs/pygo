package models

import "time"

// GeneratedElectraSmall records execution state for the electra small component.
type GeneratedElectraSmall struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedElectraSmall() *GeneratedElectraSmall {
	return &GeneratedElectraSmall{Name: "electra_small", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedElectraSmall) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedElectraSmall) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
