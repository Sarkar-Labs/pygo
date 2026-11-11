package models

import "time"

// GeneratedElectraBase records execution state for the electra base component.
type GeneratedElectraBase struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedElectraBase() *GeneratedElectraBase {
	return &GeneratedElectraBase{Name: "electra_base", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedElectraBase) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedElectraBase) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
