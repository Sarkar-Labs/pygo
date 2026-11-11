package models

import "time"

// GeneratedYoloPruned records execution state for the yolo pruned component.
type GeneratedYoloPruned struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedYoloPruned() *GeneratedYoloPruned {
	return &GeneratedYoloPruned{Name: "yolo_pruned", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedYoloPruned) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedYoloPruned) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
