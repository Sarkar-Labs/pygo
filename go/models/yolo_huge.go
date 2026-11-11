package models

import "time"

// GeneratedYoloHuge records execution state for the yolo huge component.
type GeneratedYoloHuge struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedYoloHuge() *GeneratedYoloHuge {
	return &GeneratedYoloHuge{Name: "yolo_huge", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedYoloHuge) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedYoloHuge) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
