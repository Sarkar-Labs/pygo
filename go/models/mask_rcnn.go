package models

import "time"

// GeneratedMaskRcnn records execution state for the mask rcnn component.
type GeneratedMaskRcnn struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMaskRcnn() *GeneratedMaskRcnn {
	return &GeneratedMaskRcnn{Name: "mask_rcnn", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMaskRcnn) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMaskRcnn) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
