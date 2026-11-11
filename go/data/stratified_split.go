package data

import "time"

// GeneratedStratifiedSplit records execution state for the stratified split component.
type GeneratedStratifiedSplit struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedStratifiedSplit() *GeneratedStratifiedSplit {
	return &GeneratedStratifiedSplit{Name: "stratified_split", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedStratifiedSplit) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedStratifiedSplit) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
