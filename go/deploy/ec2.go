package deploy

import "time"

// GeneratedEc2 records execution state for the ec2 component.
type GeneratedEc2 struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedEc2() *GeneratedEc2 { return &GeneratedEc2{Name: "ec2", CreatedAt: time.Now()} }

// Run records one completed operation and returns its sequence number.
func (c *GeneratedEc2) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedEc2) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
