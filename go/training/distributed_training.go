package training

import "time"

// GeneratedDistributedTraining records execution state for the distributed training component.
type GeneratedDistributedTraining struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedDistributedTraining() *GeneratedDistributedTraining {
	return &GeneratedDistributedTraining{Name: "distributed_training", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedDistributedTraining) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedDistributedTraining) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
