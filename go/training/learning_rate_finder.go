package training

import "time"

// GeneratedLearningRateFinder records execution state for the learning rate finder component.
type GeneratedLearningRateFinder struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLearningRateFinder() *GeneratedLearningRateFinder {
	return &GeneratedLearningRateFinder{Name: "learning_rate_finder", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLearningRateFinder) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLearningRateFinder) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
