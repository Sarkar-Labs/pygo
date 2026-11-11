package inference

import "time"

// GeneratedLogitsProcessor records execution state for the logits processor component.
type GeneratedLogitsProcessor struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLogitsProcessor() *GeneratedLogitsProcessor {
	return &GeneratedLogitsProcessor{Name: "logits_processor", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLogitsProcessor) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLogitsProcessor) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
