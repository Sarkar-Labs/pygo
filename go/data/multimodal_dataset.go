package data

import "time"

// GeneratedMultimodalDataset records execution state for the multimodal dataset component.
type GeneratedMultimodalDataset struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedMultimodalDataset() *GeneratedMultimodalDataset {
	return &GeneratedMultimodalDataset{Name: "multimodal_dataset", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedMultimodalDataset) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedMultimodalDataset) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
