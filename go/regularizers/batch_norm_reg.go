package regularizers

import "time"

// GeneratedBatchNormReg records execution state for the batch norm reg component.
type GeneratedBatchNormReg struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedBatchNormReg() *GeneratedBatchNormReg {
	return &GeneratedBatchNormReg{Name: "batch_norm_reg", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedBatchNormReg) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedBatchNormReg) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
