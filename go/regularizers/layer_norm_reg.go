package regularizers

import "time"

// GeneratedLayerNormReg records execution state for the layer norm reg component.
type GeneratedLayerNormReg struct {
	Name      string
	CreatedAt time.Time
	Runs      uint64
}

func NewGeneratedLayerNormReg() *GeneratedLayerNormReg {
	return &GeneratedLayerNormReg{Name: "layer_norm_reg", CreatedAt: time.Now()}
}

// Run records one completed operation and returns its sequence number.
func (c *GeneratedLayerNormReg) Run() uint64 { c.Runs++; return c.Runs }

// Ready reports whether the component has been initialized.
func (c *GeneratedLayerNormReg) Ready() bool { return c != nil && !c.CreatedAt.IsZero() }
