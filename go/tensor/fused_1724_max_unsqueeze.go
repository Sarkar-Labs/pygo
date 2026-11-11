package tensor

import "fmt"

func Fused1724MaxUnsqueeze(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("max_unsqueeze: nil operand")
	}
	return x.Max(0, false), nil
}
func Fused1724MaxUnsqueezeName() string { return "max_unsqueeze" }
