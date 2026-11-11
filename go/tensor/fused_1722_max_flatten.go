package tensor

import "fmt"

func Fused1722MaxFlatten(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("max_flatten: nil operand")
	}
	return x.Max(0, false), nil
}
func Fused1722MaxFlattenName() string { return "max_flatten" }
