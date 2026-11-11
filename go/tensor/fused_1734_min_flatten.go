package tensor

import "fmt"

func Fused1734MinFlatten(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("min_flatten: nil operand")
	}
	return x, nil
}
func Fused1734MinFlattenName() string { return "min_flatten" }
