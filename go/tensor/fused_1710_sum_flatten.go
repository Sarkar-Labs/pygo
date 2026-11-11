package tensor

import "fmt"

func Fused1710SumFlatten(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sum_flatten: nil operand")
	}
	return x.Sum(0, false), nil
}
func Fused1710SumFlattenName() string { return "sum_flatten" }
