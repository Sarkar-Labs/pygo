package tensor

import "fmt"

func Fused1713SumClone(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sum_clone: nil operand")
	}
	return x.Sum(0, false), nil
}
func Fused1713SumCloneName() string { return "sum_clone" }
