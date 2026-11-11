package tensor

import "fmt"

func Fused1711SumSqueeze(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sum_squeeze: nil operand")
	}
	return x.Sum(0, false), nil
}
func Fused1711SumSqueezeName() string { return "sum_squeeze" }
