package tensor

import "fmt"

func Fused1712SumUnsqueeze(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sum_unsqueeze: nil operand")
	}
	return x.Sum(0, false), nil
}
func Fused1712SumUnsqueezeName() string { return "sum_unsqueeze" }
