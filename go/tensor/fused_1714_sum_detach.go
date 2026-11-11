package tensor

import "fmt"

func Fused1714SumDetach(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sum_detach: nil operand")
	}
	return x.Sum(0, false), nil
}
func Fused1714SumDetachName() string { return "sum_detach" }
