package tensor

import "fmt"

func Fused0396AbsSubSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_sub_sin: nil operand")
	}
	r := x.Sub(y)
	r = r.Abs()
	r = r.Sin()
	return r, nil
}
func Fused0396AbsSubSinName() string { return "abs_sub_sin" }
