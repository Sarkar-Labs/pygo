package tensor

import "fmt"

func Fused0486SqrtSubSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_sub_sin: nil operand")
	}
	r := x.Sub(y)
	r = r.Sqrt()
	r = r.Sin()
	return r, nil
}
func Fused0486SqrtSubSinName() string { return "sqrt_sub_sin" }
