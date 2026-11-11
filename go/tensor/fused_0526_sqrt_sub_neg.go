package tensor

import "fmt"

func Fused0526SqrtSubNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("sqrt_sub_neg: nil operand")
	}
	r := x.Sub(y)
	r = r.Sqrt()
	r = r.Neg()
	return r, nil
}
func Fused0526SqrtSubNegName() string { return "sqrt_sub_neg" }
