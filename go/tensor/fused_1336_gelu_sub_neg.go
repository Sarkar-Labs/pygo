package tensor

import "fmt"

func Fused1336GeluSubNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_sub_neg: nil operand")
	}
	r := x.Sub(y)
	r = r.GELU()
	r = r.Neg()
	return r, nil
}
func Fused1336GeluSubNegName() string { return "gelu_sub_neg" }
