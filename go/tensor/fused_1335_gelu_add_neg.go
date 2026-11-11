package tensor

import "fmt"

func Fused1335GeluAddNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_add_neg: nil operand")
	}
	r := x.Add(y)
	r = r.GELU()
	r = r.Neg()
	return r, nil
}
func Fused1335GeluAddNegName() string { return "gelu_add_neg" }
