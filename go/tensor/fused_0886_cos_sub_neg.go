package tensor

import "fmt"

func Fused0886CosSubNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_sub_neg: nil operand")
	}
	r := x.Sub(y)
	r = r.Cos()
	r = r.Neg()
	return r, nil
}
func Fused0886CosSubNegName() string { return "cos_sub_neg" }
