package tensor

import "fmt"

func Fused0885CosAddNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_add_neg: nil operand")
	}
	r := x.Add(y)
	r = r.Cos()
	r = r.Neg()
	return r, nil
}
func Fused0885CosAddNegName() string { return "cos_add_neg" }
