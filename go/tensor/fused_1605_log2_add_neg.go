package tensor

import "fmt"

func Fused1605Log2AddNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_add_neg: nil operand")
	}
	r := x.Add(y)
	r = r.Log2()
	r = r.Neg()
	return r, nil
}
func Fused1605Log2AddNegName() string { return "log2_add_neg" }
