package tensor

import "fmt"

func Fused1606Log2SubNeg(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_sub_neg: nil operand")
	}
	r := x.Sub(y)
	r = r.Log2()
	r = r.Neg()
	return r, nil
}
func Fused1606Log2SubNegName() string { return "log2_sub_neg" }
